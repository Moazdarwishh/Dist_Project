"""
HTTP load balancer for FastAPI worker services.

Each worker is represented by a WorkerProxy. The proxy sends real HTTP
requests to the worker service and keeps local health/load metrics for
fast routing decisions.
"""

from __future__ import annotations

import itertools
import threading
import time
from collections import deque
from typing import List, Optional

import httpx

from common.models import Request, Response


class WorkerProxy:
    """Client-side proxy for one remote FastAPI worker."""

    def __init__(self, worker_id: int, url: str, timeout: float = 120.0):
        self.id = worker_id
        self.url = url.rstrip("/")
        self._healthy = True
        self._health_lock = threading.Lock()
        self._active = 0
        self._recent_latencies = deque(maxlen=50)
        self._metrics_lock = threading.Lock()
        self._client = httpx.Client(timeout=timeout)

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------
    def is_healthy(self) -> bool:
        with self._health_lock:
            return self._healthy

    def mark_unhealthy(self) -> None:
        with self._health_lock:
            self._healthy = False
        print(f"[Proxy {self.id}] marked UNHEALTHY ({self.url})")

    def mark_healthy(self) -> None:
        with self._health_lock:
            self._healthy = True

    def heartbeat(self) -> bool:
        """Call worker GET /health and update proxy health."""
        try:
            resp = self._client.get(f"{self.url}/health", timeout=3.0)
            if resp.status_code == 200:
                data = resp.json()
                if data.get("healthy", True):
                    self.mark_healthy()
                    return True
        except Exception as exc:
            print(f"[Proxy {self.id}] heartbeat failed: {exc}")
        self.mark_unhealthy()
        return False

    # ------------------------------------------------------------------
    # Load metrics
    # ------------------------------------------------------------------
    def active_requests(self) -> int:
        with self._metrics_lock:
            return self._active

    def avg_latency(self) -> float:
        with self._metrics_lock:
            if not self._recent_latencies:
                return 0.0
            return sum(self._recent_latencies) / len(self._recent_latencies)

    # ------------------------------------------------------------------
    # Remote processing
    # ------------------------------------------------------------------
    def process(self, request: Request) -> Response:
        if not self.is_healthy():
            raise RuntimeError(f"Worker {self.id} is unhealthy")

        with self._metrics_lock:
            self._active += 1

        wall_start = time.time()
        try:
            resp = self._client.post(
                f"{self.url}/process",
                json={"id": request.id, "query": request.query},
            )
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPStatusError as exc:
            raise RuntimeError(f"Worker {self.id} returned HTTP {exc.response.status_code}") from exc
        except (httpx.TimeoutException, httpx.ConnectError, httpx.NetworkError) as exc:
            raise RuntimeError(f"Worker {self.id} unreachable: {exc}") from exc
        finally:
            with self._metrics_lock:
                self._active -= 1

        wall_latency = time.time() - wall_start
        with self._metrics_lock:
            self._recent_latencies.append(wall_latency)

        # Use scheduler-side id so per-worker counts are stable even if the
        # worker process uses a PID fallback as its own WORKER_ID.
        return Response(
            id=int(data.get("id", request.id)),
            result=data.get("result", ""),
            latency=float(data.get("latency", wall_latency)),
            worker_id=self.id,
            success=bool(data.get("success", True)),
            error=data.get("error"),
            sources=data.get("sources") or [],
        )

    def close(self) -> None:
        self._client.close()


class LoadBalancer:
    """Routes requests to healthy workers using the selected strategy."""

    def __init__(self, workers: List[WorkerProxy], strategy: str = "round_robin"):
        if not workers:
            raise ValueError("LoadBalancer needs at least one worker")
        if strategy not in {"round_robin", "least_connections", "load_aware"}:
            raise ValueError("strategy must be round_robin, least_connections, or load_aware")
        self.workers = workers
        self.strategy = strategy
        self._rr_iter = itertools.cycle(range(len(workers)))
        self._rr_lock = threading.Lock()

    def _pick_round_robin(self) -> Optional[WorkerProxy]:
        with self._rr_lock:
            for _ in range(len(self.workers)):
                worker = self.workers[next(self._rr_iter)]
                if worker.is_healthy():
                    return worker
        return None

    def _pick_least_connections(self) -> Optional[WorkerProxy]:
        healthy = [w for w in self.workers if w.is_healthy()]
        return min(healthy, key=lambda w: w.active_requests()) if healthy else None

    def _pick_load_aware(self) -> Optional[WorkerProxy]:
        healthy = [w for w in self.workers if w.is_healthy()]
        if not healthy:
            return None
        return min(healthy, key=lambda w: (w.active_requests() + 1) * (w.avg_latency() + 1e-3))

    def _pick(self) -> Optional[WorkerProxy]:
        if self.strategy == "round_robin":
            return self._pick_round_robin()
        if self.strategy == "least_connections":
            return self._pick_least_connections()
        return self._pick_load_aware()

    def dispatch(self, request: Request, max_retries: int = 2) -> Response:
        """Send request to a worker and retry on another worker if it fails."""
        tried_ids = set()
        last_err = None

        for _ in range(max_retries + 1):
            worker = self._pick()
            if worker is None:
                last_err = "No healthy workers available"
                break
            if worker.id in tried_ids:
                continue
            tried_ids.add(worker.id)

            try:
                response = worker.process(request)
                # If the worker returned a normal JSON failure because RAG DB
                # is empty, do not mark the worker unhealthy. This is a data
                # problem, not a node failure.
                return response
            except Exception as exc:
                last_err = str(exc)
                worker.mark_unhealthy()
                print(f"[LB] worker {worker.id} failed: {exc}")

        return Response(
            id=request.id,
            result="",
            latency=0.0,
            worker_id=None,
            success=False,
            error=last_err or "dispatch failed",
        )

    def close(self) -> None:
        for worker in self.workers:
            worker.close()
