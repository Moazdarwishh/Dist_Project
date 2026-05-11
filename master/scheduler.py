"""Master scheduler with metrics and network heartbeat."""

from __future__ import annotations

import threading
import time
from collections import Counter
from typing import List

from common.models import Request, Response


def _percentile(values: list, p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * p
    f = int(k)
    c = min(f + 1, len(s) - 1)
    if f == c:
        return s[f]
    return s[f] + (s[c] - s[f]) * (k - f)


class Scheduler:
    """Single front door used by the FastAPI scheduler endpoint."""

    def __init__(self, load_balancer, heartbeat_interval: float = 5.0):
        self.lb = load_balancer
        self.heartbeat_interval = heartbeat_interval
        self.started_at = time.time()

        self._metrics_lock = threading.Lock()
        self.total_requests = 0
        self.successful = 0
        self.failed = 0
        self._latencies: List[float] = []
        self._worker_counts: Counter = Counter()

        self._stop = threading.Event()
        self._hb_thread = threading.Thread(
            target=self._heartbeat_loop,
            daemon=True,
            name="scheduler-heartbeat",
        )
        self._hb_thread.start()

    def handle_request(self, request: Request) -> Response:
        response = self.lb.dispatch(request)
        with self._metrics_lock:
            self.total_requests += 1
            if response.success:
                self.successful += 1
                self._latencies.append(response.latency)
                if response.worker_id is not None:
                    self._worker_counts[response.worker_id] += 1
            else:
                self.failed += 1
        return response

    def report(self) -> dict:
        with self._metrics_lock:
            total = self.total_requests
            success = self.successful
            failed = self.failed
            latencies = list(self._latencies)
            worker_counts = dict(sorted(self._worker_counts.items()))

        elapsed = max(time.time() - self.started_at, 1e-9)
        avg = sum(latencies) / len(latencies) if latencies else 0.0
        return {
            "total_requests": total,
            "successful_requests": success,
            "failed_requests": failed,
            "success_rate": round(success / total, 4) if total else 0.0,
            "avg_latency_s": round(avg, 4),
            "p50_latency_s": round(_percentile(latencies, 0.50), 4),
            "p95_latency_s": round(_percentile(latencies, 0.95), 4),
            "throughput_req_s": round(total / elapsed, 4),
            "per_worker_request_count": worker_counts,
            "uptime_s": round(elapsed, 2),
        }

    def worker_summary(self) -> dict:
        summary = {}
        for worker in self.lb.workers:
            summary[worker.id] = {
                "url": worker.url,
                "healthy": worker.is_healthy(),
                "active_requests": worker.active_requests(),
                "avg_latency_s": round(worker.avg_latency(), 4),
            }
        return summary

    def stop(self) -> None:
        self._stop.set()
        self._hb_thread.join(timeout=self.heartbeat_interval + 1)
        self.lb.close()

    def _heartbeat_loop(self) -> None:
        while not self._stop.is_set():
            for worker in self.lb.workers:
                worker.heartbeat()
            self._stop.wait(self.heartbeat_interval)
