"""
Worker core used by the FastAPI worker service.

A worker owns health state, metrics, and the request processing flow:
    query -> ChromaDB RAG retrieval -> FLAN-T5 generation -> Response
"""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Any, Dict

from common.models import Request, Response
from llm.inference import run_llm
from rag.retriever import retrieve_context


class Worker:
    def __init__(self, worker_id: int = 1):
        self.id = worker_id

        self._healthy = True
        self._health_lock = threading.Lock()

        self._active = 0
        self._total_processed = 0
        self._successful = 0
        self._failed = 0
        self._recent_latencies = deque(maxlen=50)
        self._metrics_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Health / failure simulation
    # ------------------------------------------------------------------
    def is_healthy(self) -> bool:
        with self._health_lock:
            return self._healthy

    def fail(self) -> None:
        with self._health_lock:
            self._healthy = False

    def recover(self) -> None:
        with self._health_lock:
            self._healthy = True

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------
    def metrics(self) -> Dict[str, Any]:
        with self._metrics_lock:
            latencies = list(self._recent_latencies)
            avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
            return {
                "worker_id": self.id,
                "healthy": self.is_healthy(),
                "active_requests": self._active,
                "total_processed": self._total_processed,
                "successful": self._successful,
                "failed": self._failed,
                "avg_latency": round(avg_latency, 4),
                "latency_samples": len(latencies),
            }

    def health(self) -> Dict[str, Any]:
        return {
            "status": "healthy" if self.is_healthy() else "failed",
            "worker_id": self.id,
            "healthy": self.is_healthy(),
            "metrics": self.metrics(),
        }

    # ------------------------------------------------------------------
    # Processing
    # ------------------------------------------------------------------
    def process(self, request: Request) -> Response:
        if not self.is_healthy():
            raise RuntimeError(f"Worker {self.id} is currently failed/unhealthy")

        with self._metrics_lock:
            self._active += 1
            self._total_processed += 1

        start = time.time()

        try:
            rag_result = retrieve_context(request.query, top_k=3)

            if rag_result.get("error"):
                latency = time.time() - start
                with self._metrics_lock:
                    self._failed += 1
                    self._recent_latencies.append(latency)

                return Response(
                    id=request.id,
                    result="",
                    latency=latency,
                    worker_id=self.id,
                    success=False,
                    error=rag_result["error"],
                    sources=[],
                )

            context = rag_result["context"]
            sources = rag_result.get("sources", [])

            llm_result = run_llm(request.query, context)
            latency = time.time() - start

            with self._metrics_lock:
                self._successful += 1
                self._recent_latencies.append(latency)

            return Response(
                id=request.id,
                result=llm_result.get("answer") or "",
                latency=latency,
                worker_id=self.id,
                success=True,
                error=None,
                sources=sources,
            )

        except Exception as exc:
            latency = time.time() - start

            with self._metrics_lock:
                self._failed += 1
                self._recent_latencies.append(latency)

            return Response(
                id=request.id,
                result="",
                latency=latency,
                worker_id=self.id,
                success=False,
                error=str(exc),
                sources=[],
            )

        finally:
            with self._metrics_lock:
                self._active -= 1
