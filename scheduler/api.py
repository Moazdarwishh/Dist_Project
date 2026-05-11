"""FastAPI scheduler/load-balancer service."""

from __future__ import annotations

import itertools
import os
import threading
from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel, Field

from common.models import Request
from lb.load_balancer import LoadBalancer, WorkerProxy
from master.scheduler import Scheduler

DEFAULT_WORKER_URLS = "http://localhost:9001,http://localhost:9002,http://localhost:9003,http://localhost:9004"


def _parse_worker_urls() -> list[str]:
    raw = os.getenv("WORKER_URLS", DEFAULT_WORKER_URLS)
    urls = [u.strip().rstrip("/") for u in raw.split(",") if u.strip()]
    if not urls:
        raise RuntimeError("WORKER_URLS must contain at least one worker URL")
    return urls


def _build_scheduler() -> Scheduler:
    urls = _parse_worker_urls()
    workers = [WorkerProxy(worker_id=i, url=url) for i, url in enumerate(urls, start=1)]
    strategy = os.getenv("LB_STRATEGY", os.getenv("STRATEGY", "load_aware"))
    heartbeat = float(os.getenv("HEARTBEAT_INTERVAL", "5.0"))
    lb = LoadBalancer(workers, strategy=strategy)
    return Scheduler(lb, heartbeat_interval=heartbeat)


app = FastAPI(title="Distributed RAG Scheduler", version="2.0")
scheduler = _build_scheduler()
_id_counter = itertools.count(1)
_id_lock = threading.Lock()


class UserRequest(BaseModel):
    id: Optional[int] = Field(default=None, description="Optional request id")
    query: str = Field(..., min_length=1, description="User question")


@app.post("/request")
def handle_request(payload: UserRequest):
    if payload.id is None:
        with _id_lock:
            req_id = next(_id_counter)
    else:
        req_id = payload.id

    response = scheduler.handle_request(Request(id=req_id, query=payload.query))
    return response.to_dict()


@app.get("/metrics")
def metrics():
    return {
        "scheduler": scheduler.report(),
        "workers": scheduler.worker_summary(),
    }


@app.get("/health")
def health():
    workers = scheduler.worker_summary()
    healthy_count = sum(1 for info in workers.values() if info["healthy"])
    return {
        "status": "healthy" if healthy_count > 0 else "degraded",
        "healthy_workers": healthy_count,
        "total_workers": len(workers),
        "workers": workers,
    }


@app.on_event("shutdown")
def shutdown_event():
    scheduler.stop()
