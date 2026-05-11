"""FastAPI worker service."""

import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from common.models import Request
from workers.worker import Worker


def _default_worker_id() -> int:
    # For clean reports, start workers with WORKER_ID=1,2,3,4.
    return int(os.getenv("WORKER_ID", "1"))


worker = Worker(worker_id=_default_worker_id())
app = FastAPI(title="RAG Worker", version="2.1")


class ProcessRequest(BaseModel):
    id: int = Field(..., description="Request id")
    query: str = Field(..., min_length=1, description="User question")


@app.get("/health")
def health():
    return worker.health()


@app.get("/metrics")
def metrics():
    return worker.metrics()


@app.post("/fail")
def fail():
    worker.fail()
    return {"status": "failed", "worker_id": worker.id}


@app.post("/recover")
def recover():
    worker.recover()
    return {"status": "healthy", "worker_id": worker.id}


@app.post("/process")
def process(payload: ProcessRequest):
    if not worker.is_healthy():
        raise HTTPException(status_code=503, detail=f"Worker {worker.id} is failed")

    response = worker.process(Request(id=payload.id, query=payload.query))
    return response.to_dict()
