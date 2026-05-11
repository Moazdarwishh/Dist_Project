"""Uvicorn entry point for a FastAPI worker service.

Run examples:
    WORKER_ID=1 uvicorn main_worker:app --host 0.0.0.0 --port 9001
    WORKER_ID=2 uvicorn main_worker:app --host 0.0.0.0 --port 9002
"""

from workers.api import app
