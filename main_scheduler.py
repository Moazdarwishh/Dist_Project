"""Uvicorn entry point for the FastAPI scheduler service.

Run example:
    WORKER_URLS=http://localhost:9001,http://localhost:9002,http://localhost:9003,http://localhost:9004 \
    uvicorn main_scheduler:app --host 0.0.0.0 --port 8000
"""

from scheduler.api import app
