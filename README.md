# Distributed RAG System

This version contains only the final distributed implementation:

- FastAPI scheduler service on port `8000`
- FastAPI worker services on ports such as `9001`-`9004`
- Real HTTP communication between scheduler and workers
- Real PDF/TXT/DOCX ingestion
- ChromaDB persistent vector database in `./vector_db`
- Source-aware RAG responses
- Real LLM inference using `google/flan-t5-small`
- `/fail` and `/recover` endpoints for fault-tolerance demos

The project is designed to run on macOS and normal laptops using CPU.

## 1. Install

```bash
pip install -r requirements.txt
```

## 2. Add real documents

Create a folder named `documents/` and add files:

```text
documents/
├── distributed_systems.pdf
├── rag_notes.docx
└── llm_inference.txt
```

## 3. Ingest documents into ChromaDB

```bash
python -m rag.ingest --documents ./documents --db ./vector_db --reset
```

You can also run:

```bash
python ingest.py --documents ./documents --db ./vector_db --reset
```

## 4. Start workers

Use a separate terminal for each worker:

```bash
WORKER_ID=1 VECTOR_DB_PATH=./vector_db uvicorn main_worker:app --host 0.0.0.0 --port 9001
WORKER_ID=2 VECTOR_DB_PATH=./vector_db uvicorn main_worker:app --host 0.0.0.0 --port 9002
WORKER_ID=3 VECTOR_DB_PATH=./vector_db uvicorn main_worker:app --host 0.0.0.0 --port 9003
WORKER_ID=4 VECTOR_DB_PATH=./vector_db uvicorn main_worker:app --host 0.0.0.0 --port 9004
```

## 5. Start scheduler

```bash
WORKER_URLS=http://localhost:9001,http://localhost:9002,http://localhost:9003,http://localhost:9004 \
LB_STRATEGY=load_aware \
uvicorn main_scheduler:app --host 0.0.0.0 --port 8000
```

Supported strategies:

```text
round_robin
least_connections
load_aware
```

## 6. Run load test

```bash
python client/load_generator.py --scheduler http://localhost:8000 --users 1005 --concurrency 64
```

## 7. Test fault tolerance

Fail worker 2 during the load test:

```bash
python client/load_generator.py \
  --scheduler http://localhost:8000 \
  --users 1005 \
  --concurrency 64 \
  --fail-worker-url http://localhost:9002 \
  --fail-after 5
```

Recover it manually:

```bash
curl -X POST http://localhost:9002/recover
```

## Useful endpoints

Scheduler:

```text
POST http://localhost:8000/request
GET  http://localhost:8000/health
GET  http://localhost:8000/metrics
```

Worker:

```text
POST http://localhost:9001/process
GET  http://localhost:9001/health
GET  http://localhost:9001/metrics
POST http://localhost:9001/fail
POST http://localhost:9001/recover
```

## Example request

```bash
curl -X POST http://localhost:8000/request \
  -H "Content-Type: application/json" \
  -d '{"id": 1, "query": "What do the documents say about load balancing?"}'
```

The response includes:

- generated answer
- worker id
- latency
- success flag
- retrieved document sources
# Dist_Project
