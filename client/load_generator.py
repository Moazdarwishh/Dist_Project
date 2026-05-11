from __future__ import annotations

import argparse
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from statistics import mean
from typing import Dict, List, Optional

import requests

_QUERIES = [
    "How does round robin load balancing work?",
    "What is least connections routing?",
    "How does load-aware routing pick a backend?",
    "What is retrieval augmented generation?",
    "How does fault tolerance work in distributed systems?",
    "What does a heartbeat mechanism do?",
    "Why batch multiple LLM requests?",
    "How does LLM inference work?",
    "What is cosine similarity?",
    "What is horizontal scaling?",
]


def _percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * p
    f = int(k)
    c = min(f + 1, len(s) - 1)
    if f == c:
        return s[f]
    return s[f] + (s[c] - s[f]) * (k - f)


def _user_task(scheduler_url: str, user_id: int, timeout: float) -> Dict:
    query = _QUERIES[user_id % len(_QUERIES)]
    response = requests.post(
        f"{scheduler_url.rstrip('/')}/request",
        json={"id": user_id, "query": query},
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()


def _schedule_failure(fail_worker_url: str, delay: float) -> None:
    def _fail():
        time.sleep(delay)
        try:
            print(f"\n[FailureDemo] Calling {fail_worker_url}/fail\n")
            requests.post(f"{fail_worker_url.rstrip('/')}/fail", timeout=5)
        except Exception as exc:
            print(f"[FailureDemo] Failed to call /fail: {exc}")

    threading.Thread(target=_fail, daemon=True).start()


def run_load_test(
    scheduler_url: str = "http://localhost:8000",
    num_users: int = 1005,
    max_concurrency: int = 64,
    timeout: float = 180.0,
    warmup: bool = True,
    fail_worker_url: Optional[str] = None,
    fail_after: float = 5.0,
) -> List[Dict]:
    if warmup:
        print("[Client] Warm-up request...")
        try:
            _user_task(scheduler_url, -1, timeout)
        except Exception as exc:
            print(f"[Client] Warm-up failed: {exc}")

    if fail_worker_url:
        _schedule_failure(fail_worker_url, fail_after)

    print(f"[Client] Starting HTTP load test: users={num_users}, concurrency<={max_concurrency}")
    start = time.time()
    results: List[Dict] = []

    with ThreadPoolExecutor(max_workers=max_concurrency) as pool:
        futures = [pool.submit(_user_task, scheduler_url, i, timeout) for i in range(num_users)]
        for future in as_completed(futures):
            try:
                results.append(future.result())
            except Exception as exc:
                results.append({"success": False, "error": str(exc), "latency": 0.0, "worker_id": None})

    elapsed = time.time() - start
    successes = [r for r in results if r.get("success")]
    failures = [r for r in results if not r.get("success")]
    latencies = [float(r.get("latency", 0.0)) for r in successes]
    worker_counts = Counter(r.get("worker_id") for r in successes)
    source_count = sum(len(r.get("sources") or []) for r in successes)

    print("\n========== HTTP Load Test Summary ==========")
    print(f"  Scheduler URL   : {scheduler_url}")
    print(f"  Users           : {num_users}")
    print(f"  Concurrency     : {max_concurrency}")
    print(f"  Wall time       : {elapsed:.2f} s")
    print(f"  Throughput      : {(len(results) / elapsed) if elapsed > 0 else 0.0:.2f} req/s")
    print(f"  Successful      : {len(successes)}")
    print(f"  Failed          : {len(failures)}")
    print(f"  Avg latency     : {(mean(latencies) if latencies else 0.0):.3f} s")
    print(f"  p95 latency     : {_percentile(latencies, 0.95):.3f} s")
    print(f"  Per-worker      : {dict(sorted(worker_counts.items(), key=lambda x: str(x[0])))}")
    print(f"  Retrieved sources total: {source_count}")
    print("===========================================\n")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run HTTP load test against the scheduler service")
    parser.add_argument("--scheduler", default="http://localhost:8000")
    parser.add_argument("--users", type=int, default=1005)
    parser.add_argument("--concurrency", type=int, default=64)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--no-warmup", action="store_true")
    parser.add_argument("--fail-worker-url", default=None, help="Optional worker URL to fail during test, e.g. http://localhost:9002")
    parser.add_argument("--fail-after", type=float, default=5.0)
    args = parser.parse_args()

    run_load_test(
        scheduler_url=args.scheduler,
        num_users=args.users,
        max_concurrency=args.concurrency,
        timeout=args.timeout,
        warmup=not args.no_warmup,
        fail_worker_url=args.fail_worker_url,
        fail_after=args.fail_after,
    )


if __name__ == "__main__":
    main()
