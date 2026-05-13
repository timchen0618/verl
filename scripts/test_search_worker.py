"""
Stage 1 test: isolate SearchExecutionWorker crash.

Tests in order:
  A. Plain actor (no options) with enable_global_rate_limit=False  → no TokenBucketWorker
  B. Plain actor with enable_global_rate_limit=True               → creates TokenBucketWorker
  C. Actor with concurrency_groups (like TokenBucketWorker itself)
  D. Full SearchTool execute against live retrieval server (port 8007)

Run: python scripts/test_search_worker.py
"""

import time
import ray
import sys

print("Initializing Ray...")
ray.init(ignore_reinit_error=True)
print(f"Ray version: {ray.__version__}")

# ── Reproduce the exact classes from search_tool.py ─────────────────────────

import threading
from contextlib import ExitStack


@ray.remote(concurrency_groups={"acquire": 1, "release": 10})
class TokenBucketWorker:
    def __init__(self, rate_limit: int):
        self.rate_limit = rate_limit
        self.current_count = 0
        self._semaphore = threading.Semaphore(rate_limit)

    @ray.method(concurrency_group="acquire")
    def acquire(self):
        self._semaphore.acquire()
        self.current_count += 1

    @ray.method(concurrency_group="release")
    def release(self):
        self._semaphore.release()
        self.current_count -= 1

    def get_current_count(self):
        return self.current_count


class SearchExecutionWorker:
    def __init__(self, enable_global_rate_limit=True, rate_limit=10):
        self.rate_limit_worker = self._init_rate_limit(rate_limit) if enable_global_rate_limit else None

    def _init_rate_limit(self, rate_limit):
        return TokenBucketWorker.options(name="rate-limiter", get_if_exists=True).remote(rate_limit)

    def ping(self):
        return True

    def execute(self, fn, *fn_args, **fn_kwargs):
        if self.rate_limit_worker:
            with ExitStack() as stack:
                stack.callback(self.rate_limit_worker.release.remote)
                ray.get(self.rate_limit_worker.acquire.remote())
                return fn(*fn_args, **fn_kwargs)
        else:
            return fn(*fn_args, **fn_kwargs)


# ── Test A: no TokenBucketWorker ─────────────────────────────────────────────
print("\n=== Test A: SearchExecutionWorker(enable_global_rate_limit=False) ===")
try:
    worker_a = ray.remote(SearchExecutionWorker).remote(enable_global_rate_limit=False)
    time.sleep(3)  # give actor time to crash if it will
    result = ray.get(worker_a.ping.remote(), timeout=10)
    print(f"  ping() = {result}  ✓  PASS")
except Exception as e:
    print(f"  FAIL: {e}")
    sys.exit(1)

# ── Test B: with TokenBucketWorker ───────────────────────────────────────────
print("\n=== Test B: SearchExecutionWorker(enable_global_rate_limit=True) ===")
try:
    worker_b = ray.remote(SearchExecutionWorker).remote(enable_global_rate_limit=True, rate_limit=10)
    time.sleep(5)  # original crash happened ~12s in; give it time
    result = ray.get(worker_b.ping.remote(), timeout=15)
    print(f"  ping() = {result}  ✓  PASS")
except Exception as e:
    print(f"  FAIL: {e}")
    print("  → Root cause: TokenBucketWorker creation inside actor __init__ crashes Ray 2.55.x")
    sys.exit(1)

# ── Test C: execute() with rate limit ────────────────────────────────────────
print("\n=== Test C: execute() with rate limiting ===")
try:
    result = ray.get(worker_b.execute.remote(lambda: "hello"), timeout=15)
    print(f"  execute() = {result}  ✓  PASS")
except Exception as e:
    print(f"  FAIL: {e}")
    sys.exit(1)

# ── Test D: SearchTool against live retrieval server ─────────────────────────
print("\n=== Test D: SearchTool HTTP call to retrieval server ===")
try:
    import requests
    resp = requests.post(
        "http://127.0.0.1:8007/retrieve",
        json={"queries": ["who is the president of France"], "topk": 1},
        timeout=10,
    )
    data = resp.json()
    print(f"  HTTP status={resp.status_code}")
    print(f"  result keys={list(data.keys())}")
    if resp.status_code == 200:
        print("  ✓  PASS")
    else:
        print("  FAIL: bad status")
        sys.exit(1)
except Exception as e:
    print(f"  FAIL (retrieval server not ready?): {e}")
    sys.exit(1)

print("\n✓ All Stage 1 tests passed.")
ray.shutdown()
