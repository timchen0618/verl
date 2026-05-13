"""
Stage 1b: Reproduce the concurrent creation race condition.

In the full training run, the TaskRunner creates TWO SearchTool instances
(one for train dataset, one for val dataset), each spawning a SearchExecutionWorker
that tries to create/get the same TokenBucketWorker('rate-limiter').

This test reproduces that exact scenario.
"""

import time
import ray
import threading

print("Initializing Ray...")
ray.init(ignore_reinit_error=True)

import threading
from contextlib import ExitStack


@ray.remote(concurrency_groups={"acquire": 1, "release": 10})
class TokenBucketWorker:
    def __init__(self, rate_limit: int):
        self.rate_limit = rate_limit
        self._semaphore = threading.Semaphore(rate_limit)

    @ray.method(concurrency_group="acquire")
    def acquire(self):
        self._semaphore.acquire()

    @ray.method(concurrency_group="release")
    def release(self):
        self._semaphore.release()

    def ping(self):
        return True


class SearchExecutionWorker:
    def __init__(self, enable_global_rate_limit=True, rate_limit=10):
        self.rate_limit_worker = self._init_rate_limit(rate_limit) if enable_global_rate_limit else None

    def _init_rate_limit(self, rate_limit):
        return TokenBucketWorker.options(name="rate-limiter", get_if_exists=True).remote(rate_limit)

    def ping(self):
        return True


print("\n=== Test: Two SearchExecutionWorkers created simultaneously ===")
print("  (mirrors train SearchTool + val SearchTool creation)")

# Create both at the same time — exactly what the training run does
w1 = ray.remote(SearchExecutionWorker).remote(enable_global_rate_limit=True, rate_limit=10)
w2 = ray.remote(SearchExecutionWorker).remote(enable_global_rate_limit=True, rate_limit=10)

print("  Both actors submitted, waiting 15s for crash window...")
time.sleep(15)

try:
    r1 = ray.get(w1.ping.remote(), timeout=10)
    r2 = ray.get(w2.ping.remote(), timeout=10)
    print(f"  w1.ping()={r1}, w2.ping()={r2}  ✓  PASS — no crash")
except Exception as e:
    print(f"  FAIL: {e}")
    print("  → Root cause confirmed: concurrent TokenBucketWorker get_if_exists race")
    import sys; sys.exit(1)

# Also test creating many at once (mirrors AgentLoopWorker scenario)
print("\n=== Test: 8 workers created simultaneously (AgentLoopWorker scenario) ===")
workers = [ray.remote(SearchExecutionWorker).remote(enable_global_rate_limit=True, rate_limit=10) for _ in range(8)]
time.sleep(20)
try:
    results = ray.get([w.ping.remote() for w in workers], timeout=15)
    print(f"  All {len(results)} workers alive  ✓  PASS")
except Exception as e:
    print(f"  FAIL: {e}")
    import sys; sys.exit(1)

print("\n✓ No concurrent crash. The crash in training must be from something else.")
ray.shutdown()
