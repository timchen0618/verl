"""
Stage 1c: Reproduce the SearchExecutionWorker crash with the EXACT ray.init()
environment the training run uses, including NCCL/CUDA env vars.
"""

import time
import ray
import threading
import sys

# Exact runtime_env from verl/trainer/constants_ppo.py / ray_trainer.py
TRAINING_RUNTIME_ENV = {
    "env_vars": {
        "TOKENIZERS_PARALLELISM": "true",
        "NCCL_DEBUG": "WARN",
        "VLLM_LOGGING_LEVEL": "WARN",
        "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "true",
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        "NCCL_CUMEM_ENABLE": "0",
        "VLLM_DISABLE_COMPILE_CACHE": "1",
        "HCCL_HOST_SOCKET_PORT_RANGE": "auto",
        "HCCL_NPU_SOCKET_PORT_RANGE": "auto",
    }
}

print("Initializing Ray with FULL training runtime_env...")
ray.init(runtime_env=TRAINING_RUNTIME_ENV, ignore_reinit_error=True)
print(f"Ray version: {ray.__version__}")


@ray.remote(concurrency_groups={"acquire": 1, "release": 10})
class TokenBucketWorker:
    def __init__(self, rate_limit: int):
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


print("\n=== Test: Two concurrent SearchExecutionWorkers with training runtime_env ===")
w1 = ray.remote(SearchExecutionWorker).remote(enable_global_rate_limit=True, rate_limit=10)
w2 = ray.remote(SearchExecutionWorker).remote(enable_global_rate_limit=True, rate_limit=10)

print("  Waiting 20s (original crash happened at ~12s)...")
time.sleep(20)

try:
    r1 = ray.get(w1.ping.remote(), timeout=10)
    r2 = ray.get(w2.ping.remote(), timeout=10)
    print(f"  w1={r1}, w2={r2}  ✓  PASS — no crash with training runtime_env")
except Exception as e:
    print(f"  FAIL: {e}")
    print("  → Root cause: training runtime_env triggers crash")
    sys.exit(1)

# Also test: does VERL_LOGGING_LEVEL=INFO (set in the training script) matter?
print("\n=== Test: With VERL_LOGGING_LEVEL=INFO (set in run script) ===")
import os
os.environ["VERL_LOGGING_LEVEL"] = "INFO"

w3 = ray.remote(SearchExecutionWorker).remote(enable_global_rate_limit=True, rate_limit=10)
time.sleep(20)

try:
    r3 = ray.get(w3.ping.remote(), timeout=10)
    print(f"  w3={r3}  ✓  PASS")
except Exception as e:
    print(f"  FAIL: {e}")
    print("  → Root cause: VERL_LOGGING_LEVEL=INFO triggers crash via ray remote env var propagation")
    sys.exit(1)

print("\n✓ All tests passed. Crash may require actual GPU workers (FSDP/SGLang) to trigger.")
ray.shutdown()
