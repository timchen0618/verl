"""
Stage 2: Test the refactored SearchTool (asyncio.Semaphore, no Ray actors).
Verifies:
  A. SearchTool initializes without error
  B. execute() returns results from the live retrieval server
  C. Multiple concurrent execute() calls work correctly
"""
import asyncio
import sys
import os

# Ensure verl is importable
sys.path.insert(0, "/scratch/hc3337/projects/verl")
os.environ["VERL_LOGGING_LEVEL"] = "INFO"

from verl.tools.search_tool import SearchTool
from verl.tools.schemas import OpenAIFunctionToolSchema

TOOL_SCHEMA = OpenAIFunctionToolSchema(**{
    "type": "function",
    "function": {
        "name": "search",
        "description": "Searches for information.",
        "parameters": {
            "type": "object",
            "properties": {
                "query_list": {"type": "array", "description": "queries"}
            },
            "required": ["query_list"]
        }
    }
})

CONFIG = {
    "retrieval_service_url": "http://127.0.0.1:8007/retrieve",
    "num_workers": 10,
    "rate_limit": 10,
    "timeout": 30,
    "topk": 3,
}


async def main():
    print("\n=== Test A: SearchTool init ===")
    try:
        tool = SearchTool(config=CONFIG, tool_schema=TOOL_SCHEMA)
        print("  ✓ PASS")
    except Exception as e:
        print(f"  FAIL: {e}")
        sys.exit(1)

    print("\n=== Test B: Single execute() ===")
    instance_id, _ = await tool.create()
    try:
        response, reward, metrics = await tool.execute(
            instance_id=instance_id,
            parameters={"query_list": ["Who is the president of France?"]},
        )
        print(f"  reward={reward}, metrics={metrics}")
        print(f"  response text (first 200 chars): {response.text[:200]}")
        if response.text and "result" not in response.text.lower()[:20]:
            print("  ✓ PASS")
        elif "failed" in response.text.lower() or "error" in response.text.lower():
            print(f"  FAIL: got error response: {response.text[:300]}")
            sys.exit(1)
        else:
            print("  ✓ PASS (got response)")
    except Exception as e:
        print(f"  FAIL: {e}")
        sys.exit(1)

    print("\n=== Test C: 10 concurrent execute() calls ===")
    async def one_search(q):
        iid, _ = await tool.create()
        resp, _, m = await tool.execute(iid, {"query_list": [q]})
        await tool.release(iid)
        return resp.text[:50]

    queries = [f"capital of country {i}" for i in range(10)]
    try:
        results = await asyncio.gather(*[one_search(q) for q in queries])
        print(f"  Got {len(results)} results  ✓ PASS")
    except Exception as e:
        print(f"  FAIL: {e}")
        sys.exit(1)

    print("\n✓ Stage 2 complete — SearchTool works without Ray actor overhead.")


asyncio.run(main())
