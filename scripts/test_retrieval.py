#!/usr/bin/env python3
"""Standalone script to test retrieval server connectivity and response format.

Run this BEFORE training to verify:
  1. The retrieval server is reachable
  2. The API returns valid results
  3. The response format is compatible

Usage:
  python scripts/test_retrieval.py --url http://127.0.0.1:8007/retrieve
  VERL_LOGGING_LEVEL=DEBUG python scripts/test_retrieval.py --url http://127.0.0.1:8007/retrieve

Environment:
  VERL_LOGGING_LEVEL: Set to INFO or DEBUG to see search utility logs
"""

import argparse
import sys
from pathlib import Path

# Add verl package to path for imports
try:
    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root))
    from verl.tools.utils.search_r1_like_utils import call_search_api, perform_single_search_batch

    USE_VERL_UTILS = True
except ImportError as e:
    USE_VERL_UTILS = False
    IMPORT_ERROR = str(e)


def test_with_requests(url: str, query: str, topk: int, timeout: int):
    """Minimal test using only requests (no verl deps)."""
    import requests

    payload = {"queries": [query], "topk": topk, "return_scores": True}
    try:
        r = requests.post(url, json=payload, timeout=timeout)
        r.raise_for_status()
        data = r.json()
    except requests.exceptions.ConnectionError as e:
        return f"Connection failed: {e}. Is the retrieval server running on {url}?"
    except requests.exceptions.Timeout:
        return f"Timeout after {timeout}s"
    except Exception as e:
        return str(e)

    raw_results = data.get("result", [])
    n_queries = len(raw_results)
    n_docs = len(raw_results[0]) if raw_results and isinstance(raw_results[0], list) else 0
    return None, data, n_queries, n_docs


def main():
    parser = argparse.ArgumentParser(description="Test retrieval server")
    parser.add_argument(
        "--url",
        default="http://127.0.0.1:8007/retrieve",
        help="Retrieval service URL (include /retrieve or not, script will append if needed)",
    )
    parser.add_argument("--query", default="What is Python programming?", help="Test query")
    parser.add_argument("--topk", type=int, default=3, help="Number of results")
    parser.add_argument("--timeout", type=int, default=30, help="Request timeout (seconds)")
    parser.add_argument(
        "--simple",
        action="store_true",
        help="Use simple requests-only test (no verl imports)",
    )
    args = parser.parse_args()

    url = args.url.rstrip("/")
    if not url.endswith("/retrieve"):
        url = url + "/retrieve"

    print("=" * 60)
    print("Retrieval Server Test")
    print("=" * 60)
    print(f"URL:      {url}")
    print(f"Query:    {args.query}")
    print(f"Top-k:    {args.topk}")
    print()

    if args.simple or not USE_VERL_UTILS:
        if not USE_VERL_UTILS and not args.simple:
            print(f"[Note] Could not import verl utils ({IMPORT_ERROR})")
            print("       Using simple requests-only test.\n")
        print("[1] Testing with requests POST...")
        err, data, n_queries, n_docs = test_with_requests(
            url, args.query, args.topk, args.timeout
        )
        if err:
            print(f"    FAIL: {err}")
            return 1
        print(f"    OK: Got response, {n_queries} query result(s), {n_docs} doc(s) per query")
        if n_docs > 0:
            print("SUCCESS: Retrieval is working.")
            return 0
        else:
            print("WARNING: No documents returned (empty index or no matches?)")
            return 0

    # Full verl utils test
    print("[1] Testing call_search_api (low-level)...")
    api_response, error_msg = call_search_api(
        retrieval_service_url=url,
        query_list=[args.query],
        topk=args.topk,
        return_scores=True,
        timeout=args.timeout,
    )

    if error_msg:
        print(f"    FAIL: {error_msg}")
        return 1

    if not api_response:
        print("    FAIL: No response and no error (unexpected)")
        return 1

    raw_results = api_response.get("result", [])
    print(f"    OK: Got HTTP response")
    print(f"    result is list of length {len(raw_results)} (one per query)")

    if raw_results:
        first_query_results = raw_results[0]
        n_docs = len(first_query_results) if isinstance(first_query_results, list) else 1
        print(f"    First query has {n_docs} document(s)")
        if n_docs > 0 and isinstance(first_query_results, list):
            first_doc = first_query_results[0]
            doc_inner = first_doc.get("document", first_doc)
            doc_keys = list(doc_inner.keys()) if isinstance(doc_inner, dict) else []
            print(f"    Document keys: {doc_keys}")
    else:
        print("    WARNING: result is empty - retrieval may have no hits")

    # Test 2: perform_single_search_batch (full pipeline)
    print()
    print("[2] Testing perform_single_search_batch (full pipeline)...")
    result_text, metadata = perform_single_search_batch(
        retrieval_service_url=url,
        query_list=[args.query],
        topk=args.topk,
        timeout=args.timeout,
    )

    print(f"    status:       {metadata['status']}")
    print(f"    total_results: {metadata['total_results']}")
    print(f"    query_count:  {metadata['query_count']}")
    if metadata.get("api_request_error"):
        print(f"    api_error:    {metadata['api_request_error']}")
    if metadata["status"] == "success":
        preview = result_text[:300] + "..." if len(result_text) > 300 else result_text
        print(f"    result (preview): {preview}")
    else:
        print(f"    result_text: {result_text[:200]}")

    print()
    if metadata["status"] == "success" and metadata["total_results"] > 0:
        print("SUCCESS: Retrieval is working correctly.")
        return 0
    elif metadata["status"] == "no_results":
        print("WARNING: Retrieval responded but returned no documents.")
        print("         Check that your index/corpus has data and matches the query.")
        return 0
    else:
        print("FAIL: Retrieval is not working as expected.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
