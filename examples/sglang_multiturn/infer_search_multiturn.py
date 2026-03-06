#!/usr/bin/env python3
# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use it except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Multi-turn inference script for search-augmented QA.
# Connects to an existing SGLang server and a separate retrieval server.
# The LLM calls search by outputting <search>query</search>; results are injected
# and generation continues until <answer> or max_turns.

import argparse
import json
import re
import sys
from pathlib import Path

import requests

# Optional: use verl's search utilities when available (running inside verl env)
try:
    from verl.tools.utils.search_r1_like_utils import call_search_api, _passages2string

    USE_VERL_SEARCH = True
except ImportError:
    USE_VERL_SEARCH = False

# Default paths relative to project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROMPT_FILE = PROJECT_ROOT / "prompts" / "sft_plan_no_replan.txt"

SEARCH_PATTERN = re.compile(r"<search>(.*?)(?:</search>|$)", re.DOTALL)
ANSWER_PATTERN = re.compile(r"<answer>(.*?)(?:</answer>|$)", re.DOTALL)


def extract_search_queries(text: str) -> list[str]:
    """Extract all search queries from model output.

    Handles both complete <search>query</search> and partial <search>query
    (when generation stopped at </search> and the closing tag was excluded).
    """
    matches = SEARCH_PATTERN.findall(text)
    queries = [m.strip() for m in matches if m.strip()]
    return queries


def extract_answer(text: str) -> str | None:
    """Extract final answer from model output."""
    match = ANSWER_PATTERN.search(text)
    return match.group(1).strip() if match else None


def _passages2string_fallback(retrieval_result: list) -> str:
    """Convert retrieval results to formatted string (minimal impl)."""
    format_reference = ""
    for idx, doc_item in enumerate(retrieval_result):
        content = doc_item.get("document", {}).get("contents", "")
        parts = content.split("\n", 1)
        title = parts[0] if parts else ""
        text = parts[1] if len(parts) > 1 else ""
        format_reference += f"Doc {idx + 1} (Title: {title})\n{text}\n\n"
    return format_reference.strip()


def call_retrieval(
    retrieval_url: str,
    query: str,
    topk: int = 3,
    timeout: int = 30,
) -> str:
    """Call the retrieval server and return formatted results.

    Args:
        retrieval_url: Base URL of retrieval service (e.g. http://127.0.0.1:8000).
                      The script will POST to {url}/retrieve.
        query: Search query string.
        topk: Number of results to retrieve.
        timeout: Request timeout in seconds.

    Returns:
        Formatted string of search results, or error message.
    """
    if not retrieval_url.endswith("/retrieve"):
        retrieval_url = retrieval_url.rstrip("/") + "/retrieve"

    payload = {
        "queries": [query],
        "topk": topk,
        "return_scores": True,
    }
    headers = {"Content-Type": "application/json", "Accept": "application/json"}

    try:
        if USE_VERL_SEARCH:
            api_response, error_msg = call_search_api(
                retrieval_service_url=retrieval_url,
                query_list=[query],
                topk=topk,
                return_scores=True,
                timeout=timeout,
            )
            if error_msg:
                return f"Search error: {error_msg}"
            if not api_response:
                return "Search request failed."
        else:
            response = requests.post(
                retrieval_url,
                headers=headers,
                json=payload,
                timeout=timeout,
            )
            response.raise_for_status()
            api_response = response.json()

        raw_results = api_response.get("result", [])
        if not raw_results:
            return "No search results found."

        passages2string = _passages2string_fallback if not USE_VERL_SEARCH else _passages2string
        pretty_results = []
        for retrieval in raw_results:
            formatted = passages2string(retrieval)
            pretty_results.append(formatted)
        return "\n---\n".join(pretty_results)

    except requests.exceptions.RequestException as e:
        return f"Search error: {e}"
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        return f"Search error: failed to parse response: {e}"


def load_system_prompt(prompt_file: Path) -> str:
    """Load system prompt from file."""
    with open(prompt_file, encoding="utf-8") as f:
        return f.read().strip()


def run_inference(
    sglang_url: str,
    retrieval_url: str,
    question: str,
    system_prompt: str,
    max_turns: int = 10,
    max_tokens: int = 2048,
    temperature: float = 0.7,
    topk: int = 3,
    verbose: bool = True,
) -> tuple[str | None, list[dict]]:
    """Run multi-turn inference with search tool.

    Returns:
        (extracted_answer, messages) - answer may be None if max_turns reached.
    """
    # Normalize URLs
    sglang_base = sglang_url.rstrip("/")
    chat_url = f"{sglang_base}/v1/chat/completions"

    question = question.strip()
    if question and question[-1] != "?":
        question += "?"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Question: {question}\n"},
    ]

    stop_sequences = ["</search>", "</answer>"]
    final_answer = None

    for turn in range(max_turns):
        payload = {
            "model": "default",
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stop": stop_sequences,
        }

        try:
            resp = requests.post(chat_url, json=payload, timeout=120)
            resp.raise_for_status()
            data = resp.json()
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"SGLang request failed: {e}") from e

        choices = data.get("choices", [])
        if not choices:
            raise RuntimeError("SGLang returned no choices")

        content = choices[0].get("message", {}).get("content", "") or ""
        finish_reason = choices[0].get("finish_reason", "")

        messages.append({"role": "assistant", "content": content})

        if verbose:
            print(f"\n--- Turn {turn + 1} ---")
            print(content)

        # Check for answer
        answer_match = extract_answer(content)
        if answer_match is not None:
            final_answer = answer_match
            if verbose:
                print(f"\n[Extracted answer: {final_answer}]")
            break

        # Check for search calls
        queries = extract_search_queries(content)
        if queries:
            all_results = []
            for q in queries:
                if verbose:
                    print(f"\n[Searching: {q}]")
                results = call_retrieval(retrieval_url, q, topk=topk)
                all_results.append(f"[Search results for '{q}']:\n{results}")
            combined = "\n\n".join(all_results)
            messages.append({
                "role": "user",
                "content": f"[Search results: {combined}]",
            })
            continue

        # No search and no answer - stop to avoid infinite loop
        if verbose:
            print("\n[Generation ended without answer or search; stopping]")
        break

    return final_answer, messages


def main() -> int:
    parser = argparse.ArgumentParser(description="Multi-turn search-augmented inference with SGLang and retrieval server.")
    parser.add_argument("--sglang-url", default="http://127.0.0.1:30000", help="Base URL of SGLang server")
    parser.add_argument("--retrieval-url", default="http://127.0.0.1:8000", help="Base URL of retrieval server (script appends /retrieve)")
    parser.add_argument("--prompt-file", type=Path, default=DEFAULT_PROMPT_FILE, help="Path to system prompt file")
    parser.add_argument("--question", type=str, help="Single question to answer")
    parser.add_argument("--questions-file", type=Path, help="JSONL file with 'question' or 'query' field per line for batch mode")
    parser.add_argument("--max-turns", type=int, default=10, help="Maximum assistant turns")
    parser.add_argument("--max-tokens", type=int, default=2048, help="Max tokens per generation")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--topk", type=int, default=3, help="Number of retrieval results per query")
    parser.add_argument("--output", type=Path, help="Output JSONL file for batch mode")
    parser.add_argument("-q", "--quiet", action="store_true", help="Reduce verbose output")

    args = parser.parse_args()

    if not args.prompt_file.exists():
        print(f"Prompt file not found: {args.prompt_file}", file=sys.stderr)
        return 1

    system_prompt = load_system_prompt(args.prompt_file)
    verbose = not args.quiet

    if args.questions_file:
        # Batch mode
        if not args.questions_file.exists():
            print(f"Questions file not found: {args.questions_file}", file=sys.stderr)
            return 1
        questions = []
        with open(args.questions_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                q = obj.get("question") or obj.get("query") or obj.get("text", "")
                questions.append(q)

        results = []
        for i, q in enumerate(questions):
            if verbose:
                print(f"\n{'='*60}\nQuestion {i+1}/{len(questions)}: {q[:80]}...")
            answer, _ = run_inference(
                sglang_url=args.sglang_url,
                retrieval_url=args.retrieval_url,
                question=q,
                system_prompt=system_prompt,
                max_turns=args.max_turns,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                topk=args.topk,
                verbose=verbose,
            )
            results.append({"question": q, "answer": answer})

        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                for r in results:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            if verbose:
                print(f"\nWrote {len(results)} results to {args.output}")
        return 0

    if not args.question:
        parser.error("Provide --question or --questions-file")
        return 1

    answer, _ = run_inference(
        sglang_url=args.sglang_url,
        retrieval_url=args.retrieval_url,
        question=args.question,
        system_prompt=system_prompt,
        max_turns=args.max_turns,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        topk=args.topk,
        verbose=verbose,
    )

    if answer is not None:
        print(f"\nAnswer: {answer}")
    return 0


if __name__ == "__main__":
    sys.exit(main())


#     python examples/sglang_multiturn/infer_search_multiturn.py \
#   --sglang-url http://localhost:8000 \
#   --retrieval-url http://localhost:8007 \
#   --question "What university did the architect of the Sydney Opera House attend?" \
#   --max-turns 5