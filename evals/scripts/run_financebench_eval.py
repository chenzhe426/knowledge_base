#!/usr/bin/env python3
"""
run_financebench_eval.py – Run eval on FinanceBench-format dataset.

Reads FinanceBench JSONL (financebench_v1_subset_3docs.jsonl), runs the RAG
pipeline via EvalAdapter for each sample, and writes a JSONL run file that
score_eval.py can consume.

Usage:
    python -m evals.scripts.run_financebench_eval \
        --dataset evals/data/financebench_v1_subset_3docs.jsonl \
        --output evals/runs/run_$(date +%Y%m%d_%H%M%S).jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from evals.utils.adapters import EvalAdapter

# Default concurrency for eval runs
DEFAULT_WORKERS = int(__import__("os").getenv("EVAL_WORKERS", "4"))


def load_financebench_dataset(path: Path) -> list[dict]:
    """Load FinanceBench-format JSONL."""
    samples = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            samples.append(json.loads(line))
    return samples


# Cache for doc_name -> internal doc_id mapping
_doc_id_cache: dict[str, int] = {}


def _ensure_doc_mapping() -> None:
    """Build doc_name -> internal doc_id cache from KB."""
    if _doc_id_cache:
        return
    from app.db.bootstrap import init_db
    from app.db import get_all_documents
    init_db()
    docs = get_all_documents()
    for d in docs:
        _doc_id_cache[d["title"]] = d["id"]


def doc_name_to_id(title: str) -> int | None:
    """Map document title to internal document_id."""
    _ensure_doc_mapping()
    return _doc_id_cache.get(title)


def run_one_sample(sample: dict, adapter: EvalAdapter) -> dict:
    """Run RAG pipeline for one FinanceBench sample and merge gold info."""
    eval_id = sample.get("id", "") or sample.get("eval_id", "")
    question = sample.get("question", "")

    result = {
        "id": eval_id,
        "query": question,
        "final_answer": "",
        "retrieved_chunks": [],
        "latency_ms": 0.0,
        "error": "",
        # Gold info needed by score_eval
        "gold_doc_name": sample.get("gold_doc_name", ""),
        "gold_pages": sample.get("gold_pages", []),
        "gold_answer": sample.get("gold_answer", ""),
        "gold_chunk_ids": sample.get("gold_chunk_ids") or [],
        "gold_chunk_status": sample.get("gold_chunk_status", "unresolved"),
        "gold_chunk_match_method": sample.get("gold_chunk_match_method", ""),
        # Map gold_doc_name -> internal doc_id for document-level scoring
        "retrieval": {
            "gold_doc_ids": [],
            "gold_chunk_ids": sample.get("gold_chunk_ids") or [],
            "label_status": "labeled_doc" if sample.get("gold_doc_name") else "unlabeled",
        },
        "answer": {
            "gold_answer": sample.get("gold_answer", ""),
            "must_include": sample.get("must_include", []),
        },
        "evaluation": {
            "expected_behavior": sample.get("expected_behavior", "answer"),
        },
    }

    # Convert gold_doc_name to gold_doc_ids
    gold_doc_name = sample.get("gold_doc_name", "")
    if gold_doc_name:
        doc_id = doc_name_to_id(gold_doc_name)
        if doc_id is not None:
            result["retrieval"]["gold_doc_ids"] = [doc_id]

    try:
        answer_result = adapter.answer(question, conversation_history=None)
        result["retrieved_chunks"] = answer_result.get("retrieved_chunks", [])
        result["final_answer"] = answer_result.get("final_answer", "")
        result["latency_ms"] = answer_result.get("latency_ms", 0.0)
    except Exception as e:
        result["error"] = str(e)

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run eval on FinanceBench dataset")
    parser.add_argument(
        "--dataset",
        required=True,
        type=Path,
        help="Path to FinanceBench JSONL dataset",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output path for run JSONL",
    )
    parser.add_argument(
        "--top-k",
        default=5,
        type=int,
        help="Number of chunks to retrieve",
    )
    parser.add_argument(
        "--workers",
        default=DEFAULT_WORKERS,
        type=int,
        help="Number of concurrent workers (default: 4)",
    )
    args = parser.parse_args()

    # Load dataset
    samples = load_financebench_dataset(args.dataset)
    print(f"[run_financebench_eval] Loaded {len(samples)} samples")
    print(f"[run_financebench_eval] Using {args.workers} concurrent workers")

    # Run concurrently
    run_id = f"run_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    results: list[dict] = [None] * len(samples)
    errors: list[str] = []

    def worker(i: int, sample: dict) -> tuple[int, dict]:
        adapter = EvalAdapter(mode="internal", top_k=args.top_k)
        result = run_one_sample(sample, adapter)
        return i, result

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(worker, i, s): i for i, s in enumerate(samples)}
        for future in as_completed(futures):
            try:
                i, result = future.result()
                results[i] = result
            except Exception as e:
                idx = futures[future]
                errors.append(f"[{idx}] {type(e).__name__}: {e}")
                results[idx] = {"id": samples[idx].get("eval_id", ""), "error": str(e)}

    # Print status in original order
    for i, result in enumerate(results):
        if result is None:
            continue
        status = "✓" if not result.get("error") else "✗"
        has_chunks = len(result.get("retrieved_chunks", [])) > 0
        has_answer = bool(result.get("final_answer"))
        print(
            f"  [{i+1}/{len(samples)}] {status} [{result.get('id', '?')}] "
            f"chunks={has_chunks} answer={has_answer}"
        )

    if errors:
        print(f"\n  Errors: {len(errors)}")
        for e in errors[:5]:
            print(f"    {e}")

    # Write JSONL
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n[run_financebench_eval] Written {len(results)} results to: {args.output}")

    # Summary
    n_total = len(results)
    n_ok = sum(1 for r in results if not r.get("error"))
    n_with_chunks = sum(1 for r in results if r.get("retrieved_chunks"))
    n_with_answer = sum(1 for r in results if r.get("final_answer"))
    print(f"\n=== Summary ===")
    print(f"  Total:       {n_total}")
    print(f"  Success:      {n_ok}")
    print(f"  With chunks:  {n_with_chunks}")
    print(f"  With answer: {n_with_answer}")


if __name__ == "__main__":
    main()
