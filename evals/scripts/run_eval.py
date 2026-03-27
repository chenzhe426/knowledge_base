#!/usr/bin/env python3
"""
run_eval.py – Run evaluation on a JSONL dataset.

Usage:
    python evals/scripts/run_eval.py \
        --dataset evals/datasets/kb_eval_seed.jsonl \
        --config evals/configs/baseline.yaml \
        --output evals/reports/run_$(date +%Y%m%d_%H%M%S).json

The script:
  1. Loads the dataset (JSONL, one EvalSample per line)
  2. For each sample, calls the RAG pipeline via EvalAdapter
  3. Scores retrieval (hit@k, MRR, recall) and answer (exact/partial/wrong/refuse/clarify)
  4. Writes a JSON report + Markdown report
"""
from __future__ import annotations

import argparse
import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

# Ensure the project root is on the path
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

try:
    import yaml
except ImportError:
    yaml = None

from evals.utils.dataset import load_dataset, EvalSample
from evals.utils.adapters import EvalAdapter
from evals.utils.scorer import RetrievalScorer, AnswerScorer
from evals.utils.report import build_json_report, build_markdown_report


def load_config(config_path: str | Path) -> dict:
    """Load YAML (or JSON) config, merge defaults for any missing keys."""
    config_path = Path(config_path)
    if not config_path.exists():
        print(f"[WARN] Config not found: {config_path}, using defaults", file=sys.stderr)
        return {}

    with config_path.open(encoding="utf-8") as f:
        raw = f.read()

    if yaml is not None:
        cfg = yaml.safe_load(raw) or {}
    else:
        # Fallback: try JSON (works for most YAML subsets)
        import json
        cfg = json.loads(raw)

    # Default values for any missing keys
    defaults = {
        "mode": "internal",
        "top_k": 5,
        "normalize_text": True,
        "use_doc_level_fallback": True,
        "use_multistage": False,
        "api": {"base_url": "http://127.0.0.1:8000"},
    }
    for k, v in defaults.items():
        if k not in cfg:
            cfg[k] = v
    return cfg


def run_one_sample(
    sample: EvalSample,
    adapter: EvalAdapter,
    retrieval_scorer: RetrievalScorer,
    answer_scorer: AnswerScorer,
) -> dict:
    """
    Run retrieval + answer for one sample and return a unified result dict.

    Retrieval scoring is skipped when label_status is 'unlabeled' or 'unanswerable'.
    """
    query = sample.question.user_query
    history = sample.question.conversation_history
    label_status = sample.retrieval.label_status

    # Skip retrieval scoring for unlabeled / unanswerable
    do_retrieval_scoring = label_status in {"labeled_doc", "labeled_chunk"}

    result = {
        "id": sample.id,
        "query": query,
        "label_status": label_status,
        "retrieved_chunks": [],
        "final_answer": "",
        "latency_ms": 0.0,
        "error": "",
        # Retrieval metrics
        "retrieval_hit_at_1": None,
        "retrieval_hit_at_3": None,
        "retrieval_hit_at_5": None,
        "retrieval_recall_at_5": None,
        "retrieval_mrr": None,
        "retrieval_skipped": not do_retrieval_scoring,
        # Answer metrics (filled below)
        "answer_label": "wrong",
        "must_include_hit_ratio": 0.0,
        "must_not_include_violations": [],
    }

    try:
        # Run full RAG pipeline (retrieve + answer)
        answer_result = adapter.answer(query, conversation_history=history if history else None)

        result["retrieved_chunks"] = answer_result.get("retrieved_chunks", [])
        result["final_answer"] = answer_result.get("final_answer", "")
        result["latency_ms"] = answer_result.get("latency_ms", 0.0)

        # Score retrieval only when gold labels exist
        if do_retrieval_scoring:
            retr_scores = retrieval_scorer.score(sample, result["retrieved_chunks"])
            result["retrieval_hit_at_1"] = retr_scores.get("hit_at_1", False)
            result["retrieval_hit_at_3"] = retr_scores.get("hit_at_3", False)
            result["retrieval_hit_at_5"] = retr_scores.get("hit_at_5", False)
            result["retrieval_recall_at_5"] = retr_scores.get("recall_at_5", 0.0)
            result["retrieval_mrr"] = retr_scores.get("mrr", 0.0)

        # Score answer
        answer_scores = answer_scorer.score(sample, answer_result)
        result["answer_label"] = answer_scores.get("answer_label", "wrong")
        result["must_include_hit_ratio"] = answer_scores.get("must_include_hit_ratio", 0.0)
        result["must_not_include_violations"] = answer_scores.get(
            "must_not_include_violations", []
        )

    except Exception as e:
        result["error"] = str(e)

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run evaluation on a KB dataset")
    parser.add_argument(
        "--dataset",
        required=True,
        help="Path to the JSONL dataset file",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the YAML config file",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path for the output JSON report",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Optional custom run ID (default: auto-generated)",
    )
    parser.add_argument(
        "--retrieve-top-k",
        type=int,
        default=None,
        help="Two-stage retrieval: fetch this many candidates, then rerank and return top_k. "
             "Default: same as top_k (single-stage). Set > top_k for two-stage.",
    )
    parser.add_argument(
        "--enable-query-enhance",
        action="store_true",
        help="Enable lightweight financial query enhancement before retrieval.",
    )
    parser.add_argument(
        "--multistage",
        action="store_true",
        help="Enable V3 multi-stage retrieval (doc → section → chunk).",
    )
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Load dataset
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"ERROR: Dataset not found: {dataset_path}", file=sys.stderr)
        sys.exit(1)

    try:
        samples = load_dataset(dataset_path)
    except ValueError as e:
        print(f"ERROR: Failed to load dataset: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"[run_eval] Loaded {len(samples)} samples from {dataset_path.name}")
    print(f"[run_eval] Mode: {config.get('mode', 'internal')}, top_k: {config.get('top_k', 5)}")

    # CLI args override config
    retrieve_top_k = args.retrieve_top_k if args.retrieve_top_k is not None else config.get("retrieve_top_k")
    enable_query_enhance = args.enable_query_enhance or config.get("enable_query_enhance", False)
    use_multistage = args.multistage or config.get("use_multistage", False)
    if retrieve_top_k:
        print(f"[run_eval] Two-stage retrieval: retrieve_top_k={retrieve_top_k}, answer_top_k={config.get('top_k', 5)}")
    if enable_query_enhance:
        print(f"[run_eval] Query enhancement: enabled")
    if use_multistage:
        print(f"[run_eval] V3 multi-stage retrieval: enabled (doc → section → chunk)")

    # Initialize adapter and scorers
    adapter = EvalAdapter(
        mode=config.get("mode", "internal"),
        top_k=config.get("top_k", 5),
        retrieve_top_k=retrieve_top_k,
        enable_query_enhance=enable_query_enhance,
        use_multistage=use_multistage,
        api_base_url=config.get("api", {}).get("base_url", "http://127.0.0.1:8000"),
    )
    retrieval_scorer = RetrievalScorer(
        use_doc_level_fallback=config.get("use_doc_level_fallback", True),
    )
    answer_scorer = AnswerScorer(
        normalize_text=config.get("normalize_text", True),
    )

    # Run evaluation
    run_id = args.run_id or f"run_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    case_results: list[dict] = []

    for i, sample in enumerate(samples):
        result = run_one_sample(sample, adapter, retrieval_scorer, answer_scorer)
        case_results.append(result)
        status = "✓" if not result.get("error") else "✗"
        retr_str = "skipped" if result.get("retrieval_skipped") else f"hit={result['retrieval_hit_at_5']}"
        print(
            f"  [{i+1}/{len(samples)}] {status} [{sample.id}] "
            f"label_status={result['label_status']} retrieval={retr_str} "
            f"answer={result['answer_label']}"
        )

    # Build report
    json_report = build_json_report(
        run_id=run_id,
        dataset_name=str(dataset_path.name),
        config=config,
        samples=samples,
        case_results=case_results,
    )

    # Write JSON report
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(json_report, f, ensure_ascii=False, indent=2)
    print(f"\n[run_eval] JSON report written to: {output_path}")

    # Also write JSONL for downstream score_eval compatibility
    jsonl_path = output_path.with_suffix(".jsonl")
    with jsonl_path.open("w", encoding="utf-8") as f:
        for cr in case_results:
            f.write(json.dumps(cr, ensure_ascii=False) + "\n")
    print(f"[run_eval] JSONL written to: {jsonl_path}")

    # Write Markdown report alongside
    md_path = output_path.with_suffix(".md")
    md_report = build_markdown_report(json_report)
    with md_path.open("w", encoding="utf-8") as f:
        f.write(md_report)
    print(f"[run_eval] Markdown report written to: {md_path}")

    # Print summary
    summary = json_report["summary"]
    print("\n=== Summary ===")
    print(f"  Retrieval  – Hit@1: {summary['retrieval']['hit_at_1']}  "
          f"Hit@3: {summary['retrieval']['hit_at_3']}  "
          f"MRR: {summary['retrieval']['mrr']}")
    print(f"  Answer     – exact: {summary['answer']['exact']}  "
          f"partial: {summary['answer']['partial']}  "
          f"wrong: {summary['answer']['wrong']}")
    print(f"\n  Reports: {output_path}  {md_path}")


if __name__ == "__main__":
    main()
