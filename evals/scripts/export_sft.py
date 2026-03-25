#!/usr/bin/env python3
"""
export_sft.py – Export evaluation samples to SFT-ready messages format.

Usage:
    python evals/scripts/export_sft.py \
        --dataset evals/datasets/kb_eval_seed.jsonl \
        --output evals/datasets/kb_eval_sft.jsonl

Each line in the output is a JSON object with keys:
  - id         : sample id
  - messages   : list of {role, content} dicts
  - gold_answer: reference answer (for validation)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from evals.utils.dataset import load_dataset


def export_sft(samples, use_context: bool = True) -> list[dict]:
    """
    Convert EvalSamples to SFT message format.

    Parameters
    ----------
    samples    : list of EvalSample
    use_context: if True, use sft_messages_with_context; else sft_messages_no_context

    Returns
    -------
    list of dicts, each containing id + messages + gold_answer
    """
    results = []
    for sample in samples:
        supervision = sample.supervision
        if use_context and supervision.sft_messages_with_context:
            messages = supervision.sft_messages_with_context
        else:
            messages = supervision.sft_messages_no_context

        results.append({
            "id": sample.id,
            "messages": messages,
            "gold_answer": sample.answer.gold_answer,
            "task_type": sample.task_type,
            "difficulty": sample.difficulty,
            "tags": sample.tags,
        })
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Export eval dataset to SFT messages format")
    parser.add_argument(
        "--dataset",
        required=True,
        help="Path to the JSONL dataset file",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path for the output JSONL file",
    )
    parser.add_argument(
        "--no-context",
        action="store_true",
        help="Use sft_messages_no_context instead of with_context",
    )
    args = parser.parse_args()

    samples = load_dataset(args.dataset)
    records = export_sft(samples, use_context=not args.no_context)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"[export_sft] Exported {len(records)} samples to {out_path}")
    print(f"[export_sft] Sample record keys: {list(records[0].keys())}")


if __name__ == "__main__":
    main()
