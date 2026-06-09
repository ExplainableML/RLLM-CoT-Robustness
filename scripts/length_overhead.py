#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Callable

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rllm_robustness.io import load_records, write_csv

logger = logging.getLogger("length_overhead")


def build_length_fn(unit: str) -> Callable[[str, dict], int]:
    if unit == "characters":
        return lambda text, _row: len(text)

    tokenizers = {}

    def token_length(text: str, row: dict) -> int:
        model_name = row.get("continuation_model") or row.get("model_name")
        if not model_name:
            raise ValueError("Token length requires continuation_model or model_name in each row")
        if model_name not in tokenizers:
            from transformers import AutoTokenizer

            tokenizers[model_name] = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True,
            )
        return len(tokenizers[model_name].encode(text))

    return token_length


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute CoT length overhead after interventions.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--unit",
        choices=["characters", "tokens"],
        default="characters",
        help="Length unit. tokens loads the tokenizer for each continuation_model or model_name.",
    )
    parser.add_argument(
        "--remove-intervention-text",
        action="store_true",
        help="Subtract inserted intervention text before comparing with the original trace.",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    rows = load_records(args.input, split=args.split, limit=args.limit)
    length_fn = build_length_fn(args.unit)
    per_completion = []
    for row in rows:
        original = str(row.get("answer_content", ""))
        original_len = length_fn(original, row)
        if original_len == 0:
            continue
        intervention_len = (
            length_fn(str(row.get("intervention_text", "")), row) if args.remove_intervention_text else 0
        )
        complete_answers = row.get("complete_answers") or []
        verifier_scores = row.get("verifier_scores") or [None] * len(complete_answers)
        for idx, complete_answer in enumerate(complete_answers):
            adjusted_len = max(0, length_fn(str(complete_answer), row) - intervention_len)
            percent_change = (adjusted_len - original_len) / original_len * 100.0
            per_completion.append(
                {
                    "record_id": row.get("record_id", ""),
                    "completion_index": idx,
                    "model_name": row.get("continuation_model", row.get("model_name", "")),
                    "domain": row.get("domain", ""),
                    "intervention": row.get("intervention", ""),
                    "append_after_intervention": row.get("append_after_intervention"),
                    "target_timestep": row.get("target_timestep"),
                    "verifier_score": verifier_scores[idx] if idx < len(verifier_scores) else None,
                    "unit": args.unit,
                    "original_length": original_len,
                    "adjusted_complete_length": adjusted_len,
                    "percent_change": percent_change,
                }
            )

    groups: defaultdict[tuple, list[dict]] = defaultdict(list)
    for row in per_completion:
        groups[
            (
                row["domain"],
                row["model_name"],
                row["intervention"],
                row["append_after_intervention"],
                row["target_timestep"],
            )
        ].append(row)

    summary = []
    for (domain, model_name, intervention, append_after_intervention, timestep), group_rows in groups.items():
        df = pd.DataFrame(group_rows)
        summary.append(
            {
                "domain": domain,
                "model_name": model_name,
                "intervention": intervention,
                "append_after_intervention": append_after_intervention,
                "target_timestep": timestep,
                "unit": args.unit,
                "num_completions": len(group_rows),
                "mean_percent_change": float(df["percent_change"].mean()),
                "std_percent_change": float(df["percent_change"].std(ddof=0)),
            }
        )
    write_csv(args.output_csv, summary)
    logger.info("Wrote %s length summary rows", len(summary))


if __name__ == "__main__":
    main()
