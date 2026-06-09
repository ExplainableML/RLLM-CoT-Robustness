#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rllm_robustness.io import load_records, write_csv, write_json
from rllm_robustness.metrics import robustness_from_scores

logger = logging.getLogger("compute_metrics")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute sampling-based robustness metrics.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--scores-column", default="verifier_scores")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    rows = load_records(args.input, split=args.split, limit=args.limit)
    item_rows = []
    for row in rows:
        scores = row.get(args.scores_column) or []
        metrics = robustness_from_scores(scores)
        item_rows.append(
            {
                "record_id": row.get("record_id", ""),
                "model_name": row.get("continuation_model", row.get("model_name", "")),
                "domain": row.get("domain", ""),
                "intervention": row.get("intervention", ""),
                "append_after_intervention": row.get("append_after_intervention"),
                "target_timestep": row.get("target_timestep"),
                **metrics,
            }
        )

    groups: defaultdict[tuple, list[dict]] = defaultdict(list)
    for row in item_rows:
        key = (
            row["domain"],
            row["model_name"],
            row["intervention"],
            row["append_after_intervention"],
            row["target_timestep"],
        )
        groups[key].append(row)

    summary_rows = []
    for (domain, model_name, intervention, append_after_intervention, timestep), group_rows in groups.items():
        df = pd.DataFrame(group_rows)
        summary_rows.append(
            {
                "domain": domain,
                "model_name": model_name,
                "intervention": intervention,
                "append_after_intervention": append_after_intervention,
                "target_timestep": timestep,
                "num_items": len(group_rows),
                "at_least_once_robustness": float(df["at_least_once_robust"].mean()),
                "majority_robustness": float(df["majority_robust"].mean()),
                "all_robustness": float(df["all_robust"].mean()),
                "mean_k_correct": float(df["k_correct"].mean()),
                "std_k_correct": float(df["k_correct"].std(ddof=0)),
            }
        )

    payload = {"items": item_rows, "summary": summary_rows}
    write_json(args.output_json, payload)
    write_csv(args.output_csv, summary_rows)
    logger.info("Wrote %s summary rows", len(summary_rows))


if __name__ == "__main__":
    main()
