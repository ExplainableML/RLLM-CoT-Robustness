#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import math
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rllm_robustness.io import append_jsonl, load_records, stable_id

logger = logging.getLogger("filter_correct")


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def has_closing_think(row: dict) -> bool:
    return "</think>" in str(row.get("answer_content", ""))


def length_percentile_cutoff(rows: list[dict], percentile: float) -> int | None:
    if not rows or percentile >= 1.0:
        return None
    lengths = sorted(len(str(row.get("answer_content", ""))) for row in rows)
    index = min(len(lengths) - 1, max(0, math.floor(percentile * len(lengths)) - 1))
    return lengths[index]


def filter_one(
    rows: list[dict],
    *,
    score_column: str,
    require_closing_think: bool,
    drop_top_length_fraction: float,
    max_per_reference_answer: int | None,
    keep_ids: set[str] | None,
) -> list[dict]:
    filtered = [row for row in rows if row.get(score_column) == 1]
    if keep_ids is not None:
        filtered = [row for row in filtered if str(row.get("id", row.get("source_id", ""))) in keep_ids]
    if require_closing_think:
        filtered = [row for row in filtered if has_closing_think(row)]
    if drop_top_length_fraction:
        cutoff = length_percentile_cutoff(filtered, 1.0 - drop_top_length_fraction)
        if cutoff is not None:
            filtered = [row for row in filtered if len(str(row.get("answer_content", ""))) <= cutoff]
    if max_per_reference_answer is not None:
        buckets: defaultdict[str, int] = defaultdict(int)
        downsampled = []
        for row in filtered:
            key = str(row.get("reference_answer", ""))
            if buckets[key] >= max_per_reference_answer:
                continue
            buckets[key] += 1
            downsampled.append(row)
        filtered = downsampled
    for row in filtered:
        row["record_id"] = stable_id(row.get("record_id"), "filtered_correct")
    return filtered


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter correctly solved original traces.")
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--outputs", nargs="+", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--score-column", default="verifier_score")
    parser.add_argument("--require-intersection", action="store_true")
    parser.add_argument("--require-closing-think", action="store_true")
    parser.add_argument(
        "--drop-top-length-fraction",
        type=float,
        default=0.0,
        help="Drop this top fraction of longest traces. The MATH setup in the paper uses 0.02.",
    )
    parser.add_argument(
        "--max-per-reference-answer",
        type=int,
        default=None,
        help="Keep at most this many rows per reference answer. The MATH setup in the paper uses 20.",
    )
    args = parser.parse_args()
    setup_logging()

    if len(args.outputs) != len(args.inputs):
        raise ValueError("--outputs must have the same length as --inputs")

    all_rows = [load_records(path, split=args.split, limit=args.limit) for path in args.inputs]
    keep_ids = None
    if args.require_intersection:
        correct_id_sets = []
        for rows in all_rows:
            correct_id_sets.append(
                {str(row.get("id", row.get("source_id", ""))) for row in rows if row.get(args.score_column) == 1}
            )
        keep_ids = set.intersection(*correct_id_sets) if correct_id_sets else set()
        logger.info("Intersection has %s correct IDs", len(keep_ids))

    for rows, output in zip(all_rows, args.outputs):
        filtered = filter_one(
            rows,
            score_column=args.score_column,
            require_closing_think=args.require_closing_think,
            drop_top_length_fraction=args.drop_top_length_fraction,
            max_per_reference_answer=args.max_per_reference_answer,
            keep_ids=keep_ids,
        )
        Path(output).unlink(missing_ok=True)
        append_jsonl(output, filtered)
        logger.info("Wrote %s filtered rows to %s", len(filtered), output)


if __name__ == "__main__":
    main()
