#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rllm_robustness.constants import INTERVENTIONS, TIMESTEPS
from rllm_robustness.io import load_records


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate an intervention, continuation, or judged grid.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--expected-items", type=int, required=True)
    parser.add_argument("--interventions", default=",".join(INTERVENTIONS))
    parser.add_argument("--timesteps", default=",".join(str(t) for t in TIMESTEPS))
    parser.add_argument("--conditions", default="baseline")
    parser.add_argument(
        "--kind",
        choices=["interventions", "continuations", "judged"],
        default="judged",
    )
    parser.add_argument("--num-completions", type=int, default=8)
    args = parser.parse_args()

    interventions = parse_csv(args.interventions)
    timesteps = parse_csv(args.timesteps)
    conditions = parse_csv(args.conditions)
    rows = load_records(args.input, split=args.split)

    counts = Counter(
        (
            row.get("append_after_intervention") or "baseline",
            row.get("intervention"),
            str(row.get("target_timestep")),
        )
        for row in rows
    )
    expected_cells = {
        (condition, intervention, timestep)
        for condition in conditions
        for intervention in interventions
        for timestep in timesteps
    }
    actual_cells = set(counts)
    missing = sorted(expected_cells - actual_cells)
    extra = sorted(actual_cells - expected_cells)
    wrong_counts = sorted(
        (cell, counts[cell])
        for cell in expected_cells & actual_cells
        if counts[cell] != args.expected_items
    )

    bad_rows: list[tuple[int, str]] = []
    for index, row in enumerate(rows, start=1):
        if args.kind in {"continuations", "judged"}:
            continuations = row.get("continuations") or []
            complete_answers = row.get("complete_answers") or []
            if len(continuations) != args.num_completions:
                bad_rows.append((index, f"continuations={len(continuations)}"))
            if len(complete_answers) != args.num_completions:
                bad_rows.append((index, f"complete_answers={len(complete_answers)}"))
        if args.kind == "judged":
            scores = row.get("verifier_scores") or []
            if len(scores) != args.num_completions:
                bad_rows.append((index, f"verifier_scores={len(scores)}"))

    expected_rows = len(expected_cells) * args.expected_items
    errors = []
    if len(rows) != expected_rows:
        errors.append(f"expected {expected_rows} rows, found {len(rows)}")
    if missing:
        errors.append(f"missing cells: {missing[:10]}")
    if extra:
        errors.append(f"extra cells: {extra[:10]}")
    if wrong_counts:
        errors.append(f"wrong cell counts: {wrong_counts[:10]}")
    if bad_rows:
        errors.append(f"bad rows: {bad_rows[:10]}")

    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        raise SystemExit(1)

    print(
        f"OK: {len(rows)} rows, {len(actual_cells)} cells, "
        f"{args.expected_items} items per cell, kind={args.kind}"
    )


if __name__ == "__main__":
    main()
