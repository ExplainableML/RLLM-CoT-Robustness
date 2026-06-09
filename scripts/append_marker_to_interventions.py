#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rllm_robustness.interventions import append_marker_to_intervention_row
from rllm_robustness.io import JsonlBatchWriter, batched, existing_ids, load_records

logger = logging.getLogger("append_marker_to_interventions")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Append a marker such as 'Wait' to already generated intervention rows."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--marker", default="Wait")
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--flush-every-batches", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s - %(levelname)s - %(message)s")

    rows = load_records(args.input, split=args.split, limit=args.limit)
    seen = existing_ids(args.output) if args.resume else set()
    writer = JsonlBatchWriter(args.output, flush_every_batches=args.flush_every_batches)

    total = 0
    for batch in batched(rows, args.batch_size):
        out_rows = [append_marker_to_intervention_row(row, args.marker) for row in batch]
        if seen:
            out_rows = [row for row in out_rows if row["record_id"] not in seen]
        total += len(out_rows)
        writer.add_batch(out_rows)
        logger.info("Wrote %s marker-appended rows so far", total)

    writer.close()
    logger.info("Done. Wrote %s rows to %s", total, args.output)


if __name__ == "__main__":
    main()
