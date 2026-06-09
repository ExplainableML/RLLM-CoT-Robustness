#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rllm_robustness.cli_common import add_common_io_args, add_runner_args, build_runner, setup_logging
from rllm_robustness.constants import CONTINUATION_DECODING
from rllm_robustness.continuation import (
    continuation_model_for_row,
    parse_model_csv,
    separator_for,
)
from rllm_robustness.io import JsonlBatchWriter, batched, existing_ids, load_records, stable_id

logger = logging.getLogger("continue_interventions")


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample continuations from intervened CoT prefixes.")
    add_common_io_args(parser)
    add_runner_args(parser)
    parser.add_argument("--model", default=None, help="Override continuation model. Defaults to row model_name.")
    parser.add_argument(
        "--models",
        default=None,
        help=(
            "Comma-separated continuation models to alternate by input batch. "
            "Useful for trace-swapping or continuation-with-other-model ablations."
        ),
    )
    parser.add_argument("--num-completions", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=32768)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    setup_logging(args.log_level)
    alternating_models = parse_model_csv(args.models)
    if args.model and alternating_models:
        raise ValueError("Use either --model or --models, not both")

    rows = load_records(args.input, split=args.split, limit=args.limit)
    seen = existing_ids(args.output) if args.resume else set()
    writer = JsonlBatchWriter(args.output, flush_every_batches=args.flush_every_batches)
    runner = build_runner(args)
    decoding = replace(
        CONTINUATION_DECODING,
        n=args.num_completions,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed,
        max_new_tokens=args.max_new_tokens,
    )

    total = 0
    for batch_index, batch in enumerate(batched(rows, args.batch_size)):
        by_model: dict[str, list[dict]] = {}
        for row in batch:
            model_name, schedule, schedule_index = continuation_model_for_row(
                row,
                override_model=args.model,
                alternating_models=alternating_models,
                batch_index=batch_index,
            )
            record_id = stable_id(row.get("record_id"), model_name, "continuations", args.num_completions)
            if record_id in seen:
                continue
            row = dict(row)
            row["_continuation_record_id"] = record_id
            row["_continuation_schedule"] = schedule
            row["_continuation_schedule_index"] = schedule_index
            by_model.setdefault(model_name, []).append(row)

        out_rows: list[dict] = []
        for model_name, model_rows in by_model.items():
            logger.info("Continuing %s rows with %s", len(model_rows), model_name)
            prompts = [str(row.get("question", "")) for row in model_rows]
            traces = [str(row.get("mutated_answer_content", "")) for row in model_rows]
            outputs = runner.generate(
                prompts,
                reasoning_traces=traces,
                model_name=model_name,
                decoding=decoding,
            )
            for row, completions in zip(model_rows, outputs):
                prefix = str(row.get("mutated_answer_content", ""))
                sep = separator_for(prefix)
                complete_answers = [prefix + sep + completion for completion in completions]
                out = dict(row)
                out.pop("_continuation_record_id", None)
                out.pop("_continuation_schedule", None)
                out.pop("_continuation_schedule_index", None)
                out.update(
                    {
                        "record_id": row["_continuation_record_id"],
                        "intervention_record_id": row.get("record_id", ""),
                        "source_model_name": row.get("model_name", ""),
                        "continuation_model": model_name,
                        "continuation_model_schedule": row["_continuation_schedule"],
                        "continuation_model_schedule_index": row["_continuation_schedule_index"],
                        "continuations": completions,
                        "complete_answers": complete_answers,
                        "num_completions": args.num_completions,
                        "continuation_temperature": decoding.temperature,
                        "continuation_top_p": decoding.top_p,
                        "continuation_seed": decoding.seed,
                        "continuation_max_new_tokens": decoding.max_new_tokens,
                    }
                )
                out_rows.append(out)
        total += len(out_rows)
        writer.add_batch(out_rows)
        logger.info("Wrote %s continued rows so far", total)

    writer.close()
    runner.unload_model()
    logger.info("Done. Wrote %s rows to %s", total, args.output)


if __name__ == "__main__":
    main()
