#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rllm_robustness.cli_common import add_common_io_args, add_runner_args, build_runner, setup_logging
from rllm_robustness.constants import (
    DEFAULT_ORIGINAL_GENERATION,
    DEFAULT_SYSTEM_PROMPT,
    MODEL_GENERATION_OVERRIDES,
    MODEL_SYSTEM_PROMPTS,
)
from rllm_robustness.io import JsonlBatchWriter, batched, existing_ids, load_records, stable_id

logger = logging.getLogger("generate_traces")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate original CoT traces locally.")
    add_common_io_args(parser)
    add_runner_args(parser)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--question-column",
        default="question",
        help="Column containing the task prompt.",
    )
    parser.add_argument("--reference-column", default="reference_answer")
    parser.add_argument("--id-column", default="id")
    parser.add_argument(
        "--reasoning-instruction",
        default="Please reason step by step, and put your final answer within \\boxed{}.",
    )
    parser.add_argument("--system-prompt", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    setup_logging(args.log_level)

    rows = load_records(args.input, split=args.split, limit=args.limit)
    runner = build_runner(args)
    decoding = MODEL_GENERATION_OVERRIDES.get(args.model, DEFAULT_ORIGINAL_GENERATION)
    if args.max_new_tokens is not None:
        decoding = replace(decoding, max_new_tokens=args.max_new_tokens)
    if args.temperature is not None:
        decoding = replace(decoding, temperature=args.temperature)
    if args.top_p is not None:
        decoding = replace(decoding, top_p=args.top_p)
    if args.top_k is not None:
        decoding = replace(decoding, top_k=args.top_k)
    if args.seed is not None:
        decoding = replace(decoding, seed=args.seed)

    system_prompt = args.system_prompt
    if system_prompt is None:
        system_prompt = MODEL_SYSTEM_PROMPTS.get(args.model, DEFAULT_SYSTEM_PROMPT)

    seen = existing_ids(args.output) if args.resume else set()
    writer = JsonlBatchWriter(args.output, flush_every_batches=args.flush_every_batches)
    total = 0
    for batch in batched(rows, args.batch_size):
        prompts = []
        active_rows = []
        for row in batch:
            raw_question = str(row.get(args.question_column, ""))
            prompt_text = f"{raw_question}\n\n{args.reasoning_instruction}"
            row_id = row.get(args.id_column) or stable_id(raw_question, row.get(args.reference_column, ""))
            record_id = stable_id(row_id, args.model, "original_trace")
            if record_id in seen:
                continue
            active_rows.append((row, row_id, prompt_text, record_id))
            prompts.append(prompt_text)
        if not prompts:
            continue
        outputs = runner.generate(
            prompts,
            system_prompts=system_prompt,
            model_name=args.model,
            decoding=decoding,
        )
        out_rows = []
        for (row, row_id, prompt_text, record_id), output_list in zip(active_rows, outputs):
            answer_content = output_list[0] if output_list else ""
            out_rows.append(
                {
                    "record_id": record_id,
                    "id": row_id,
                    "question": prompt_text,
                    "raw_question": row.get(args.question_column, ""),
                    "reference_answer": row.get(args.reference_column, ""),
                    "metadata": row.get("metadata", {}),
                    "domain": row.get("domain", ""),
                    "model_name": args.model,
                    "system_prompt": system_prompt,
                    "answer_content": answer_content,
                    "generation_temperature": decoding.temperature,
                    "generation_top_p": decoding.top_p,
                    "generation_top_k": decoding.top_k,
                    "generation_seed": decoding.seed,
                    "generation_max_new_tokens": decoding.max_new_tokens,
                }
            )
        total += len(out_rows)
        writer.add_batch(out_rows)
        logger.info("Generated %s traces so far", total)
    writer.close()
    runner.unload_model()
    logger.info("Done. Wrote %s rows to %s", total, args.output)


if __name__ == "__main__":
    main()

