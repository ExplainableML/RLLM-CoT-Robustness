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
    ANSWER_JUDGE_DECODING,
    ANSWER_JUDGE_SYSTEM_PROMPT,
    DEFAULT_JUDGE_MODEL,
)
from rllm_robustness.io import JsonlBatchWriter, batched, existing_ids, load_records, stable_id

logger = logging.getLogger("judge_answers")


def extract_answer_for_judging(answer_content: str | None) -> str:
    text = "" if answer_content is None else str(answer_content)
    tag = "</think>"
    if tag in text:
        return text[text.rindex(tag) + len(tag) :].strip()
    return text.strip()


def build_judge_prompt(question: str, reference_answer: str, answer_content: str) -> str:
    return (
        f"Question:\n{question}\n\n"
        f"Reference Answer:\n{reference_answer}\n\n"
        f"Answer Content (extract):\n{answer_content}\n\n"
        "Is the Reference Answer present in the Answer Content? Respond with only "
        "'Correct' or 'Incorrect'."
    )


def token_truncate(text: str, tokenizer, max_tokens: int) -> str:
    if tokenizer is None:
        return text
    token_ids = tokenizer.encode(text)
    if len(token_ids) <= max_tokens:
        return text
    return tokenizer.decode(
        token_ids[-max_tokens:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Judge original answers or sampled continuations.")
    add_common_io_args(parser)
    add_runner_args(parser)
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    parser.add_argument("--mode", choices=["single", "completions"], default="single")
    parser.add_argument("--answer-column", default="answer_content")
    parser.add_argument("--answers-list-column", default="complete_answers")
    parser.add_argument("--score-column", default="verifier_score")
    parser.add_argument("--scores-list-column", default="verifier_scores")
    parser.add_argument("--max-answer-chars", type=int, default=8000)
    parser.add_argument("--max-prompt-tokens", type=int, default=30000)
    parser.add_argument("--max-new-tokens", type=int, default=10)
    parser.add_argument(
        "--on-error",
        choices=["raise", "score_zero"],
        default="raise",
        help="Use score_zero only to reproduce the old scripts' failure behavior.",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    setup_logging(args.log_level)

    rows = load_records(args.input, split=args.split, limit=args.limit)
    seen = existing_ids(args.output) if args.resume else set()
    writer = JsonlBatchWriter(args.output, flush_every_batches=args.flush_every_batches)
    runner = build_runner(args)
    decoding = replace(ANSWER_JUDGE_DECODING, max_new_tokens=args.max_new_tokens)
    tokenizer = None
    if args.backend == "vllm":
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.judge_model, trust_remote_code=True)

    total = 0
    for batch in batched(rows, args.batch_size):
        prompts = []
        prompt_index: list[tuple[int, int | None]] = []
        out_rows = []
        for row in batch:
            record_id = stable_id(row.get("record_id"), args.judge_model, args.mode, "judge")
            if args.resume and record_id in seen:
                continue
            row = dict(row)
            row["record_id"] = record_id
            row_idx = len(out_rows)
            out_rows.append(row)
            if args.mode == "single":
                extracted = extract_answer_for_judging(row.get(args.answer_column))
                answer_extract = extracted[-args.max_answer_chars :]
                prompt = build_judge_prompt(
                    str(row.get("question", "")),
                    str(row.get("reference_answer", "")),
                    answer_extract,
                )
                prompts.append(token_truncate(prompt, tokenizer, args.max_prompt_tokens))
                prompt_index.append((row_idx, None))
            else:
                answers = row.get(args.answers_list_column) or []
                if not isinstance(answers, list):
                    raise ValueError(f"{args.answers_list_column} must contain a list")
                row[args.scores_list_column] = [None] * len(answers)
                for answer_idx, answer in enumerate(answers):
                    extracted = extract_answer_for_judging(answer)
                    answer_extract = extracted[-args.max_answer_chars :]
                    prompt = build_judge_prompt(
                        str(row.get("question", "")),
                        str(row.get("reference_answer", "")),
                        answer_extract,
                    )
                    prompts.append(token_truncate(prompt, tokenizer, args.max_prompt_tokens))
                    prompt_index.append((row_idx, answer_idx))

        try:
            raw_outputs = runner.generate(
                prompts,
                system_prompts=ANSWER_JUDGE_SYSTEM_PROMPT,
                model_name=args.judge_model,
                decoding=decoding,
            )
        except Exception:
            if args.on_error == "raise":
                raise
            logger.exception("Judging failed; scoring this batch as zero")
            raw_outputs = [["Incorrect"] for _ in prompts]

        for (row_idx, answer_idx), output in zip(prompt_index, raw_outputs):
            judgment = output[0].strip() if output else ""
            score = 1 if judgment == "Correct" else 0
            if judgment not in {"Correct", "Incorrect"}:
                logger.warning("Unexpected judge output %r; scoring as 0", judgment)
            if answer_idx is None:
                out_rows[row_idx][args.score_column] = score
                out_rows[row_idx]["judge_output"] = judgment
            else:
                out_rows[row_idx][args.scores_list_column][answer_idx] = score
                out_rows[row_idx].setdefault("judge_outputs", [None] * len(out_rows[row_idx][args.scores_list_column]))
                out_rows[row_idx]["judge_outputs"][answer_idx] = judgment

        total += len(out_rows)
        writer.add_batch(out_rows)
        logger.info("Judged %s rows so far", total)

    writer.close()
    runner.unload_model()
    logger.info("Done. Wrote %s rows to %s", total, args.output)


if __name__ == "__main__":
    main()
