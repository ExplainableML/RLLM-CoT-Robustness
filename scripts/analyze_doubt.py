#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rllm_robustness.cli_common import add_common_io_args, add_runner_args, build_runner, setup_logging
from rllm_robustness.constants import DEFAULT_JUDGE_MODEL, DOUBT_DECODING, DOUBT_PROMPT_TEMPLATE, DOUBT_SYSTEM_PROMPT
from rllm_robustness.io import JsonlBatchWriter, batched, existing_ids, load_records, stable_id
from rllm_robustness.segmentation import first_n_sentences, segment_trace

logger = logging.getLogger("analyze_doubt")


def sentence_units(text: str, *, max_sentences: int, spacy_nlp=None) -> list[str]:
    segments, _ = segment_trace(text)
    sentences: list[str] = []
    if spacy_nlp is None:
        for segment in segments:
            sentences.extend(first_n_sentences(segment, max_sentences))
            if len(sentences) >= max_sentences:
                break
        return sentences[:max_sentences]
    for segment in segments:
        if not str(segment).strip():
            continue
        doc = spacy_nlp(str(segment).strip())
        sentences.extend(sent.text.strip() for sent in doc.sents if sent.text.strip())
        if len(sentences) >= max_sentences:
            break
    return sentences[:max_sentences]


def segment_units(text: str, *, max_segments: int) -> list[str]:
    segments, _ = segment_trace(text)
    return [segment.strip() for segment in segments if segment and segment.strip()][:max_segments]


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify post-intervention text units for doubt.")
    add_common_io_args(parser)
    add_runner_args(parser)
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    parser.add_argument("--unit", choices=["sentence", "segment"], default="sentence")
    parser.add_argument("--text-source", choices=["continuations", "complete_answers"], default="continuations")
    parser.add_argument("--max-sentences", type=int, default=20)
    parser.add_argument("--max-segments", type=int, default=10)
    parser.add_argument("--max-unit-chars", type=int, default=1000)
    parser.add_argument("--spacy-model", default="sentencizer_en")
    parser.add_argument("--max-new-tokens", type=int, default=10)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    setup_logging(args.log_level)

    spacy_nlp = None
    if args.unit == "sentence":
        import spacy

        if args.spacy_model == "sentencizer_en":
            spacy_nlp = spacy.blank("en")
            spacy_nlp.add_pipe("sentencizer")
        else:
            spacy_nlp = spacy.load(args.spacy_model)

    rows = load_records(args.input, split=args.split, limit=args.limit)
    seen = existing_ids(args.output) if args.resume else set()
    writer = JsonlBatchWriter(args.output, flush_every_batches=args.flush_every_batches)
    runner = build_runner(args)
    decoding = replace(DOUBT_DECODING, max_new_tokens=args.max_new_tokens)

    total = 0
    for batch in batched(rows, args.batch_size):
        prompts = []
        prompt_index: list[tuple[int, int, list[str]]] = []
        out_rows: list[dict] = []
        for row in batch:
            texts = row.get(args.text_source) or []
            if not isinstance(texts, list):
                raise ValueError(f"{args.text_source} must contain a list")
            verifier_scores = row.get("verifier_scores") or []
            for completion_idx, text in enumerate(texts):
                record_id = stable_id(row.get("record_id"), completion_idx, "doubt", args.unit)
                if record_id in seen:
                    continue
                units = (
                    sentence_units(str(text), max_sentences=args.max_sentences, spacy_nlp=spacy_nlp)
                    if args.unit == "sentence"
                    else segment_units(str(text), max_segments=args.max_segments)
                )
                units = [unit[: args.max_unit_chars] for unit in units]
                base = {
                    "record_id": record_id,
                    "source_record_id": row.get("record_id", ""),
                    "completion_index": completion_idx,
                    "model_name": row.get("continuation_model", row.get("model_name", "")),
                    "domain": row.get("domain", ""),
                    "intervention": row.get("intervention", ""),
                    "append_after_intervention": row.get("append_after_intervention"),
                    "target_timestep": row.get("target_timestep"),
                    "verifier_score": verifier_scores[completion_idx]
                    if completion_idx < len(verifier_scores)
                    else None,
                    "units": units,
                    "doubt_flags": [],
                    "judge_outputs": [],
                }
                row_pos = len(out_rows)
                out_rows.append(base)
                for unit in units:
                    prompts.append(DOUBT_PROMPT_TEMPLATE.format(unit_text=unit.replace("\n", " ")))
                    prompt_index.append((row_pos, completion_idx, units))

        raw_outputs = runner.generate(
            prompts,
            system_prompts=DOUBT_SYSTEM_PROMPT,
            model_name=args.judge_model,
            decoding=decoding,
        )
        for (row_pos, _completion_idx, _units), output in zip(prompt_index, raw_outputs):
            text = output[0].strip() if output else ""
            answer = text.lower()
            flag = 1 if answer.startswith("yes") else 0
            if not (answer.startswith("yes") or answer.startswith("no")):
                logger.warning("Unexpected doubt judge output %r; scoring as 0", text)
            out_rows[row_pos]["doubt_flags"].append(flag)
            out_rows[row_pos]["judge_outputs"].append(text)

        total += len(out_rows)
        writer.add_batch(out_rows)
        logger.info("Wrote %s doubt rows so far", total)

    writer.close()
    runner.unload_model()
    logger.info("Done. Wrote %s rows to %s", total, args.output)


if __name__ == "__main__":
    main()
