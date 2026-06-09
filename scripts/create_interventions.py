#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rllm_robustness.cli_common import add_common_io_args, add_runner_args, build_runner, setup_logging
from rllm_robustness.constants import (
    ADVERSARIAL_CONTINUE_UNRELATED,
    ADVERSARIAL_CONTINUE_WITH_WRONG_REASONING,
    ADVERSARIAL_INSERT_WRONG_FACT,
    BENIGN_COMPLETE_STEP,
    BENIGN_REWRITE_TRACE,
    INTERVENTION_MODEL,
    INTERVENTIONS,
    NEUTRAL_ADD_RANDOM_TEXT,
    NEUTRAL_INSERT_RANDOM_CHARACTERS,
    TIMESTEPS,
)
from rllm_robustness.interventions import (
    apply_llm_intervention_batch,
    apply_neutral_intervention_batch,
    load_wikipedia_pool,
    prepare_inputs_for_batch,
)
from rllm_robustness.io import JsonlBatchWriter, batched, existing_ids, load_records

logger = logging.getLogger("create_interventions")

LLM_INTERVENTIONS = {
    BENIGN_COMPLETE_STEP,
    BENIGN_REWRITE_TRACE,
    ADVERSARIAL_CONTINUE_WITH_WRONG_REASONING,
    ADVERSARIAL_INSERT_WRONG_FACT,
    ADVERSARIAL_CONTINUE_UNRELATED,
}
NEUTRAL_INTERVENTIONS = {NEUTRAL_ADD_RANDOM_TEXT, NEUTRAL_INSERT_RANDOM_CHARACTERS}


def parse_csv(values: str, cast):
    return [cast(value.strip()) for value in values.split(",") if value.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Create intervened CoT prefixes locally.")
    add_common_io_args(parser)
    add_runner_args(parser)
    parser.add_argument("--interventions", default=",".join(INTERVENTIONS))
    parser.add_argument("--timesteps", default=",".join(str(t) for t in TIMESTEPS))
    parser.add_argument("--intervention-model", default=INTERVENTION_MODEL)
    parser.add_argument("--min-steps", type=int, default=2)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--wikipedia-pool-size", type=int, default=1000)
    parser.add_argument("--allow-wikipedia-fallback", action="store_true")
    parser.add_argument(
        "--append-after-intervention",
        default=None,
        help="Ablation hook, e.g. 'Wait'. Appended after the intervention text.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="Override intervention generation length for smoke tests.",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    setup_logging(args.log_level)

    interventions = parse_csv(args.interventions, str)
    timesteps = parse_csv(args.timesteps, float)
    unknown = sorted(set(interventions) - set(INTERVENTIONS))
    if unknown:
        raise ValueError(f"Unknown interventions: {unknown}")

    rows = load_records(args.input, split=args.split, limit=args.limit)
    seen = existing_ids(args.output) if args.resume else set()
    writer = JsonlBatchWriter(args.output, flush_every_batches=args.flush_every_batches)
    runner = build_runner(args) if any(name in LLM_INTERVENTIONS for name in interventions) else None
    wikipedia_pool = None
    if NEUTRAL_ADD_RANDOM_TEXT in interventions:
        wikipedia_pool = load_wikipedia_pool(
            pool_size=args.wikipedia_pool_size,
            seed=args.random_seed,
            allow_fallback=args.allow_wikipedia_fallback,
        )

    total = 0
    for timestep in timesteps:
        for intervention in interventions:
            logger.info("Creating intervention=%s timestep=%s", intervention, timestep)
            for batch in batched(rows, args.batch_size):
                items = prepare_inputs_for_batch(batch, timestep, min_steps=args.min_steps)
                if intervention in NEUTRAL_INTERVENTIONS:
                    out_rows = apply_neutral_intervention_batch(
                        items,
                        intervention=intervention,
                        timestep=timestep,
                        random_seed=args.random_seed,
                        wikipedia_pool=wikipedia_pool,
                        append_after_intervention=args.append_after_intervention,
                    )
                else:
                    assert runner is not None
                    out_rows = apply_llm_intervention_batch(
                        items,
                        intervention=intervention,
                        timestep=timestep,
                        runner=runner,
                        intervention_model=args.intervention_model,
                        append_after_intervention=args.append_after_intervention,
                        max_new_tokens_override=args.max_new_tokens,
                    )
                if seen:
                    out_rows = [row for row in out_rows if row["record_id"] not in seen]
                total += len(out_rows)
                writer.add_batch(out_rows)
                logger.info("Wrote %s intervention rows so far", total)
    writer.close()
    if runner is not None:
        runner.unload_model()
    logger.info("Done. Wrote %s rows to %s", total, args.output)


if __name__ == "__main__":
    main()

