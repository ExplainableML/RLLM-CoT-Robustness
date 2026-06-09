from __future__ import annotations

import logging
import random
import string
from dataclasses import dataclass
from typing import Iterable

from .constants import (
    ADVERSARIAL_CONTINUE_UNRELATED,
    ADVERSARIAL_CONTINUE_WITH_WRONG_REASONING,
    ADVERSARIAL_INSERT_WRONG_FACT,
    BENIGN_COMPLETE_STEP,
    BENIGN_REWRITE_TRACE,
    COMPLETE_STEP_SYSTEM_PROMPT,
    INTERVENTION_CATEGORY,
    INTERVENTION_DECODING,
    INTERVENTION_MODEL,
    NEUTRAL_ADD_RANDOM_TEXT,
    NEUTRAL_INSERT_RANDOM_CHARACTERS,
    REWRITE_TRACE_SYSTEM_PROMPT,
    UNRELATED_COT_SYSTEM_PROMPT,
    UNRELATED_TOPICS,
    WRONG_FACT_SYSTEM_PROMPT,
    WRONG_REASONING_SYSTEM_PROMPT,
)
from .inference import MockRunner, VllmRunner
from .io import stable_id
from .segmentation import (
    extract_reasoning,
    restore_trace_from_segments,
    segment_trace,
    trace_prefix_at_timestep,
)

logger = logging.getLogger(__name__)


FALLBACK_WIKIPEDIA_PARAGRAPHS = [
    (
        "The Federal government's power to end slavery was limited by the "
        "Constitution before 1865, when many legal questions about federal and "
        "state authority were resolved through amendments and court decisions."
    ),
    (
        "The Great Barrier Reef is the world's largest coral reef system, composed "
        "of thousands of individual reefs and islands extending along the coast of "
        "Queensland, Australia."
    ),
    (
        "Plate tectonics describes the movement of large sections of Earth's "
        "lithosphere, whose interactions explain many earthquakes, mountain "
        "ranges, ocean trenches, and volcanic arcs."
    ),
]


@dataclass
class InterventionInput:
    row: dict
    reasoning: str
    segments: list[str]
    metadata: list[dict]
    prefix_segments: list[str]
    cutoff_index: int
    actual_timestep: float


def prepare_intervention_input(row: dict, timestep: float, *, min_steps: int = 2) -> InterventionInput:
    reasoning = extract_reasoning(row.get("answer_content", ""))
    segments, metadata = segment_trace(reasoning)
    prefix_segments, cutoff_index, actual_timestep = trace_prefix_at_timestep(
        segments,
        timestep,
        min_steps=min_steps,
    )
    return InterventionInput(
        row=row,
        reasoning=reasoning,
        segments=segments,
        metadata=metadata,
        prefix_segments=prefix_segments,
        cutoff_index=cutoff_index,
        actual_timestep=actual_timestep,
    )


def build_complete_step_prompts(items: list[InterventionInput]) -> list[str]:
    prompts = []
    for item in items:
        reasoning_so_far = "\n".join(item.prefix_segments[:-1])
        prompts.append(
            f"Problem:\n{item.row.get('question', '')}\n\n"
            f"Reasoning so far:\n{reasoning_so_far}\n\n"
            "Next step:"
        )
    return prompts


def build_rewrite_prompts(items: list[InterventionInput]) -> list[str]:
    prompts = []
    for item in items:
        original_trace = "\n\n".join(item.prefix_segments)
        prompts.append(
            f"Problem:\n{item.row.get('question', '')}\n\n"
            f"Original reasoning trace:\n{original_trace}\n\n"
            "Rewritten trace:"
        )
    return prompts


def build_wrong_reasoning_prompts(items: list[InterventionInput]) -> list[str]:
    prompts = []
    for item in items:
        reasoning_so_far = "\n".join(item.prefix_segments[:-1])
        prompts.append(
            f"Problem:\n{item.row.get('question', '')}\n\n"
            f"Reasoning so far:\n{reasoning_so_far}\n\n"
            "Next (incorrect) step:"
        )
    return prompts


def build_wrong_fact_prompts(items: list[InterventionInput]) -> list[str]:
    prompts = []
    for item in items:
        reasoning_so_far = "\n".join(item.prefix_segments)
        prompts.append(
            f"Problem:\n{item.row.get('question', '')}\n\n"
            f"Reasoning so far:\n{reasoning_so_far}\n\n"
            "Wrong statement:"
        )
    return prompts


def build_unrelated_prompts(
    items: list[InterventionInput],
    *,
    seed: int = 80129,
) -> tuple[list[str], list[str]]:
    rng = random.Random(seed)
    topics = rng.choices(UNRELATED_TOPICS, k=len(items))
    prompts = [f"Topic: {topic}\nUnrelated reasoning step about {topic}:" for topic in topics]
    return prompts, topics


def insert_random_characters(segment: str, *, ratio: float = 0.5, rng: random.Random) -> str:
    ratio = max(0.0, min(1.0, ratio))
    if not segment or ratio == 0.0:
        return segment
    n_insertions = max(1, int(round(ratio * len(segment))))
    positions = sorted(rng.randint(0, len(segment)) for _ in range(n_insertions))
    chars = list(segment)
    alphabet = string.ascii_letters + string.digits + string.punctuation + " "
    for offset, position in enumerate(positions):
        chars.insert(position + offset, rng.choice(alphabet))
    return "".join(chars)


def load_wikipedia_pool(
    *,
    pool_size: int = 1000,
    seed: int = 42,
    allow_fallback: bool = False,
) -> list[str]:
    """Load and shuffle an English Wikipedia paragraph pool for neutral insertions."""
    if allow_fallback:
        logger.warning("Using bundled fallback Wikipedia paragraphs for this run")
        return FALLBACK_WIKIPEDIA_PARAGRAPHS.copy()

    try:
        from datasets import load_dataset

        dataset = load_dataset(
            "wikimedia/wikipedia",
            "20231101.en",
            split="train",
            streaming=True,
        )
        rows = list(dataset.take(pool_size))
        paragraphs: list[str] = []
        for row in rows:
            for paragraph in str(row.get("text", "")).split("\n\n"):
                if len(paragraph) >= 100:
                    paragraphs.append(paragraph)
        if not paragraphs:
            raise RuntimeError("Wikipedia stream returned no paragraphs with >=100 characters")
        rng = random.Random(seed)
        rng.shuffle(paragraphs)
        return paragraphs
    except Exception as exc:
        raise RuntimeError(
            "Could not load the Wikipedia paragraph pool. Re-run with "
            "--allow-wikipedia-fallback for smoke tests, or provide network/cache access."
        ) from exc


def _restore_for_intervention(
    item: InterventionInput,
    mutated_segments: list[str],
    *,
    skip_metadata: bool = False,
) -> str:
    return restore_trace_from_segments(
        mutated_segments,
        item.metadata[: len(mutated_segments)],
        skip_metadata=skip_metadata,
    )


def _base_output_record(
    item: InterventionInput,
    intervention: str,
    timestep: float,
    mutated_segments: list[str],
    mutated_answer_content: str,
    intervention_text: str,
    *,
    append_after_intervention: str | None,
    extra: dict | None = None,
) -> dict:
    row = item.row
    if append_after_intervention:
        suffix = append_after_intervention
        if not mutated_answer_content.endswith(("\n", " ")):
            mutated_answer_content += "\n\n"
        mutated_answer_content += suffix
        intervention_text = f"{intervention_text}\n\n{suffix}" if intervention_text else suffix

    record = {
        "record_id": stable_id(
            row.get("id") or row.get("question"),
            row.get("model_name", ""),
            intervention,
            timestep,
            append_after_intervention or "",
        ),
        "source_id": row.get("id", ""),
        "question": row.get("question", ""),
        "reference_answer": row.get("reference_answer", ""),
        "metadata": row.get("metadata", {}),
        "model_name": row.get("model_name", ""),
        "domain": row.get("domain", ""),
        "answer_content": row.get("answer_content", ""),
        "original_reasoning": item.reasoning,
        "original_num_segments": len(item.segments),
        "intervention": intervention,
        "intervention_category": INTERVENTION_CATEGORY[intervention],
        "target_timestep": timestep,
        "actual_timestep": item.actual_timestep,
        "cutoff_index": item.cutoff_index,
        "mutated_answer_content": mutated_answer_content,
        "intervention_text": intervention_text,
        "mutated_num_segments": len(mutated_segments),
        "append_after_intervention": append_after_intervention,
    }
    if extra:
        record.update(extra)
    return record


def append_marker_to_intervention_row(row: dict, marker: str) -> dict:
    """Return an intervention row with a doubt marker appended at the resume point."""
    if not marker:
        return dict(row)

    out = dict(row)
    mutated_answer_content = str(out.get("mutated_answer_content", ""))
    if mutated_answer_content and not mutated_answer_content.endswith(("\n", " ")):
        mutated_answer_content += "\n\n"
    mutated_answer_content += marker

    intervention_text = str(out.get("intervention_text", ""))
    intervention_text = f"{intervention_text}\n\n{marker}" if intervention_text else marker

    out["base_intervention_record_id"] = row.get("record_id", "")
    out["record_id"] = stable_id(row.get("record_id", ""), marker, "append_marker")
    out["mutated_answer_content"] = mutated_answer_content
    out["intervention_text"] = intervention_text
    out["append_after_intervention"] = marker
    return out


def apply_neutral_intervention_batch(
    items: list[InterventionInput],
    *,
    intervention: str,
    timestep: float,
    random_seed: int,
    wikipedia_pool: list[str] | None,
    append_after_intervention: str | None = None,
) -> list[dict]:
    outputs: list[dict] = []
    for index, item in enumerate(items):
        mutated_segments = item.prefix_segments.copy()
        rng = random.Random(random_seed + index)
        if not mutated_segments:
            continue
        if intervention == NEUTRAL_INSERT_RANDOM_CHARACTERS:
            original_last = mutated_segments[-1]
            mutated_last = insert_random_characters(original_last, ratio=0.5, rng=rng)
            mutated_segments[-1] = mutated_last
            intervention_text = mutated_last
            extra = {"random_character_ratio": 0.5}
        elif intervention == NEUTRAL_ADD_RANDOM_TEXT:
            if not wikipedia_pool:
                raise ValueError("wikipedia_pool is required for neutral_add_random_text")
            text = wikipedia_pool[index % len(wikipedia_pool)]
            mutated_segments[-1] = text
            intervention_text = text
            extra = {"wikipedia_pool_index": index % len(wikipedia_pool)}
        else:
            raise ValueError(f"Not a neutral intervention: {intervention}")

        restored = _restore_for_intervention(item, mutated_segments)
        outputs.append(
            _base_output_record(
                item,
                intervention,
                timestep,
                mutated_segments,
                restored,
                intervention_text,
                append_after_intervention=append_after_intervention,
                extra=extra,
            )
        )
    return outputs


def apply_llm_intervention_batch(
    items: list[InterventionInput],
    *,
    intervention: str,
    timestep: float,
    runner: MockRunner | VllmRunner,
    intervention_model: str = INTERVENTION_MODEL,
    append_after_intervention: str | None = None,
    max_new_tokens_override: int | None = None,
) -> list[dict]:
    if not items:
        return []
    decoding = INTERVENTION_DECODING[intervention]
    extra_rows: list[dict] = [{} for _ in items]

    if intervention == BENIGN_COMPLETE_STEP:
        prompts = build_complete_step_prompts(items)
        system_prompt = COMPLETE_STEP_SYSTEM_PROMPT
    elif intervention == BENIGN_REWRITE_TRACE:
        prompts = build_rewrite_prompts(items)
        system_prompt = REWRITE_TRACE_SYSTEM_PROMPT
    elif intervention == ADVERSARIAL_CONTINUE_WITH_WRONG_REASONING:
        prompts = build_wrong_reasoning_prompts(items)
        system_prompt = WRONG_REASONING_SYSTEM_PROMPT
    elif intervention == ADVERSARIAL_INSERT_WRONG_FACT:
        prompts = build_wrong_fact_prompts(items)
        system_prompt = WRONG_FACT_SYSTEM_PROMPT
    elif intervention == ADVERSARIAL_CONTINUE_UNRELATED:
        prompts, topics = build_unrelated_prompts(items, seed=decoding.seed or 80129)
        system_prompt = UNRELATED_COT_SYSTEM_PROMPT
        extra_rows = [{"unrelated_topic": topic} for topic in topics]
    else:
        raise ValueError(f"Unsupported LLM intervention: {intervention}")

    raw_outputs = runner.generate(
        prompts,
        system_prompts=system_prompt,
        model_name=intervention_model,
        decoding=decoding,
        max_new_tokens_override=max_new_tokens_override,
    )

    outputs: list[dict] = []
    for item, output_list, extra in zip(items, raw_outputs, extra_rows):
        generated = output_list[0].strip() if output_list else ""
        if intervention == BENIGN_REWRITE_TRACE:
            mutated_segments = [segment.strip() for segment in generated.split("\n\n") if segment.strip()]
            if not mutated_segments:
                mutated_segments = [""]
            restored = _restore_for_intervention(item, mutated_segments, skip_metadata=True)
            intervention_text = restored
        elif intervention == ADVERSARIAL_INSERT_WRONG_FACT:
            mutated_segments = item.prefix_segments + [generated]
            restored = _restore_for_intervention(item, mutated_segments)
            intervention_text = generated
        else:
            mutated_segments = item.prefix_segments[:-1] + [generated]
            restored = _restore_for_intervention(item, mutated_segments)
            intervention_text = generated

        outputs.append(
            _base_output_record(
                item,
                intervention,
                timestep,
                mutated_segments,
                restored,
                intervention_text,
                append_after_intervention=append_after_intervention,
                extra={
                    **extra,
                    "intervention_model": intervention_model,
                    "intervention_temperature": decoding.temperature,
                    "intervention_top_p": decoding.top_p,
                    "intervention_seed": decoding.seed,
                },
            )
        )
    return outputs


def prepare_inputs_for_batch(
    rows: Iterable[dict],
    timestep: float,
    *,
    min_steps: int = 2,
) -> list[InterventionInput]:
    items: list[InterventionInput] = []
    for row in rows:
        item = prepare_intervention_input(row, timestep, min_steps=min_steps)
        if len(item.prefix_segments) >= min_steps:
            items.append(item)
    return items
