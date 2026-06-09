import random

from rllm_robustness.constants import (
    ADVERSARIAL_INSERT_WRONG_FACT,
    BENIGN_COMPLETE_STEP,
    BENIGN_REWRITE_TRACE,
    NEUTRAL_ADD_RANDOM_TEXT,
    NEUTRAL_INSERT_RANDOM_CHARACTERS,
)
from rllm_robustness.inference import MockRunner
from rllm_robustness.interventions import (
    append_marker_to_intervention_row,
    apply_llm_intervention_batch,
    apply_neutral_intervention_batch,
    build_complete_step_prompts,
    insert_random_characters,
    prepare_inputs_for_batch,
)


def sample_row():
    return {
        "id": "sample-1",
        "domain": "MATH",
        "question": "What is 2+2?",
        "reference_answer": "4",
        "model_name": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        "answer_content": (
            "<think>I need to add the numbers.\n\n"
            "2 plus 2 equals 4.\n\n"
            "Therefore the final answer is 4.</think>\n"
            "\\boxed{4}"
        ),
    }


def prepared_item():
    return prepare_inputs_for_batch([sample_row()], 0.3)[0]


def test_prepare_inputs_segments_reasoning_prefix():
    item = prepared_item()
    assert item.reasoning.startswith("I need to add")
    assert item.prefix_segments == ["I need to add the numbers.", "2 plus 2 equals 4."]
    assert item.cutoff_index == 1


def test_complete_step_prompt_uses_context_before_replaced_step():
    item = prepared_item()
    prompt = build_complete_step_prompts([item])[0]
    assert "I need to add the numbers." in prompt
    assert "2 plus 2 equals 4." not in prompt
    assert prompt.endswith("Next step:")


def test_random_character_insertion_is_deterministic_and_grows_text():
    rng_a = random.Random(7)
    rng_b = random.Random(7)
    mutated_a = insert_random_characters("abcdef", ratio=0.5, rng=rng_a)
    mutated_b = insert_random_characters("abcdef", ratio=0.5, rng=rng_b)
    assert mutated_a == mutated_b
    assert len(mutated_a) == 9


def test_neutral_random_characters_mutates_last_prefix_segment():
    item = prepared_item()
    rows = apply_neutral_intervention_batch(
        [item],
        intervention=NEUTRAL_INSERT_RANDOM_CHARACTERS,
        timestep=0.3,
        random_seed=11,
        wikipedia_pool=None,
    )
    assert len(rows) == 1
    row = rows[0]
    assert row["intervention"] == NEUTRAL_INSERT_RANDOM_CHARACTERS
    assert row["intervention_text"] != item.prefix_segments[-1]
    assert row["mutated_num_segments"] == 2
    assert "I need to add the numbers." in row["mutated_answer_content"]


def test_neutral_wikipedia_replaces_last_prefix_segment():
    item = prepared_item()
    rows = apply_neutral_intervention_batch(
        [item],
        intervention=NEUTRAL_ADD_RANDOM_TEXT,
        timestep=0.3,
        random_seed=11,
        wikipedia_pool=["A long unrelated encyclopedia paragraph about plate tectonics."],
    )
    row = rows[0]
    assert row["intervention_text"].startswith("A long unrelated")
    assert "2 plus 2 equals 4." not in row["mutated_answer_content"]
    assert row["wikipedia_pool_index"] == 0


def test_llm_complete_step_replaces_last_segment_with_generated_step():
    item = prepared_item()
    rows = apply_llm_intervention_batch(
        [item],
        intervention=BENIGN_COMPLETE_STEP,
        timestep=0.3,
        runner=MockRunner(),
        intervention_model="mock-intervention-model",
    )
    row = rows[0]
    assert row["intervention"] == BENIGN_COMPLETE_STEP
    assert row["intervention_model"] == "mock-intervention-model"
    assert row["intervention_text"].startswith("Mock completion 1")
    assert "2 plus 2 equals 4." not in row["mutated_answer_content"]


def test_llm_wrong_fact_appends_a_new_segment():
    item = prepared_item()
    rows = apply_llm_intervention_batch(
        [item],
        intervention=ADVERSARIAL_INSERT_WRONG_FACT,
        timestep=0.3,
        runner=MockRunner(),
        intervention_model="mock-intervention-model",
    )
    row = rows[0]
    assert row["intervention"] == ADVERSARIAL_INSERT_WRONG_FACT
    assert row["mutated_num_segments"] == 3
    assert row["mutated_answer_content"].count("\n\n") == 2


def test_llm_rewrite_trace_uses_generated_trace_and_skips_old_metadata():
    item = prepared_item()
    rows = apply_llm_intervention_batch(
        [item],
        intervention=BENIGN_REWRITE_TRACE,
        timestep=0.3,
        runner=MockRunner(),
        intervention_model="mock-intervention-model",
    )
    row = rows[0]
    assert row["intervention"] == BENIGN_REWRITE_TRACE
    assert row["mutated_num_segments"] == 1
    assert row["mutated_answer_content"].startswith("Mock completion 1")


def test_append_marker_to_existing_intervention_row_updates_prompt_and_id():
    item = prepared_item()
    row = apply_neutral_intervention_batch(
        [item],
        intervention=NEUTRAL_ADD_RANDOM_TEXT,
        timestep=0.3,
        random_seed=11,
        wikipedia_pool=["A long unrelated encyclopedia paragraph about plate tectonics."],
    )[0]
    marked = append_marker_to_intervention_row(row, "Wait")

    assert marked["record_id"] != row["record_id"]
    assert marked["base_intervention_record_id"] == row["record_id"]
    assert marked["append_after_intervention"] == "Wait"
    assert marked["mutated_answer_content"].endswith("\n\nWait")
    assert marked["intervention_text"].endswith("\n\nWait")
