from rllm_robustness.segmentation import (
    extract_reasoning,
    restore_trace_from_segments,
    segment_trace,
    trace_prefix_at_timestep,
)
from rllm_robustness.metrics import robustness_from_scores


def test_extract_reasoning_from_think_tags():
    text = "<think>Step one.\n\nStep two.</think>\nAnswer: 4"
    assert extract_reasoning(text) == "Step one.\n\nStep two."


def test_segment_and_restore_round_trip():
    text = " First step. \n\nSecond step."
    segments, metadata = segment_trace(text)
    assert segments == [" First step. ", "Second step."]
    assert restore_trace_from_segments(segments, metadata) == text


def test_trace_prefix_selects_nearest_timestep_with_min_steps():
    segments = ["aaa", "bbb", "cccc"]
    prefix, index, actual_timestep = trace_prefix_at_timestep(segments, 0.1, min_steps=2)
    assert prefix == ["aaa", "bbb"]
    assert index == 1
    assert round(actual_timestep, 2) == 0.6


def test_robustness_thresholds_use_strict_majority():
    metrics = robustness_from_scores([1, 1, 0, 0])
    assert metrics["at_least_once_robust"] == 1
    assert metrics["majority_robust"] == 0
    assert metrics["all_robust"] == 0

    metrics = robustness_from_scores([1, 1, 1, 0])
    assert metrics["majority_robust"] == 1


def test_empty_score_list_is_not_robust():
    metrics = robustness_from_scores([])
    assert metrics["n"] == 0
    assert metrics["at_least_once_robust"] == 0
    assert metrics["majority_robust"] == 0
    assert metrics["all_robust"] == 0
