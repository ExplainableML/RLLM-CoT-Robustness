import pytest

from rllm_robustness.continuation import (
    continuation_model_for_row,
    parse_model_csv,
    separator_for,
)


def test_separator_preserves_existing_paragraph_boundary():
    assert separator_for("prefix\n\n") == ""
    assert separator_for("prefix\n") == "\n"
    assert separator_for("prefix") == "\n\n"


def test_parse_model_csv_strips_empty_entries():
    assert parse_model_csv(" model-a,model-b, ") == ["model-a", "model-b"]
    assert parse_model_csv(None) == []


def test_continuation_model_defaults_to_source_model():
    model, schedule, index = continuation_model_for_row(
        {"model_name": "source-model"},
        override_model=None,
        alternating_models=[],
        batch_index=0,
    )
    assert model == "source-model"
    assert schedule == "source_model"
    assert index is None


def test_continuation_model_supports_single_override():
    model, schedule, index = continuation_model_for_row(
        {"model_name": "source-model"},
        override_model="override-model",
        alternating_models=[],
        batch_index=3,
    )
    assert model == "override-model"
    assert schedule == "override"
    assert index is None


def test_continuation_model_alternates_by_batch():
    models = ["model-a", "model-b", "model-c"]
    chosen = [
        continuation_model_for_row(
            {"model_name": "source-model"},
            override_model=None,
            alternating_models=models,
            batch_index=batch_index,
        )
        for batch_index in range(5)
    ]
    assert [item[0] for item in chosen] == [
        "model-a",
        "model-b",
        "model-c",
        "model-a",
        "model-b",
    ]
    assert [item[2] for item in chosen] == [0, 1, 2, 0, 1]


def test_continuation_model_rejects_override_and_alternation_together():
    with pytest.raises(ValueError):
        continuation_model_for_row(
            {"model_name": "source-model"},
            override_model="override-model",
            alternating_models=["model-a"],
            batch_index=0,
        )
