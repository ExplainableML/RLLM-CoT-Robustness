from __future__ import annotations


def separator_for(prefix: str) -> str:
    if prefix.endswith("\n\n"):
        return ""
    if prefix.endswith("\n"):
        return "\n"
    return "\n\n"


def parse_model_csv(models: str | None) -> list[str]:
    if not models:
        return []
    parsed = [model.strip() for model in models.split(",") if model.strip()]
    if not parsed:
        raise ValueError("--models was supplied but no model names were parsed")
    return parsed


def continuation_model_for_row(
    row: dict,
    *,
    override_model: str | None,
    alternating_models: list[str] | None,
    batch_index: int,
) -> tuple[str, str, int | None]:
    """Return continuation model, schedule name, and schedule index for one row."""
    if override_model and alternating_models:
        raise ValueError("Use either --model or --models, not both")
    if alternating_models:
        schedule_index = batch_index % len(alternating_models)
        return alternating_models[schedule_index], "alternating_by_batch", schedule_index
    if override_model:
        return override_model, "override", None
    model_name = row.get("model_name")
    if not model_name:
        raise ValueError("Row is missing model_name and neither --model nor --models was supplied")
    return str(model_name), "source_model", None
