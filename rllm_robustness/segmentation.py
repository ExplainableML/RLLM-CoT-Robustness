from __future__ import annotations


def extract_reasoning(trace: str | None) -> str:
    """Extract the reasoning part of a model answer using the original tag logic."""
    if not trace:
        return ""
    text = str(trace)
    if "</think>" in text:
        text = text.split("</think>")[0]
    elif "</thought>" in text:
        text = text.split("</thought>")[0]

    if "<think>" in text:
        text = text.split("<think>")[-1]
    elif "<thought>" in text:
        text = text.split("<thought>")[-1]
    elif "<think" in text:
        text = text.split("<think")[-1]
    elif "<thought" in text:
        text = text.split("<thought")[-1]
    return text


def segment_trace(trace: str | list[str]) -> tuple[list[str], list[dict[str, int | str]]]:
    """Split a CoT into double-newline segments, preserving formatting metadata."""
    paragraphs = trace if isinstance(trace, list) else str(trace).split("\n\n")
    segments: list[str] = []
    metadata: list[dict[str, int | str]] = []
    for i, paragraph in enumerate(paragraphs):
        leading_spaces = len(paragraph) - len(paragraph.lstrip()) if paragraph else 0
        trailing_spaces = len(paragraph) - len(paragraph.rstrip()) if paragraph else 0
        segments.append(paragraph)
        metadata.append(
            {
                "leading_spaces": leading_spaces,
                "trailing_spaces": trailing_spaces,
                "divider": "\n\n" if i < len(paragraphs) - 1 else "",
            }
        )
    return segments, metadata


def restore_trace_from_segments(
    segments: list[str],
    metadata: list[dict[str, int | str]],
    *,
    skip_metadata: bool = False,
) -> str:
    if skip_metadata:
        return "\n\n".join(segment.strip() for segment in segments)
    if not segments:
        return ""

    result: list[str] = []
    process_len = min(len(segments), len(metadata))
    for i in range(process_len):
        segment = segments[i]
        meta = metadata[i]
        formatted = (
            " " * int(meta["leading_spaces"])
            + segment.strip()
            + " " * int(meta["trailing_spaces"])
        )
        result.append(formatted)
        if i < len(metadata) - 1 and meta.get("divider") == "\n\n":
            result.append("")

    if len(segments) > len(metadata):
        if result and metadata and metadata[-1].get("divider") == "\n\n" and result[-1] != "":
            result.append("")
        for i in range(len(metadata), len(segments)):
            result.append(segments[i].strip())
            if i < len(segments) - 1:
                result.append("")

    return "\n".join(result)


def normalized_trace_timesteps(segments: list[str]) -> list[float]:
    total_length = sum(len(segment) for segment in segments)
    if total_length == 0:
        return [0.0] * len(segments)
    timesteps: list[float] = []
    running = 0
    for segment in segments:
        running += len(segment)
        timesteps.append(running / total_length)
    return timesteps


def trace_prefix_at_timestep(
    segments: list[str],
    timestep: float,
    *,
    min_steps: int = 2,
) -> tuple[list[str], int, float]:
    """Return prefix through the segment nearest to a normalized timestep."""
    timesteps = normalized_trace_timesteps(segments)
    if not timesteps:
        return [], -1, 0.0
    closest_index = min(range(len(timesteps)), key=lambda i: abs(timesteps[i] - timestep))
    final_count = min(max(closest_index + 1, min_steps), len(segments))
    selected_index = final_count - 1
    return segments[:final_count], selected_index, timesteps[selected_index]


def first_n_sentences(text: str, n: int) -> list[str]:
    """Small dependency-free fallback sentence splitter for tests and smoke runs."""
    import re

    candidates = re.split(r"(?<=[.!?])\s+", text.strip())
    return [candidate.strip() for candidate in candidates if candidate.strip()][:n]

