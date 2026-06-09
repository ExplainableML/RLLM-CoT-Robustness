from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Iterable, Iterator


def stable_id(*parts: object) -> str:
    payload = "\x1f".join("" if part is None else str(part) for part in parts)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def read_jsonl(path: str | Path, *, limit: int | None = None) -> list[dict]:
    records: list[dict] = []
    with Path(path).open("r", encoding="utf-8") as fp:
        for line in fp:
            if not line.strip():
                continue
            records.append(json.loads(line))
            if limit is not None and len(records) >= limit:
                break
    return records


def iter_jsonl(path: str | Path) -> Iterator[dict]:
    with Path(path).open("r", encoding="utf-8") as fp:
        for line in fp:
            if line.strip():
                yield json.loads(line)


def append_jsonl(path: str | Path, records: Iterable[dict]) -> int:
    output_path = Path(path)
    ensure_parent(output_path)
    count = 0
    with output_path.open("a", encoding="utf-8") as fp:
        for record in records:
            fp.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def write_json(path: str | Path, payload: object) -> None:
    output_path = Path(path)
    ensure_parent(output_path)
    with output_path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2, ensure_ascii=False)


def write_csv(path: str | Path, rows: list[dict]) -> None:
    output_path = Path(path)
    ensure_parent(output_path)
    if not rows:
        output_path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with output_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def existing_ids(path: str | Path, key: str = "record_id") -> set[str]:
    output_path = Path(path)
    if not output_path.exists():
        return set()
    ids: set[str] = set()
    for row in iter_jsonl(output_path):
        value = row.get(key)
        if value is not None:
            ids.add(str(value))
    return ids


def load_records(
    source: str,
    *,
    split: str = "train",
    limit: int | None = None,
    trust_remote_code: bool = False,
) -> list[dict]:
    """Load records from JSONL/JSON/CSV/Parquet or a Hugging Face dataset name."""
    source_path = Path(source)
    local_suffixes = {".jsonl", ".json", ".csv", ".parquet", ".pq"}
    if source_path.suffix.lower() in local_suffixes and not source_path.exists():
        raise FileNotFoundError(f"Local input file does not exist: {source_path}")
    if source_path.exists():
        suffix = source_path.suffix.lower()
        if suffix == ".jsonl":
            return read_jsonl(source_path, limit=limit)
        if suffix == ".json":
            payload = json.loads(source_path.read_text(encoding="utf-8"))
            records = payload if isinstance(payload, list) else payload.get(split, [])
            return list(records[:limit] if limit is not None else records)
        if suffix == ".csv":
            import pandas as pd

            df = pd.read_csv(source_path)
            if limit is not None:
                df = df.head(limit)
            return df.to_dict(orient="records")
        if suffix in {".parquet", ".pq"}:
            import pandas as pd

            df = pd.read_parquet(source_path)
            if limit is not None:
                df = df.head(limit)
            return df.to_dict(orient="records")
        raise ValueError(f"Unsupported local source suffix: {source_path.suffix}")

    from datasets import load_dataset

    dataset = load_dataset(source, split=split, trust_remote_code=trust_remote_code)
    if limit is not None:
        dataset = dataset.select(range(min(limit, len(dataset))))
    return [dict(row) for row in dataset]


def batched(records: list[dict], batch_size: int) -> Iterator[list[dict]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    for i in range(0, len(records), batch_size):
        yield records[i : i + batch_size]


class JsonlBatchWriter:
    """Buffer rows and append to JSONL every k batches."""

    def __init__(self, path: str | Path, *, flush_every_batches: int = 1) -> None:
        self.path = Path(path)
        self.flush_every_batches = max(1, flush_every_batches)
        self._buffer: list[dict] = []
        self._batches_seen = 0
        ensure_parent(self.path)
        self.path.touch(exist_ok=True)

    def add_batch(self, rows: list[dict]) -> int:
        self._batches_seen += 1
        self._buffer.extend(rows)
        if self._batches_seen % self.flush_every_batches == 0:
            return self.flush()
        return 0

    def flush(self) -> int:
        if not self._buffer:
            return 0
        count = append_jsonl(self.path, self._buffer)
        self._buffer = []
        return count

    def close(self) -> int:
        return self.flush()
