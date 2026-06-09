import pytest

from rllm_robustness.io import JsonlBatchWriter, load_records, read_jsonl


def test_jsonl_batch_writer_touches_empty_output(tmp_path):
    output = tmp_path / "empty.jsonl"
    writer = JsonlBatchWriter(output, flush_every_batches=2)
    assert output.exists()
    assert output.read_text(encoding="utf-8") == ""
    assert writer.close() == 0


def test_jsonl_batch_writer_flushes_every_k_batches(tmp_path):
    output = tmp_path / "rows.jsonl"
    writer = JsonlBatchWriter(output, flush_every_batches=2)
    assert writer.add_batch([{"id": 1}]) == 0
    assert output.read_text(encoding="utf-8") == ""
    assert writer.add_batch([{"id": 2}]) == 2
    assert read_jsonl(output) == [{"id": 1}, {"id": 2}]
    assert writer.close() == 0


def test_missing_local_jsonl_fails_instead_of_loading_hf_dataset(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_records(str(tmp_path / "missing.jsonl"))
