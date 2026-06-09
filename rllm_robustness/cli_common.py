from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .inference import make_runner


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )


def add_common_io_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input", required=True, help="Input dataset or local JSONL/JSON/CSV/Parquet.")
    parser.add_argument("--output", required=True, help="Output JSONL/CSV/JSON path.")
    parser.add_argument("--split", default="train", help="HF dataset split when --input is a dataset name.")
    parser.add_argument("--limit", type=int, default=None, help="Optional row limit for smoke runs.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--flush-every-batches", type=int, default=1)
    parser.add_argument("--resume", action="store_true", help="Skip record_ids already present in output.")


def add_runner_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--backend", choices=["vllm", "mock"], default="vllm")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--max-num-seqs", type=int, default=None)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--dtype", default="auto")


def build_runner(args: argparse.Namespace):
    return make_runner(
        args.backend,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        enforce_eager=args.enforce_eager,
        dtype=args.dtype,
    )


def project_path(path: str) -> Path:
    return Path(path).expanduser().resolve()
