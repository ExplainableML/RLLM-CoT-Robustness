#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = REPO_ROOT
SCRIPT_ROOT = Path("scripts")


def csv(values: list[float | str]) -> str:
    return ",".join(str(value) for value in values)


def add_if_present(command: list[str], flag: str, value: str | int | float | None) -> None:
    if value is not None:
        command.extend([flag, str(value)])


def runner_args(args: argparse.Namespace) -> list[str]:
    command = [
        "--backend",
        args.backend,
        "--tensor-parallel-size",
        str(args.tensor_parallel_size),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--max-model-len",
        str(args.max_model_len),
        "--dtype",
        args.dtype,
    ]
    add_if_present(command, "--max-num-seqs", args.max_num_seqs)
    if args.enforce_eager:
        command.append("--enforce-eager")
    return command


def run(command: list[str], *, dry_run: bool, env: dict[str, str]) -> None:
    printable = " ".join(command)
    if dry_run:
        print(printable)
        return
    subprocess.run(command, cwd=REPO_ROOT, env=env, check=True)


def script(name: str) -> str:
    return str(SCRIPT_ROOT / name)


def common_io(
    *,
    input_path: str,
    output_path: Path,
    split: str,
    limit: int | None,
    batch_size: int,
    flush_every_batches: int,
    resume: bool,
) -> list[str]:
    command = [
        "--input",
        input_path,
        "--output",
        str(output_path),
        "--split",
        split,
        "--batch-size",
        str(batch_size),
        "--flush-every-batches",
        str(flush_every_batches),
    ]
    add_if_present(command, "--limit", limit)
    if resume:
        command.append("--resume")
    return command


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the CoT robustness pipeline for one trace pool. The input should "
            "contain original answer traces with question, reference_answer, "
            "answer_content, model_name, and domain fields."
        )
    )
    parser.add_argument("--input", required=True, help="HF dataset name or local trace file.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--interventions", default=None)
    parser.add_argument("--timesteps", default=None)
    parser.add_argument("--intervention-model", default="Qwen/Qwen2.5-32B-Instruct")
    parser.add_argument("--judge-model", default="Qwen/Qwen2.5-32B-Instruct")
    parser.add_argument("--model", default=None, help="Optional continuation-model override.")
    parser.add_argument("--models", default=None, help="Optional comma-separated continuation-model schedule.")
    parser.add_argument("--num-completions", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--continuation-batch-size", type=int, default=4)
    parser.add_argument("--judge-batch-size", type=int, default=4)
    parser.add_argument("--flush-every-batches", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--include-wait-ablation", action="store_true")
    parser.add_argument("--wait-marker", default="Wait")
    parser.add_argument("--run-doubt", action="store_true")
    parser.add_argument("--run-length", action="store_true")
    parser.add_argument("--allow-wikipedia-fallback", action="store_true")
    parser.add_argument("--backend", choices=["vllm", "mock"], default="vllm")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--max-num-seqs", type=int, default=None)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--intervention-max-new-tokens", type=int, default=None)
    parser.add_argument("--continuation-max-new-tokens", type=int, default=32768)
    parser.add_argument("--judge-max-new-tokens", type=int, default=10)
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used for child script commands.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.model and args.models:
        raise ValueError("Use either --model or --models, not both")

    output_dir = Path(args.output_dir)
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(CODE_ROOT) if not existing_pythonpath else f"{CODE_ROOT}:{existing_pythonpath}"

    interventions = args.interventions or (
        "benign_complete_step,benign_rewrite_trace,neutral_add_random_text,"
        "neutral_insert_random_characters,adversarial_continue_with_wrong_reasoning,"
        "adversarial_insert_wrong_fact,adversarial_continue_unrelated"
    )
    timesteps = args.timesteps or "0.1,0.3,0.5,0.7,0.9"

    intervention_files = [("baseline", output_dir / "interventions.jsonl")]
    create_cmd = [
        args.python,
        script("create_interventions.py"),
        *common_io(
            input_path=args.input,
            output_path=intervention_files[0][1],
            split=args.split,
            limit=args.limit,
            batch_size=args.batch_size,
            flush_every_batches=args.flush_every_batches,
            resume=args.resume,
        ),
        *runner_args(args),
        "--interventions",
        interventions,
        "--timesteps",
        timesteps,
        "--intervention-model",
        args.intervention_model,
    ]
    add_if_present(create_cmd, "--max-new-tokens", args.intervention_max_new_tokens)
    if args.allow_wikipedia_fallback:
        create_cmd.append("--allow-wikipedia-fallback")
    run(create_cmd, dry_run=args.dry_run, env=env)

    if args.include_wait_ablation:
        wait_file = output_dir / "interventions_wait.jsonl"
        intervention_files.append((args.wait_marker, wait_file))
        run(
            [
                args.python,
                script("append_marker_to_interventions.py"),
                "--input",
                str(intervention_files[0][1]),
                "--output",
                str(wait_file),
                "--split",
                args.split,
                "--marker",
                args.wait_marker,
                "--batch-size",
                str(args.batch_size),
                "--flush-every-batches",
                str(args.flush_every_batches),
                *(["--resume"] if args.resume else []),
            ],
            dry_run=args.dry_run,
            env=env,
        )

    for label, interventions_file in intervention_files:
        suffix = "" if label == "baseline" else f"_{label.lower()}"
        continued = output_dir / f"continued{suffix}.jsonl"
        judged = output_dir / f"judged{suffix}.jsonl"
        metrics_json = output_dir / f"metrics{suffix}.json"
        metrics_csv = output_dir / f"metrics{suffix}.csv"

        continue_cmd = [
            args.python,
            script("continue_interventions.py"),
            *common_io(
                input_path=str(interventions_file),
                output_path=continued,
                split=args.split,
                limit=None,
                batch_size=args.continuation_batch_size,
                flush_every_batches=args.flush_every_batches,
                resume=args.resume,
            ),
            *runner_args(args),
            "--num-completions",
            str(args.num_completions),
            "--max-new-tokens",
            str(args.continuation_max_new_tokens),
        ]
        if args.model:
            continue_cmd.extend(["--model", args.model])
        if args.models:
            continue_cmd.extend(["--models", args.models])
        run(continue_cmd, dry_run=args.dry_run, env=env)

        run(
            [
                args.python,
                script("judge_answers.py"),
                *common_io(
                    input_path=str(continued),
                    output_path=judged,
                    split=args.split,
                    limit=None,
                    batch_size=args.judge_batch_size,
                    flush_every_batches=args.flush_every_batches,
                    resume=args.resume,
                ),
                *runner_args(args),
                "--mode",
                "completions",
                "--judge-model",
                args.judge_model,
                "--max-new-tokens",
                str(args.judge_max_new_tokens),
            ],
            dry_run=args.dry_run,
            env=env,
        )

        run(
            [
                args.python,
                script("compute_metrics.py"),
                "--input",
                str(judged),
                "--output-json",
                str(metrics_json),
                "--output-csv",
                str(metrics_csv),
                "--split",
                args.split,
            ],
            dry_run=args.dry_run,
            env=env,
        )

        if args.run_doubt:
            run(
                [
                    args.python,
                    script("analyze_doubt.py"),
                    *common_io(
                        input_path=str(judged),
                        output_path=output_dir / f"doubt{suffix}.jsonl",
                        split=args.split,
                        limit=None,
                        batch_size=args.judge_batch_size,
                        flush_every_batches=args.flush_every_batches,
                        resume=args.resume,
                    ),
                    *runner_args(args),
                    "--judge-model",
                    args.judge_model,
                ],
                dry_run=args.dry_run,
                env=env,
            )

        if args.run_length:
            run(
                [
                    args.python,
                    script("length_overhead.py"),
                    "--input",
                    str(judged),
                    "--output-csv",
                    str(output_dir / f"length{suffix}.csv"),
                    "--split",
                    args.split,
                    "--remove-intervention-text",
                ],
                dry_run=args.dry_run,
                env=env,
            )


if __name__ == "__main__":
    main()
