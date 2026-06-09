import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

from rllm_robustness.constants import INTERVENTIONS, TIMESTEPS


REPO_ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = REPO_ROOT


def run_cli(args, tmp_path):
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(CODE_ROOT) if not existing_pythonpath else f"{CODE_ROOT}:{existing_pythonpath}"
    subprocess.run(args, cwd=REPO_ROOT, env=env, check=True)


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_trace_fixture(path):
    row = {
        "record_id": "fixture-trace-1",
        "id": "fixture-trace-1",
        "domain": "MATH",
        "question": "What is 2+2?",
        "raw_question": "What is 2+2?",
        "reference_answer": "4",
        "model_name": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        "answer_content": (
            "<think>I need to add the two numbers.\n\n"
            "2 plus 2 equals 4.\n\n"
            "The final answer should therefore be 4.</think>\n"
            "\\boxed{4}"
        ),
    }
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")


def test_cli_all_interventions_all_timesteps_grid(tmp_path):
    trace_fixture = tmp_path / "trace_fixture.jsonl"
    interventions = tmp_path / "interventions.jsonl"
    continued = tmp_path / "continued.jsonl"
    judged = tmp_path / "judged.jsonl"
    metrics_json = tmp_path / "metrics.json"
    metrics_csv = tmp_path / "metrics.csv"
    write_trace_fixture(trace_fixture)

    run_cli(
        [
            sys.executable,
            "scripts/create_interventions.py",
            "--backend",
            "mock",
            "--input",
            str(trace_fixture),
            "--output",
            str(interventions),
            "--allow-wikipedia-fallback",
            "--batch-size",
            "1",
            "--flush-every-batches",
            "1",
        ],
        tmp_path,
    )
    rows = read_jsonl(interventions)
    assert len(rows) == len(INTERVENTIONS) * len(TIMESTEPS)
    counts = Counter((row["intervention"], row["target_timestep"]) for row in rows)
    assert set(counts.values()) == {1}

    run_cli(
        [
            sys.executable,
            "scripts/continue_interventions.py",
            "--backend",
            "mock",
            "--input",
            str(interventions),
            "--output",
            str(continued),
            "--num-completions",
            "2",
            "--batch-size",
            "3",
            "--flush-every-batches",
            "1",
        ],
        tmp_path,
    )
    continued_rows = read_jsonl(continued)
    assert len(continued_rows) == len(rows)
    assert all(len(row["continuations"]) == 2 for row in continued_rows)
    assert all(len(row["complete_answers"]) == 2 for row in continued_rows)

    run_cli(
        [
            sys.executable,
            "scripts/judge_answers.py",
            "--backend",
            "mock",
            "--input",
            str(continued),
            "--output",
            str(judged),
            "--mode",
            "completions",
            "--batch-size",
            "4",
            "--flush-every-batches",
            "1",
        ],
        tmp_path,
    )
    judged_rows = read_jsonl(judged)
    assert len(judged_rows) == len(rows)
    assert all(len(row["verifier_scores"]) == 2 for row in judged_rows)

    run_cli(
        [
            sys.executable,
            "scripts/compute_metrics.py",
            "--input",
            str(judged),
            "--output-json",
            str(metrics_json),
            "--output-csv",
            str(metrics_csv),
        ],
        tmp_path,
    )
    payload = json.loads(metrics_json.read_text(encoding="utf-8"))
    assert len(payload["summary"]) == len(INTERVENTIONS) * len(TIMESTEPS)

    run_cli(
        [
            sys.executable,
            "scripts/validate_grid.py",
            "--input",
            str(judged),
            "--expected-items",
            "1",
            "--kind",
            "judged",
            "--num-completions",
            "2",
        ],
        tmp_path,
    )
