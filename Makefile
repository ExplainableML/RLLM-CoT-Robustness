PYTHON ?= python
CODE_ROOT := .
PYTEST_DISABLE_PLUGIN_AUTOLOAD ?= 1
RESULTS_DIR ?= results/pipeline_make
MODEL ?= deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
ALT_MODELS ?= deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B,deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
PIPELINE_INPUT ?= reasoning-proj/filtered_math_traces_original_DeepSeek-R1-Distill-Qwen-1.5B
PIPELINE_OUT ?= results/pipeline_run
EXPECTED_ITEMS ?= 1
FIXTURE_TRACE := $(RESULTS_DIR)/fixtures/tiny_trace.jsonl

.PHONY: help test smoke-mock smoke-gpu run-pipeline-dry validate-grid clean-results fixture-trace

help:
	@echo "Targets:"
	@echo "  make test              Run unit tests"
	@echo "  make smoke-mock        Run local mock intervention -> continuation -> judge -> metrics"
	@echo "  make smoke-gpu         Run a tiny vLLM continuation smoke on MODEL"
	@echo "  make run-pipeline-dry  Show commands for the full robustness pipeline"
	@echo "  make validate-grid     Validate a completed judged grid"
	@echo "Variables:"
	@echo "  MODEL=$(MODEL)"
	@echo "  ALT_MODELS=$(ALT_MODELS)"
	@echo "  RESULTS_DIR=$(RESULTS_DIR)"
	@echo "  PIPELINE_INPUT=$(PIPELINE_INPUT)"
	@echo "  PIPELINE_OUT=$(PIPELINE_OUT)"

test:
	PYTHONPATH=$(CODE_ROOT) PYTEST_DISABLE_PLUGIN_AUTOLOAD=$(PYTEST_DISABLE_PLUGIN_AUTOLOAD) $(PYTHON) -m pytest -q tests

fixture-trace:
	@mkdir -p $(dir $(FIXTURE_TRACE))
	@$(PYTHON) -c 'import json; from pathlib import Path; path = Path("$(FIXTURE_TRACE)"); row = {"record_id": "fixture-trace-1", "id": "fixture-trace-1", "domain": "MATH", "question": "What is 2+2?", "raw_question": "What is 2+2?", "reference_answer": "4", "model_name": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", "answer_content": "<think>I need to add the two numbers.\n\n2 plus 2 equals 4.\n\nThe final answer should therefore be 4.</think>\n\\boxed{4}"}; path.write_text(json.dumps(row) + "\n", encoding="utf-8")'

verify-mock: fixture-trace
	rm -rf $(RESULTS_DIR)/mock
	PYTHONPATH=$(CODE_ROOT) $(PYTHON) scripts/create_interventions.py \
		--backend mock \
		--input $(FIXTURE_TRACE) \
		--output $(RESULTS_DIR)/mock/interventions.jsonl \
		--interventions neutral_insert_random_characters \
		--timesteps 0.3 \
		--limit 1 --batch-size 1 --flush-every-batches 1
	PYTHONPATH=$(CODE_ROOT) $(PYTHON) scripts/continue_interventions.py \
		--backend mock \
		--input $(RESULTS_DIR)/mock/interventions.jsonl \
		--output $(RESULTS_DIR)/mock/continued.jsonl \
		--models $(ALT_MODELS) \
		--num-completions 2 \
		--limit 1 --batch-size 1 --flush-every-batches 1
	PYTHONPATH=$(CODE_ROOT) $(PYTHON) scripts/judge_answers.py \
		--backend mock \
		--input $(RESULTS_DIR)/mock/continued.jsonl \
		--output $(RESULTS_DIR)/mock/judged.jsonl \
		--mode completions \
		--judge-model Qwen/Qwen2.5-32B-Instruct \
		--limit 1 --batch-size 1 --on-error score_zero
	PYTHONPATH=$(CODE_ROOT) $(PYTHON) scripts/compute_metrics.py \
		--input $(RESULTS_DIR)/mock/judged.jsonl \
		--output-json $(RESULTS_DIR)/mock/metrics.json \
		--output-csv $(RESULTS_DIR)/mock/metrics.csv

verify-gpu: fixture-trace
	rm -rf $(RESULTS_DIR)/gpu
	PYTHONPATH=$(CODE_ROOT) $(PYTHON) scripts/create_interventions.py \
		--backend mock \
		--input $(FIXTURE_TRACE) \
		--output $(RESULTS_DIR)/gpu/interventions.jsonl \
		--interventions neutral_insert_random_characters \
		--timesteps 0.3 \
		--limit 1 --batch-size 1
	PYTHONPATH=$(CODE_ROOT) $(PYTHON) scripts/continue_interventions.py \
		--input $(RESULTS_DIR)/gpu/interventions.jsonl \
		--output $(RESULTS_DIR)/gpu/continued.jsonl \
		--model $(MODEL) \
		--num-completions 2 \
		--limit 1 --batch-size 1 --max-new-tokens 64 \
		--max-model-len 2048 --gpu-memory-utilization 0.60

run-pipeline-dry:
	PYTHONPATH=$(CODE_ROOT) $(PYTHON) scripts/run_pipeline.py \
		--input $(PIPELINE_INPUT) \
		--output-dir $(PIPELINE_OUT) \
		--backend mock \
		--allow-wikipedia-fallback \
		--include-wait-ablation \
		--run-length \
		--limit $(EXPECTED_ITEMS) \
		--python $(PYTHON) \
		--dry-run

validate-grid:
	PYTHONPATH=$(CODE_ROOT) $(PYTHON) scripts/validate_grid.py \
		--input $(PIPELINE_OUT)/judged.jsonl \
		--expected-items $(EXPECTED_ITEMS) \
		--kind judged

clean-results:
	rm -rf $(RESULTS_DIR)
