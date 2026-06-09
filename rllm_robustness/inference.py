from __future__ import annotations

import gc
import logging
from dataclasses import replace
from typing import Any

from .constants import DecodingConfig, DEFAULT_SYSTEM_PROMPT

logger = logging.getLogger(__name__)


class MockRunner:
    """Deterministic runner for unit tests and CLI dry-runs."""

    def __init__(self, model_name: str = "mock-model") -> None:
        self.current_model_name = model_name

    def load_model(self, model_name: str, **_: Any) -> None:
        self.current_model_name = model_name

    def generate(
        self,
        prompts: list[str],
        *,
        system_prompts: str | list[str] = DEFAULT_SYSTEM_PROMPT,
        reasoning_traces: str | list[str] = "",
        model_name: str | None = None,
        decoding: DecodingConfig | None = None,
        max_new_tokens_override: int | None = None,
    ) -> list[list[str]]:
        if model_name:
            self.current_model_name = model_name
        n = decoding.n if decoding else 1
        outputs: list[list[str]] = []
        for idx, prompt in enumerate(prompts):
            if "Is the Reference Answer present in the Answer Content?" in prompt:
                outputs.append([self._mock_answer_judgment(prompt)])
                continue
            if "Does this text indicate that the previous reasoning contains errors or irrelevant information?" in prompt:
                outputs.append(["No"])
                continue
            short = prompt.replace("\n", " ")[:80]
            outputs.append([f"Mock completion {j + 1} for item {idx}: {short}" for j in range(n)])
        return outputs

    @staticmethod
    def _mock_answer_judgment(prompt: str) -> str:
        reference_marker = "Reference Answer:\n"
        answer_marker = "\n\nAnswer Content (extract):\n"
        end_marker = "\n\nIs the Reference Answer present"
        if reference_marker not in prompt or answer_marker not in prompt:
            return "Incorrect"
        reference = prompt.split(reference_marker, 1)[1].split(answer_marker, 1)[0].strip()
        answer = prompt.split(answer_marker, 1)[1].split(end_marker, 1)[0].strip()
        return "Correct" if reference and reference in answer else "Incorrect"

    def unload_model(self) -> None:
        return None


class VllmRunner:
    def __init__(
        self,
        *,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        max_model_len: int | None = 32768,
        max_num_seqs: int | None = None,
        enforce_eager: bool = False,
        dtype: str = "auto",
    ) -> None:
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.requested_max_model_len = max_model_len
        self.max_num_seqs = max_num_seqs
        self.enforce_eager = enforce_eager
        self.dtype = dtype
        self.tokenizer = None
        self.vllm_model = None
        self.current_model_name: str | None = None
        self.current_max_model_len: int | None = None

    def load_model(
        self,
        model_name: str,
        *,
        tensor_parallel_size: int | None = None,
        gpu_memory_utilization: float | None = None,
        max_model_len: int | None = None,
    ) -> None:
        if self.vllm_model is not None and self.current_model_name == model_name:
            return
        if self.vllm_model is not None:
            self.unload_model()

        from vllm import LLM
        from transformers import AutoConfig, AutoTokenizer

        tensor_parallel_size = tensor_parallel_size or self.tensor_parallel_size
        gpu_memory_utilization = gpu_memory_utilization or self.gpu_memory_utilization
        requested_max_model_len = max_model_len or self.requested_max_model_len

        logger.info("Loading tokenizer and config for %s", model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        native_max_len = None
        for attr in ("max_position_embeddings", "n_positions", "seq_length", "max_seq_len"):
            if hasattr(cfg, attr):
                native_max_len = getattr(cfg, attr)
                break
        if native_max_len is None:
            native_max_len = getattr(self.tokenizer, "model_max_length", 32768)
        effective_max_len = min(native_max_len, requested_max_model_len or native_max_len)
        logger.info("Loading %s with max_model_len=%s", model_name, effective_max_len)
        llm_kwargs: dict[str, Any] = {}
        if self.max_num_seqs is not None:
            llm_kwargs["max_num_seqs"] = self.max_num_seqs
        self.vllm_model = LLM(
            model=model_name,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=effective_max_len,
            trust_remote_code=True,
            dtype=self.dtype,
            enforce_eager=self.enforce_eager,
            **llm_kwargs,
        )
        self.current_model_name = model_name
        self.current_max_model_len = effective_max_len

    def unload_model(self) -> None:
        if self.vllm_model is None:
            return
        try:
            from vllm.distributed.parallel_state import destroy_model_parallel

            destroy_model_parallel()
        except Exception:
            logger.debug("Could not destroy vLLM model parallel state", exc_info=True)
        del self.vllm_model
        self.vllm_model = None
        self.tokenizer = None
        self.current_model_name = None
        self.current_max_model_len = None
        gc.collect()
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            try:
                torch.distributed.destroy_process_group()
            except Exception:
                logger.debug("Could not destroy torch process group", exc_info=True)

    def _format_prompts(
        self,
        prompts: list[str],
        system_prompts: str | list[str],
        reasoning_traces: str | list[str],
    ) -> list[str]:
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer is not loaded")
        num_prompts = len(prompts)
        if isinstance(system_prompts, str):
            system_prompts = [system_prompts] * num_prompts
        if isinstance(reasoning_traces, str):
            reasoning_traces = [reasoning_traces] * num_prompts
        if len(system_prompts) != num_prompts or len(reasoning_traces) != num_prompts:
            raise ValueError("system_prompts and reasoning_traces must match prompts length")

        formatted_prompts: list[str] = []
        is_gpt_oss = isinstance(self.current_model_name, str) and self.current_model_name.startswith(
            "openai/gpt-oss"
        )
        for prompt, system_prompt, reasoning_trace in zip(
            prompts, system_prompts, reasoning_traces
        ):
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            if is_gpt_oss and reasoning_trace:
                prompt = f"{prompt}\n\n{reasoning_trace}"
            messages.append({"role": "user", "content": prompt})
            formatted = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            if reasoning_trace and not is_gpt_oss:
                formatted += reasoning_trace
            formatted_prompts.append(formatted)
        return formatted_prompts

    def _sampling_params(self, decoding: DecodingConfig):
        from vllm import SamplingParams

        kwargs = decoding.sampling_kwargs()
        if decoding.use_greedy:
            return SamplingParams(
                max_tokens=decoding.max_new_tokens,
                temperature=0.0,
                seed=decoding.seed,
                **kwargs,
            )
        return SamplingParams(
            max_tokens=decoding.max_new_tokens,
            temperature=decoding.temperature,
            seed=decoding.seed,
            **kwargs,
        )

    def generate(
        self,
        prompts: list[str],
        *,
        system_prompts: str | list[str] = DEFAULT_SYSTEM_PROMPT,
        reasoning_traces: str | list[str] = "",
        model_name: str | None = None,
        decoding: DecodingConfig | None = None,
        max_new_tokens_override: int | None = None,
    ) -> list[list[str]]:
        if not prompts:
            return []
        if model_name is None and self.current_model_name is None:
            raise ValueError("model_name is required when no model is loaded")
        if model_name is not None and model_name != self.current_model_name:
            self.load_model(model_name)
        if self.vllm_model is None or self.tokenizer is None:
            self.load_model(model_name or self.current_model_name)  # type: ignore[arg-type]
        if decoding is None:
            decoding = DecodingConfig(temperature=0.0, use_greedy=True)
        if max_new_tokens_override is not None:
            decoding = replace(decoding, max_new_tokens=max_new_tokens_override)

        formatted_prompts = self._format_prompts(prompts, system_prompts, reasoning_traces)
        sampling_params = self._sampling_params(decoding)
        outputs = self.vllm_model.generate(prompts=formatted_prompts, sampling_params=sampling_params)
        if len(outputs) != len(formatted_prompts):
            raise RuntimeError(
                f"vLLM returned {len(outputs)} outputs for {len(formatted_prompts)} prompts"
            )
        expected_n = decoding.n
        generated: list[list[str]] = []
        for output in outputs:
            completions = [completion.text.strip() for completion in output.outputs]
            if len(completions) < expected_n:
                completions.extend([""] * (expected_n - len(completions)))
            generated.append(completions[:expected_n])
        return generated


def make_runner(backend: str, **kwargs: Any) -> MockRunner | VllmRunner:
    if backend == "mock":
        return MockRunner()
    if backend == "vllm":
        return VllmRunner(**kwargs)
    raise ValueError(f"Unknown backend: {backend}")
