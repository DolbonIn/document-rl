"""Async sampler backends for Atropos rollout generation.

Supports:
- VLLMSampler: High-throughput async inference via vLLM (production)
- LocalSampler: HuggingFace Transformers fallback (development)
"""

from __future__ import annotations

import importlib
import logging
from dataclasses import dataclass, field
from typing import Any, Protocol

from .environment import Problem

logger = logging.getLogger(__name__)


@dataclass
class SamplerOutput:
    """Output from a single generation."""

    problem_id: str
    text: str
    token_ids: list[int] = field(default_factory=list)
    logprobs: list[float] = field(default_factory=list)
    finish_reason: str = "stop"


class AsyncSamplerBackend(Protocol):
    """Protocol for async sampler backends."""

    async def generate(
        self,
        problems: list[Problem],
        temperature: float,
        top_p: float,
        max_tokens: int,
    ) -> list[SamplerOutput]: ...

    async def close(self) -> None: ...


@dataclass
class VLLMSampler:
    """Async sampler using vLLM's OpenAI-compatible API.

    Connects to a running vLLM server (e.g., on GPU 0-1) for high-throughput
    inference during async GRPO rollout generation.
    """

    api_base: str = "http://localhost:8000/v1"
    model_name: str = "default"
    api_key: str = "EMPTY"
    _client: Any = field(default=None, repr=False)

    async def _ensure_client(self) -> None:
        if self._client is not None:
            return
        try:
            openai = importlib.import_module("openai")
            self._client = openai.AsyncOpenAI(
                base_url=self.api_base,
                api_key=self.api_key,
            )
        except ImportError as exc:
            raise RuntimeError(
                "VLLMSampler requires the openai package: pip install openai"
            ) from exc

    async def generate(
        self,
        problems: list[Problem],
        temperature: float = 0.8,
        top_p: float = 0.95,
        max_tokens: int = 768,
    ) -> list[SamplerOutput]:
        await self._ensure_client()
        import asyncio

        async def _generate_one(problem: Problem) -> SamplerOutput:
            messages = [
                {"role": "system", "content": problem.system_prompt},
                {"role": "user", "content": problem.user_prompt},
            ]
            try:
                response = await self._client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                    logprobs=True,
                    top_logprobs=1,
                )
                choice = response.choices[0]
                text = choice.message.content or ""

                # Extract logprobs if available
                token_logprobs: list[float] = []
                if choice.logprobs and choice.logprobs.content:
                    token_logprobs = [t.logprob for t in choice.logprobs.content]

                return SamplerOutput(
                    problem_id=problem.problem_id,
                    text=text,
                    logprobs=token_logprobs,
                    finish_reason=choice.finish_reason or "stop",
                )
            except Exception as exc:
                logger.warning("Generation failed for %s: %s", problem.problem_id, exc)
                return SamplerOutput(
                    problem_id=problem.problem_id,
                    text="",
                    finish_reason="error",
                )

        results = await asyncio.gather(*[_generate_one(p) for p in problems])
        return list(results)

    async def close(self) -> None:
        if self._client is not None:
            await self._client.close()
            self._client = None


@dataclass
class LocalSampler:
    """Local HuggingFace Transformers sampler for development/testing.

    Loads a model locally with optional LoRA adapter.
    """

    model_path: str = ""
    lora_path: str | None = None
    device: str = "cuda"
    _model: Any = field(default=None, repr=False)
    _tokenizer: Any = field(default=None, repr=False)

    async def _ensure_model(self) -> None:
        if self._model is not None:
            return
        if not self.model_path.strip():
            raise RuntimeError("LocalSampler requires a non-empty model_path")
        torch = importlib.import_module("torch")
        transformers = importlib.import_module("transformers")

        self._tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_path)
        self._model = transformers.AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
        )
        self._model = self._model.to(self.device)

        if self.lora_path:
            peft = importlib.import_module("peft")
            self._model = peft.PeftModel.from_pretrained(self._model, self.lora_path)
            logger.info("Loaded LoRA adapter from %s", self.lora_path)

        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        self._model.eval()

    async def generate(
        self,
        problems: list[Problem],
        temperature: float = 0.8,
        top_p: float = 0.95,
        max_tokens: int = 768,
    ) -> list[SamplerOutput]:
        await self._ensure_model()
        torch = importlib.import_module("torch")

        results: list[SamplerOutput] = []
        for problem in problems:
            prompt = f"{problem.system_prompt}\n\n{problem.user_prompt}"
            encoded = self._tokenizer(prompt, return_tensors="pt")
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
            prompt_len = encoded["input_ids"].shape[1]

            with torch.no_grad():
                outputs = self._model.generate(
                    **encoded,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    max_new_tokens=max_tokens,
                    return_dict_in_generate=True,
                    output_scores=True,
                    pad_token_id=self._tokenizer.pad_token_id,
                )

            tokens = outputs.sequences[0][prompt_len:].tolist()
            text = self._tokenizer.decode(tokens, skip_special_tokens=True)

            logprobs: list[float] = []
            for step, scores in enumerate(outputs.scores[: len(tokens)]):
                lp = torch.log_softmax(scores[0], dim=-1)
                logprobs.append(lp[tokens[step]].item())

            results.append(
                SamplerOutput(
                    problem_id=problem.problem_id,
                    text=text,
                    token_ids=tokens,
                    logprobs=logprobs,
                    finish_reason="stop",
                )
            )

        return results

    async def close(self) -> None:
        self._model = None
        self._tokenizer = None


def build_sampler(kind: str, **kwargs: Any) -> AsyncSamplerBackend:
    """Factory for sampler backends."""
    if kind == "vllm":
        return VLLMSampler(
            **{k: v for k, v in kwargs.items() if k in VLLMSampler.__dataclass_fields__}
        )
    return LocalSampler(
        **{k: v for k, v in kwargs.items() if k in LocalSampler.__dataclass_fields__}
    )
