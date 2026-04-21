from __future__ import annotations

import importlib
import math
import random
import re
from dataclasses import dataclass
from typing import Protocol

from .schemas import HtmlCandidate, SamplingParams, TaskSpec


def _byte_tokens(text: str) -> list[int]:
    return list(text.encode("utf-8"))


def _placeholder_logprobs(length: int) -> list[float]:
    return [-math.log(256.0)] * length


def _slug(prompt: str) -> str:
    words = re.findall(r"[A-Za-z0-9]+", prompt.lower())
    return " ".join(words[:8])


def _build_stub_html(task: TaskSpec, index: int) -> str:
    title = task.prompt.split(".")[0][:80]
    theme = task.constraints.theme
    accent = ["#0f62fe", "#7c3aed", "#dc2626", "#059669"][index % 4]
    density = [1.0, 1.15, 1.35, 1.6][index % 4]
    extra = "<li>Placeholder text</li>" if index == 3 else ""
    long_tail = " " + ("Growth. " * (120 if index == 2 else 12))
    body_size = max(12, int(24 - (index * 2)))
    return f"""<!DOCTYPE html>
<html>
<head>
  <meta charset=\"UTF-8\" />
  <title>{title}</title>
  <style>
    html, body {{ margin:0; padding:0; width:{task.constraints.viewport.width}px; height:{task.constraints.viewport.height}px; background:#f8fafc; color:#0f172a; font-family:Arial, sans-serif; }}
    body {{ display:flex; align-items:stretch; justify-content:center; }}
    main.slide {{ box-sizing:border-box; width:100%; height:100%; padding:{int(72 / density)}px; display:flex; flex-direction:column; gap:{int(22 / density)}px; background:linear-gradient(135deg, #ffffff 0%, #eef2ff 100%); }}
    .eyebrow {{ color:{accent}; font-weight:700; letter-spacing:0.08em; text-transform:uppercase; font-size:18px; }}
    h1 {{ margin:0; font-size:{max(28, int(54 / density))}px; line-height:1.05; }}
    .grid {{ display:grid; grid-template-columns:1.2fr 0.8fr; gap:24px; flex:1; }}
    .card {{ background:white; border:1px solid #dbe4ff; border-radius:24px; padding:28px; box-shadow:0 10px 25px rgba(15,23,42,0.08); overflow:hidden; }}
    p, li {{ font-size:{body_size}px; line-height:1.45; }}
    ul {{ margin:0; padding-left:22px; }}
    .metric {{ font-size:{max(22, int(46 / density))}px; color:{accent}; font-weight:800; }}
  </style>
</head>
<body>
  <main class=\"slide\">
    <div class=\"eyebrow\">{theme}</div>
    <h1>{title}</h1>
    <div class=\"grid\">
      <section class=\"card\">
        <p>Problem</p>
        <ul>
          <li>Teams need a reliable way to turn prompts into HTML slides.</li>
          <li>They need stable rendering, quality scoring, and training data export.</li>
          {extra}
        </ul>
      </section>
      <section class=\"card\">
        <p>Solution</p>
        <div class=\"metric\">+32%</div>
        <p>Traction and renderer-first feedback loop.{long_tail}</p>
      </section>
    </div>
    <section class=\"card\"><p>Traction</p><p>{_slug(task.prompt)}</p></section>
  </main>
</body>
</html>"""


class SamplerBackend(Protocol):
    def sample(
        self,
        task: TaskSpec,
        group_id: str,
        num_candidates: int,
        sampling_params: SamplingParams,
    ) -> list[HtmlCandidate]: ...


@dataclass
class StubSampler:
    model_version: str = "stub_sampler_v1"

    def sample(
        self,
        task: TaskSpec,
        group_id: str,
        num_candidates: int,
        sampling_params: SamplingParams,
    ) -> list[HtmlCandidate]:
        random.seed(sampling_params.seed)
        candidates: list[HtmlCandidate] = []
        for index in range(num_candidates):
            html = _build_stub_html(task, index)
            tokens = _byte_tokens(html)
            candidates.append(
                HtmlCandidate(
                    candidate_id=f"{group_id}_cand_{index + 1}",
                    task_id=task.task_id,
                    group_id=group_id,
                    model_version=self.model_version,
                    output_format="constrained_html",
                    html=html,
                    sampling_params=sampling_params,
                    prompt=task.prompt,
                    completion_tokens=tokens,
                    completion_mask=[1] * len(tokens),
                    old_logprobs=_placeholder_logprobs(len(tokens)),
                    ref_logprobs=_placeholder_logprobs(len(tokens)),
                )
            )
        return candidates


@dataclass
class TransformersSampler:
    model_path: str
    device: str = "cpu"
    model_version: str = "transformers_local"

    def sample(
        self,
        task: TaskSpec,
        group_id: str,
        num_candidates: int,
        sampling_params: SamplingParams,
    ) -> list[HtmlCandidate]:
        try:
            torch = importlib.import_module("torch")
            transformers = importlib.import_module("transformers")
            AutoModelForCausalLM = getattr(transformers, "AutoModelForCausalLM")
            AutoTokenizer = getattr(transformers, "AutoTokenizer")
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "transformers sampler requires torch and transformers"
            ) from exc

        tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        model = AutoModelForCausalLM.from_pretrained(self.model_path)
        model.to(self.device)
        model.eval()
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        prompt = (
            "Generate a single self-contained constrained HTML/CSS page for the following task. "
            "Use inline CSS only, no external assets, no JavaScript, and include the required sections.\n\n"
            f"Task: {task.prompt}\n"
            f"Must include: {', '.join(task.constraints.must_include)}\n"
            f"Must not include: {', '.join(task.constraints.must_not_include)}\n"
            "Return only HTML."
        )
        encoded = tokenizer(prompt, return_tensors="pt")
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        generator = torch.Generator(device=self.device)
        generator.manual_seed(sampling_params.seed)
        outputs = model.generate(
            **encoded,
            do_sample=True,
            temperature=sampling_params.temperature,
            top_p=sampling_params.top_p,
            max_new_tokens=sampling_params.max_new_tokens,
            num_return_sequences=num_candidates,
            return_dict_in_generate=True,
            output_scores=True,
            pad_token_id=tokenizer.pad_token_id,
            generator=generator,
        )
        prompt_len = encoded["input_ids"].shape[1]
        sequences = outputs.sequences
        candidates: list[HtmlCandidate] = []
        stacked_scores = outputs.scores
        for idx in range(num_candidates):
            sequence = sequences[idx]
            completion_tokens = sequence[prompt_len:].tolist()
            html = tokenizer.decode(completion_tokens, skip_special_tokens=True)
            old_logprobs: list[float] = []
            for step, scores in enumerate(stacked_scores[: len(completion_tokens)]):
                token_id = completion_tokens[step]
                score_row = scores[idx]
                old_logprobs.append(
                    torch.log_softmax(score_row, dim=-1)[token_id].item()
                )
            candidates.append(
                HtmlCandidate(
                    candidate_id=f"{group_id}_cand_{idx + 1}",
                    task_id=task.task_id,
                    group_id=group_id,
                    model_version=f"{self.model_version}:{self.model_path}",
                    output_format="constrained_html",
                    html=html,
                    sampling_params=sampling_params,
                    prompt=prompt,
                    completion_tokens=completion_tokens,
                    completion_mask=[1] * len(completion_tokens),
                    old_logprobs=old_logprobs,
                    ref_logprobs=list(old_logprobs),
                )
            )
        return candidates


def build_sampler(kind: str, model_path: str | None, device: str) -> SamplerBackend:
    if kind == "transformers":
        if not model_path:
            raise RuntimeError("--model-path is required for transformers sampler")
        return TransformersSampler(model_path=model_path, device=device)
    return StubSampler()
