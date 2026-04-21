from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path

from .io_utils import read_jsonl, write_json


@dataclass
class TrainConfig:
    input_path: Path
    output_path: Path
    model_path: str | None = None
    device: str = "cpu"
    learning_rate: float = 1e-6
    beta_kl: float = 0.02
    epochs: int = 1
    dry_run: bool = False


def _load_records(path: Path) -> list[dict]:
    return read_jsonl(path)


def dry_run_train(config: TrainConfig) -> dict:
    records = _load_records(config.input_path)
    mean_reward = sum(record["reward"] for record in records) / max(1, len(records))
    mean_advantage = sum(record["advantage"] for record in records) / max(
        1, len(records)
    )
    summary = {
        "mode": "dry_run",
        "records": len(records),
        "mean_reward": round(mean_reward, 6),
        "mean_advantage": round(mean_advantage, 6),
        "device": config.device,
        "model_path": config.model_path,
    }
    write_json(config.output_path, summary)
    return summary


def train_grpo(config: TrainConfig) -> dict:
    if config.dry_run:
        return dry_run_train(config)
    try:
        torch = importlib.import_module("torch")
        transformers = importlib.import_module("transformers")
        AutoModelForCausalLM = getattr(transformers, "AutoModelForCausalLM")
        AutoTokenizer = getattr(transformers, "AutoTokenizer")
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "Actual training requires torch and transformers; use --dry-run otherwise"
        ) from exc
    if not config.model_path:
        raise RuntimeError("--model-path is required for actual training")

    records = _load_records(config.input_path)
    tokenizer = AutoTokenizer.from_pretrained(config.model_path)
    model = AutoModelForCausalLM.from_pretrained(config.model_path)
    model.to(config.device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    losses: list[float] = []
    for _epoch in range(config.epochs):
        for record in records:
            prompt = record["prompt"]
            completion = record["completion"]
            encoded = tokenizer(prompt + completion, return_tensors="pt")
            encoded = {key: value.to(config.device) for key, value in encoded.items()}
            outputs = model(**encoded)
            logits = outputs.logits[:, :-1, :]
            labels = encoded["input_ids"][:, 1:]
            log_probs = (
                torch.log_softmax(logits, dim=-1)
                .gather(-1, labels.unsqueeze(-1))
                .squeeze(-1)
            )
            seq_logprob = log_probs.mean()
            advantage = torch.tensor(
                record["advantage"], device=config.device, dtype=seq_logprob.dtype
            )
            old_logprobs = record.get("old_logprobs") or [0.0]
            ref_logprobs = record.get("ref_logprobs") or old_logprobs
            old_mean = torch.tensor(
                sum(old_logprobs) / max(1, len(old_logprobs)),
                device=config.device,
                dtype=seq_logprob.dtype,
            )
            ref_mean = torch.tensor(
                sum(ref_logprobs) / max(1, len(ref_logprobs)),
                device=config.device,
                dtype=seq_logprob.dtype,
            )
            ratio = torch.exp(seq_logprob - old_mean)
            clipped_ratio = torch.clamp(ratio, 0.8, 1.2)
            policy_loss = -torch.min(ratio * advantage, clipped_ratio * advantage)
            kl = torch.abs(seq_logprob - ref_mean)
            loss = policy_loss + config.beta_kl * kl
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

    summary = {
        "mode": "train",
        "records": len(records),
        "epochs": config.epochs,
        "mean_loss": round(sum(losses) / max(1, len(losses)), 6),
        "device": config.device,
        "model_path": config.model_path,
    }
    write_json(config.output_path, summary)
    return summary
