"""Atropos async GRPO integration for document-rl.

Provides:
- HtmlFormattingEnv: Atropos-compatible environment for HTML formatting tasks
- AsyncGRPOTrainer: Asynchronous GRPO trainer with LoRA adapter support
- VLLMSampler: vLLM-based async inference for rollout generation
"""

from .environment import HtmlFormattingEnv
from .trainer import AsyncGRPOTrainer
from .sampler import VLLMSampler

__all__ = ["HtmlFormattingEnv", "AsyncGRPOTrainer", "VLLMSampler"]
