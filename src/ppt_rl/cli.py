from __future__ import annotations

import argparse
import json
from pathlib import Path

from .pipeline import PipelineConfig, run_pipeline
from .trainer import TrainConfig, train_grpo


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ppt_rl")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--input", required=True)
    run_parser.add_argument("--output", required=True)
    run_parser.add_argument(
        "--sampler", choices=["stub", "transformers"], default="stub"
    )
    run_parser.add_argument("--model-path")
    run_parser.add_argument("--num-candidates", type=int, default=4)
    run_parser.add_argument("--seed", type=int, default=42)
    run_parser.add_argument(
        "--renderer", choices=["fallback", "playwright"], default="fallback"
    )
    run_parser.add_argument("--device", default="cpu")
    run_parser.add_argument("--max-new-tokens", type=int, default=768)

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--input", required=True)
    train_parser.add_argument("--output", required=True)
    train_parser.add_argument("--model-path")
    train_parser.add_argument("--device", default="cpu")
    train_parser.add_argument("--learning-rate", type=float, default=1e-6)
    train_parser.add_argument("--beta-kl", type=float, default=0.02)
    train_parser.add_argument("--epochs", type=int, default=1)
    train_parser.add_argument("--dry-run", action="store_true")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "run":
        result = run_pipeline(
            PipelineConfig(
                input_path=Path(args.input),
                output_dir=Path(args.output),
                sampler=args.sampler,
                model_path=args.model_path,
                num_candidates=args.num_candidates,
                seed=args.seed,
                renderer=args.renderer,
                device=args.device,
                max_new_tokens=args.max_new_tokens,
            )
        )
    else:
        result = train_grpo(
            TrainConfig(
                input_path=Path(args.input),
                output_path=Path(args.output),
                model_path=args.model_path,
                device=args.device,
                learning_rate=args.learning_rate,
                beta_kl=args.beta_kl,
                epochs=args.epochs,
                dry_run=args.dry_run,
            )
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
