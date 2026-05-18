"""Run a cloud or edge VLM evaluation. Fleshed out in task 06."""

from __future__ import annotations

import argparse
import sys


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a VLM zero-shot anomaly evaluation.")
    parser.add_argument(
        "--backend",
        required=True,
        choices=[
            "mock",
            "together",
            "gemini",
            "anthropic",
            "groq",
            "edge_minicpm_llamacpp",
            "edge_minicpm_mlx",
        ],
    )
    parser.add_argument("--dataset", required=True, choices=["mvtec", "visa", "infra"])
    parser.add_argument("--category", required=True)
    parser.add_argument("--prompt", default="generic.simple")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--budget", type=float, default=None, help="Max USD to spend.")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    print(f"[run_vlm_eval] not yet implemented — args={vars(args)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
