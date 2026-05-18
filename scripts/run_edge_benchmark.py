"""Measure first-token latency and tok/s for an edge VLM on macOS.

This is the macOS proxy for the iPhone benchmark in §1 of CLAUDE.md.
Fleshed out in task 05.
"""

from __future__ import annotations

import argparse
import sys


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Edge VLM latency / throughput benchmark.")
    parser.add_argument("--model", default="minicpm-v")
    parser.add_argument("--prompt", default="prompts/generic.yaml")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument(
        "--backend",
        default="edge_minicpm_llamacpp",
        choices=["edge_minicpm_llamacpp", "edge_minicpm_mlx"],
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    print(f"[run_edge_benchmark] not yet implemented — args={vars(args)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
