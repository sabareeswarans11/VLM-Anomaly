"""Run an Anomalib classical baseline. Fleshed out in task 07."""

from __future__ import annotations

import argparse
import sys


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a classical anomaly-detection baseline.")
    parser.add_argument(
        "--model",
        required=True,
        choices=["padim", "patchcore", "efficientad", "stfpm"],
    )
    parser.add_argument("--dataset", required=True, choices=["mvtec", "visa"])
    parser.add_argument("--category", required=True)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    print(f"[run_classical_eval] not yet implemented — args={vars(args)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
