"""CLI: run a classical Anomalib baseline.

Examples::

    python scripts/run_classical_eval.py --model padim --dataset mvtec --category bottle
    python scripts/run_classical_eval.py --model patchcore --dataset visa --category candle --image-size 224
"""

from __future__ import annotations

import argparse
import sys

from vlm_anomaly.logging import configure_logging, get_logger


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train and evaluate a classical Anomalib baseline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", required=True, choices=["padim", "patchcore", "efficientad", "stfpm"])
    parser.add_argument("--dataset", required=True, choices=["mvtec", "visa"])
    parser.add_argument("--category", required=True)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-format", choices=["console", "json"], default="console")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging(json_logs=(args.log_format == "json"))
    log = get_logger(__name__)

    try:
        from vlm_anomaly.evaluators.classical_evaluator import ClassicalEvaluator
    except ImportError as exc:
        log.error("run_classical_eval.import_error", error=str(exc))
        print(str(exc), file=sys.stderr)
        return 1

    ev = ClassicalEvaluator(
        model_name=args.model,
        dataset_name=args.dataset,
        category=args.category,
        image_size=args.image_size,
    )

    try:
        result = ev.run()
    except Exception as exc:
        log.error("run_classical_eval.error", error=str(exc))
        return 1

    print(
        f"\n{args.model.upper():12s}  {args.dataset}/{args.category}\n"
        f"  AUROC     : {result.auroc:.4f}" if result.auroc is not None else "  AUROC     : N/A"
    )
    if result.f1 is not None:
        print(f"  F1        : {result.f1:.4f}")
    print(f"  Latency   : {result.mean_latency_ms/1000:.1f}s")
    print(f"  Cost      : $0.00 (on-device)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
