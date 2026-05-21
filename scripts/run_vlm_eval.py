"""CLI: run a VLM zero-shot anomaly evaluation.

Examples::

    # Quick smoke run — no API keys needed
    python scripts/run_vlm_eval.py --backend mock --dataset mvtec \\
        --category bottle --limit 5

    # With budget cap
    python scripts/run_vlm_eval.py --backend mock --dataset mvtec \\
        --category bottle --limit 5 --budget 0.01

    # Cloud backend (requires API key)
    python scripts/run_vlm_eval.py --backend anthropic --dataset mvtec \\
        --category bottle --limit 10 --budget 1.00 --prompt generic.detailed
"""

from __future__ import annotations

import argparse
import sys

from vlm_anomaly.config import get_settings
from vlm_anomaly.logging import configure_logging, get_logger
from vlm_anomaly.schemas import ExperimentConfig


def _build_backend(name: str):
    if name == "mock":
        from vlm_anomaly.backends.mock import MockVLMBackend

        return MockVLMBackend()
    if name == "together":
        from vlm_anomaly.backends.together import TogetherBackend

        return TogetherBackend()
    if name == "gemini":
        from vlm_anomaly.backends.gemini import GeminiBackend

        return GeminiBackend()
    if name == "anthropic":
        from vlm_anomaly.backends.anthropic_backend import AnthropicBackend

        return AnthropicBackend()
    if name == "groq":
        from vlm_anomaly.backends.groq import GroqBackend

        return GroqBackend()
    if name == "edge_minicpm_llamacpp":
        from vlm_anomaly.backends.edge.minicpm_llamacpp import MiniCPMLlamaCppBackend

        return MiniCPMLlamaCppBackend()
    if name == "edge_minicpm_mlx":
        from vlm_anomaly.backends.edge.minicpm_mlx import MiniCPMMLXBackend

        return MiniCPMMLXBackend()
    raise ValueError(f"Unknown backend: {name!r}")


def _build_dataset(name: str, category: str | None):
    settings = get_settings()
    if name == "mvtec":
        from vlm_anomaly.datasets.mvtec import MVTec

        return MVTec(root_dir=settings.data_dir / "mvtec")
    if name == "visa":
        from vlm_anomaly.datasets.visa import VisA

        return VisA(root_dir=settings.data_dir / "visa")
    if name == "infra":
        from vlm_anomaly.datasets.infra import InfraAD

        return InfraAD(root_dir=settings.data_dir / "infra")
    raise ValueError(f"Unknown dataset: {name!r}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a VLM zero-shot anomaly evaluation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
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
    parser.add_argument(
        "--category",
        default=None,
        help="Category to evaluate. Omit to run all categories.",
    )
    parser.add_argument(
        "--prompt", default="generic.simple", help="Prompt key in format '<name>.<variant>'."
    )
    parser.add_argument("--limit", type=int, default=None, help="Max images per category.")
    parser.add_argument(
        "--budget", type=float, default=None, help="Max USD to spend (overrides env default)."
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-format", choices=["console", "json"], default="console")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging(json_logs=(args.log_format == "json"))
    log = get_logger(__name__)

    config = ExperimentConfig(
        backend=args.backend,
        dataset=args.dataset,
        categories=[args.category] if args.category else None,
        prompt=args.prompt,
        limit=args.limit,
        budget_usd=args.budget,
        seed=args.seed,
    )

    backend = _build_backend(args.backend)
    dataset = _build_dataset(args.dataset, args.category)

    from vlm_anomaly.evaluators.vlm_evaluator import VLMEvaluator

    evaluator = VLMEvaluator(backend=backend, dataset=dataset, config=config)

    try:
        results = evaluator.run()
    except Exception as exc:
        log.error("run_vlm_eval.error", error=str(exc))
        return 1

    for r in results:
        auroc = f"{r.auroc:.3f}" if r.auroc is not None else "N/A"
        f1 = f"{r.f1:.3f}" if r.f1 is not None else "N/A"
        print(
            f"{r.category:20s}  n={r.n_images:4d}  "
            f"auroc={auroc:>8}  f1={f1:>8}  "
            f"cost=${r.total_cost_usd:.4f}"
        )

    settings = get_settings()
    log.info("run_vlm_eval.results_dir", path=str(settings.results_dir))
    return 0


if __name__ == "__main__":
    sys.exit(main())
