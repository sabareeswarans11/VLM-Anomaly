"""Bundle a subset of results/ into a JSON the iOS app can ship offline.

Fleshed out in task 12.
"""

from __future__ import annotations

import argparse
import sys


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export results for the iOS app bundle.")
    parser.add_argument("--results-dir", default="./results")
    parser.add_argument("--out", default="ios/VLMAnomalyEdge/Resources/bundled_results.json")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    print(f"[export_results_for_app] not yet implemented — args={vars(args)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
