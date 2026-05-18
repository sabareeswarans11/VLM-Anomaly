"""Entry point for the ``vlm-anomaly`` console script.

The real CLI is built out in later tasks. For now this prints the version.
"""

from __future__ import annotations

import argparse

from vlm_anomaly import __version__


def main() -> int:
    """Run the VLM-Anomaly CLI."""
    parser = argparse.ArgumentParser(prog="vlm-anomaly")
    parser.add_argument("--version", action="version", version=f"vlm-anomaly {__version__}")
    parser.parse_args()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
