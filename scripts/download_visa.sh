#!/usr/bin/env bash
# Download VisA into $VLM_ANOMALY_DATA_DIR (default ./data).
# Filled out in task 03.
set -euo pipefail

DATA_DIR="${VLM_ANOMALY_DATA_DIR:-./data}"
mkdir -p "$DATA_DIR"
echo "[visa] download not yet implemented — target dir: $DATA_DIR/visa"
exit 0
