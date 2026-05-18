#!/usr/bin/env bash
# Download MVTec AD into $VLM_ANOMALY_DATA_DIR (default ./data).
# Filled out in task 03.
set -euo pipefail

DATA_DIR="${VLM_ANOMALY_DATA_DIR:-./data}"
mkdir -p "$DATA_DIR"
echo "[mvtec] download not yet implemented — target dir: $DATA_DIR/mvtec"
exit 0
