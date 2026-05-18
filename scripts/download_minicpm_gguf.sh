#!/usr/bin/env bash
# Pull the quantized MiniCPM-V GGUF into $VLM_ANOMALY_MODELS_DIR (default ./models).
# Filled out in task 05.
set -euo pipefail

MODELS_DIR="${VLM_ANOMALY_MODELS_DIR:-./models}"
mkdir -p "$MODELS_DIR"
echo "[minicpm-v] GGUF download not yet implemented — target dir: $MODELS_DIR/minicpm-v"
exit 0
