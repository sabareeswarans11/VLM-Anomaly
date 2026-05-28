#!/usr/bin/env bash
# Download MiniCPM-V-4.6 GGUF files from ggml-org/MiniCPM-V-4.6-GGUF on HuggingFace.
#
# Downloads two files:
#   MiniCPM-V-4.6-Q4_K_M.gguf       — language model (~2.6 GB)
#   mmproj-MiniCPM-V-4.6-Q8_0.gguf   — vision encoder  (~0.5 GB)
#
# Target directory: $VLM_ANOMALY_MODELS_DIR/minicpm-v/  (default: ./models/minicpm-v)
set -euo pipefail

HF_BASE="https://huggingface.co/ggml-org/MiniCPM-V-4.6-GGUF/resolve/main"
MODELS_DIR="${VLM_ANOMALY_MODELS_DIR:-./models}/minicpm-v"

mkdir -p "$MODELS_DIR"

download() {
    local name="$1"
    local url="$HF_BASE/$name"
    local dest="$MODELS_DIR/$name"
    if [ -f "$dest" ]; then
        echo "[minicpm-v] $name already present — skipping."
        return
    fi
    echo "[minicpm-v] Downloading $name …"
    if command -v wget &>/dev/null; then
        wget --continue --show-progress -O "$dest.part" "$url"
    elif command -v curl &>/dev/null; then
        curl -L --progress-bar -C - -o "$dest.part" "$url"
    else
        echo "[minicpm-v] ERROR: neither wget nor curl found." >&2
        exit 1
    fi
    mv "$dest.part" "$dest"
    echo "[minicpm-v] Saved: $dest"
}

download "MiniCPM-V-4.6-Q4_K_M.gguf"
download "mmproj-MiniCPM-V-4.6-Q8_0.gguf"

echo "[minicpm-v] All files ready in $MODELS_DIR"
