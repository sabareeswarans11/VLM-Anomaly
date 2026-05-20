#!/usr/bin/env bash
# Download and verify the MVTec Anomaly Detection dataset.
#
# Usage:
#   bash scripts/download_mvtec.sh
#   VLM_ANOMALY_DATA_DIR=/mnt/data bash scripts/download_mvtec.sh
#
# The archive is ~4.9 GB.  SHA-256 is verified before extraction.
# Skips download if all 15 category directories already exist.

set -euo pipefail

DATA_DIR="${VLM_ANOMALY_DATA_DIR:-./data}"
DEST="${DATA_DIR}/mvtec"
ARCHIVE="${DATA_DIR}/mvtec_anomaly_detection.tar.xz"
URL="https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420938113-1629952094/mvtec_anomaly_detection.tar.xz"
SHA256="cf4313b13603ab4c36b9b1ced58a1c1e43d0ff381c01f36e7f1aadf8e1b36f5d"

CATEGORIES=(bottle cable capsule carpet grid hazelnut leather metal_nut pill screw tile toothbrush transistor wood zipper)

echo "[mvtec] Target: ${DEST}"

# Skip if all categories present
all_present=true
for cat in "${CATEGORIES[@]}"; do
    if [[ ! -d "${DEST}/${cat}" ]]; then
        all_present=false
        break
    fi
done
if [[ "${all_present}" == "true" ]]; then
    echo "[mvtec] All 15 categories already present — skipping."
    exit 0
fi

mkdir -p "${DATA_DIR}"

# Download
if [[ ! -f "${ARCHIVE}" ]]; then
    echo "[mvtec] Downloading (~4.9 GB) …"
    curl -L --progress-bar -o "${ARCHIVE}" "${URL}"
else
    echo "[mvtec] Archive already downloaded."
fi

# Verify
echo "[mvtec] Verifying SHA-256 …"
if command -v sha256sum &>/dev/null; then
    echo "${SHA256}  ${ARCHIVE}" | sha256sum --check --quiet
elif command -v shasum &>/dev/null; then
    echo "${SHA256}  ${ARCHIVE}" | shasum -a 256 --check --quiet
else
    echo "[mvtec] WARNING: no sha256sum/shasum found — skipping checksum."
fi
echo "[mvtec] Checksum OK."

# Extract
echo "[mvtec] Extracting …"
mkdir -p "${DEST}"
tar -xJf "${ARCHIVE}" -C "${DEST}" --strip-components=1
echo "[mvtec] Done → ${DEST}"
