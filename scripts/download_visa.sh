#!/usr/bin/env bash
# Download the VisA (Visual Anomaly) dataset.
#
# Usage:
#   bash scripts/download_visa.sh
#   VLM_ANOMALY_DATA_DIR=/mnt/data bash scripts/download_visa.sh
#
# VisA is hosted on AWS S3 by Amazon Science.  The dataset requires
# accepting a licence; the download link below is the public S3 URL
# from the official spot-diff repository.
#
# The archive is ~9 GB.  Skips download if all 12 category directories
# already exist in MVTec-style layout.

set -euo pipefail

DATA_DIR="${VLM_ANOMALY_DATA_DIR:-./data}"
DEST="${DATA_DIR}/visa"
ARCHIVE="${DATA_DIR}/VisA_20220922.tar"
# Official public S3 URL from amazon-science/spot-diff
URL="https://amazon-visual-anomaly.s3.us-east-2.amazonaws.com/VisA_20220922.tar"

CATEGORIES=(candle capsules cashew chewinggum fryum macaroni1 macaroni2 pcb1 pcb2 pcb3 pcb4 pipe_fryum)

echo "[visa] Target: ${DEST}"

# Skip if all categories present in MVTec-style layout
all_present=true
for cat in "${CATEGORIES[@]}"; do
    if [[ ! -d "${DEST}/${cat}/train" ]]; then
        all_present=false
        break
    fi
done
if [[ "${all_present}" == "true" ]]; then
    echo "[visa] All 12 categories already present — skipping."
    exit 0
fi

mkdir -p "${DATA_DIR}"

# Download
if [[ ! -f "${ARCHIVE}" ]]; then
    echo "[visa] Downloading (~9 GB) …"
    curl -L --progress-bar -o "${ARCHIVE}" "${URL}"
else
    echo "[visa] Archive already downloaded."
fi

# Extract raw layout
echo "[visa] Extracting raw VisA …"
mkdir -p "${DEST}_raw"
tar -xf "${ARCHIVE}" -C "${DEST}_raw"

# Convert to MVTec-style layout using Python (handles the split CSV)
echo "[visa] Converting to MVTec-style layout …"
python3 - <<'PY'
import csv, os, shutil, pathlib

raw_root = pathlib.Path(os.environ.get("VLM_ANOMALY_DATA_DIR", "./data")) / "visa_raw"
dest_root = pathlib.Path(os.environ.get("VLM_ANOMALY_DATA_DIR", "./data")) / "visa"

CATEGORIES = [
    "candle","capsules","cashew","chewinggum","fryum",
    "macaroni1","macaroni2","pcb1","pcb2","pcb3","pcb4","pipe_fryum",
]

for cat in CATEGORIES:
    csv_path = raw_root / cat / "split_csv" / "train.csv"
    if not csv_path.exists():
        print(f"  [skip] {cat}: no split CSV found at {csv_path}")
        continue

    for split_name, csv_name in [("train", "train.csv"), ("test", "val.csv")]:
        split_csv = raw_root / cat / "split_csv" / csv_name
        if not split_csv.exists():
            continue
        with split_csv.open() as f:
            reader = csv.DictReader(f)
            for row in reader:
                src = raw_root / cat / row["image"]
                label = row.get("label", "").lower()
                if label == "normal":
                    dst = dest_root / cat / split_name / "good" / src.name
                else:
                    defect = row.get("defect", "bad")
                    dst = dest_root / cat / split_name / defect / src.name
                    # Copy mask if available
                    mask_src = raw_root / cat / row.get("mask", "")
                    if mask_src.exists():
                        mask_dst = dest_root / cat / "ground_truth" / defect / f"{src.stem}_mask.png"
                        mask_dst.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(mask_src, mask_dst)
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
    print(f"  [done] {cat}")

print("[visa] Conversion complete.")
PY

echo "[visa] Done → ${DEST}"
