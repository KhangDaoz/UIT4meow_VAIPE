#!/usr/bin/env bash
# Train YOLOv8 (default yolov8s) to detect drug-name regions on prescriptions, on Kaggle.
# Independent of scripts/kaggle_train_ocr.sh (YOLOv7 + VietOCR): it only adds yolov8/.
# Results (weights + plots + log) are copied to $OUT_DIR, which Kaggle keeps after "Save Version".
#
# Usage, from anywhere:
#   bash scripts/kaggle_train_yolov8.sh
# Optional env vars:
#   DATA_ROOT     public_train folder (auto-detected under /kaggle/input if unset)
#   OUT_DIR       where to put results           (default /kaggle/working/output)
#   YOLO_MODEL    pretrained weights             (default yolov8s.pt)
#   EPOCHS        max epochs                     (default 100)
#   PATIENCE      early-stop patience            (default 30)
#   YOLO_BATCH    batch size                     (default 16)
#   YOLO_DEVICE   '0', '0,1' (both T4) or 'cpu'  (default 0)
#   KEEP_LAST     1 = also save last.pt          (default 0)
set -uo pipefail
cd "$(dirname "$0")/.."

# ---- locate the dataset ------------------------------------------------------
if [ -z "${DATA_ROOT:-}" ]; then
    DATA_ROOT=$(find /kaggle/input -maxdepth 6 -type d -name public_train 2>/dev/null \
                | while read -r d; do [ -d "$d/prescription/label" ] && echo "$d"; done | head -1)
fi
if [ -z "${DATA_ROOT:-}" ] || [ ! -d "$DATA_ROOT/prescription/label" ]; then
    echo "ERROR: cannot find public_train/prescription/label. Add the vaipepill2022 dataset" \
         "to the notebook or set DATA_ROOT." >&2
    exit 1
fi
export DATA_ROOT
echo "DATA_ROOT = $DATA_ROOT"

OUT_DIR=${OUT_DIR:-/kaggle/working/output}
YOLO_MODEL=${YOLO_MODEL:-yolov8s.pt}
EPOCHS=${EPOCHS:-100}
PATIENCE=${PATIENCE:-30}
YOLO_BATCH=${YOLO_BATCH:-16}
YOLO_DEVICE=${YOLO_DEVICE:-0}
mkdir -p "$OUT_DIR/logs" "$OUT_DIR/yolov8"

export WANDB_MODE=disabled   # Kaggle has wandb installed; avoid the login prompt

if [ "$YOLO_DEVICE" != cpu ]; then
    nvidia-smi -L || { echo "ERROR: no GPU. Set Accelerator = GPU T4 x2." >&2; exit 1; }
fi

# ---- install + prepare data --------------------------------------------------
set -e
[ "${SKIP_INSTALL:-0}" = 1 ] || pip install -q "ultralytics==8.4.209"
python -c "import ultralytics; print('ultralytics', ultralytics.__version__)"

rm -rf dataset/pres/yolov7
python tools/generate_train_detect_pres.py
echo "train images: $(ls dataset/pres/yolov7/images/train | wc -l), val: $(ls dataset/pres/yolov7/images/val | wc -l)"
set +e

# ---- train -------------------------------------------------------------------
python yolov8/train.py --model "$YOLO_MODEL" --epochs "$EPOCHS" --patience "$PATIENCE" \
    --batch "$YOLO_BATCH" --device "$YOLO_DEVICE" --workers 4 \
    2>&1 | tee "$OUT_DIR/logs/yolov8.log" | stdbuf -oL tr '\r' '\n' \
    | sed -u 's/\x1b\[[0-9;]*[A-Za-z]//g' | grep --line-buffered -E \
    "^ +[0-9]+/[0-9]+ .*: 100%|^ +Epoch |^ +all |Error|Traceback|epochs completed|Results saved|EarlyStopping"
RC=${PIPESTATUS[0]}

# ---- collect results ---------------------------------------------------------
RUN=yolov8/runs/pres
cp "$RUN"/weights/best.pt "$OUT_DIR/yolov8/yolov8s_ocr.pt" 2>/dev/null
[ "${KEEP_LAST:-0}" = 1 ] && cp "$RUN"/weights/last.pt "$OUT_DIR/yolov8/last.pt" 2>/dev/null
cp "$RUN"/*.png "$RUN"/*.csv "$RUN"/*.yaml "$OUT_DIR/yolov8/" 2>/dev/null

# ---- zip everything into one file for easy download ---------------------------
# Made even if training failed, so the log is still easy to get.
ZIP="$OUT_DIR/yolov8_results.zip"
rm -f "$ZIP"
python - "$OUT_DIR" "$ZIP" <<'EOF'
import sys, zipfile
from pathlib import Path
out, zip_path = Path(sys.argv[1]), Path(sys.argv[2])
with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
    for sub in ('yolov8', 'logs'):
        for f in sorted((out / sub).rglob('*')):
            if f.is_file():
                zf.write(f, f.relative_to(out))
print(f'zip: {zip_path} ({zip_path.stat().st_size / 1e6:.1f} MB)')
for name in zipfile.ZipFile(zip_path).namelist():
    print('   ', name)
EOF

echo
echo "YOLOv8 exit code: $RC"
echo "Download: $ZIP"
exit "$RC"
