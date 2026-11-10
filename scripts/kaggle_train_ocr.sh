#!/usr/bin/env bash
# Train the prescription-OCR branch (YOLOv7x + VietOCR) on Kaggle (GPU T4 x2).
# YOLOv7x runs on GPU 0 and VietOCR on GPU 1 at the same time.
# Results (weights + logs) are copied to $OUT_DIR, which Kaggle keeps after "Save Version".
#
# Usage, from the repo root:
#   bash scripts/kaggle_train_ocr.sh
# Optional env vars:
#   DATA_ROOT   public_train folder (auto-detected under /kaggle/input if unset)
#   OUT_DIR     where to put results           (default /kaggle/working/output)
#   EPOCHS      YOLOv7 epochs                  (default 50)
#   YOLO_BATCH  YOLOv7 batch size              (default 8)
#   OCR_ITERS   VietOCR iterations             (default 20000)
set -uo pipefail
cd "$(dirname "$0")/.."
REPO=$(pwd)

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
EPOCHS=${EPOCHS:-50}
YOLO_BATCH=${YOLO_BATCH:-8}
export OCR_ITERS=${OCR_ITERS:-20000}
mkdir -p "$OUT_DIR/logs"

export WANDB_MODE=disabled                  # Kaggle has wandb installed; avoid the login prompt
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1   # torch>=2.6 defaults torch.load(weights_only=True)

nvidia-smi -L || { echo "ERROR: no GPU. Set Accelerator = GPU T4 x2." >&2; exit 1; }

# ---- install -----------------------------------------------------------------
set -e
# vietocr 0.3.13 pins old pillow/einops/imgaug; install it without its pins.
pip install -q --no-deps vietocr==0.3.13
pip install -q einops lmdb prefetch_generator albumentations Levenshtein
python - <<'EOF'
import sys, types
try:
    import imgaug
except Exception:
    m = types.ModuleType('imgaug'); m.augmenters = types.ModuleType('imgaug.augmenters')
    sys.modules['imgaug'] = m; sys.modules['imgaug.augmenters'] = m.augmenters
from vietocr.model.trainer import Trainer
print("vietocr import OK")
EOF

# ---- prepare data ------------------------------------------------------------
rm -rf dataset/pres
python tools/generate_train_detect_pres.py
python tools/crop_pres.py
echo "YOLO train images: $(ls dataset/pres/yolov7/images/train | wc -l), val: $(ls dataset/pres/yolov7/images/val | wc -l)"
echo "OCR train lines:   $(wc -l < dataset/pres/vietocr/train.txt), val: $(wc -l < dataset/pres/vietocr/val.txt)"
set +e

# ---- train both models in parallel --------------------------------------------
(cd yolov7 && CUDA_VISIBLE_DEVICES=0 python train.py \
    --epochs "$EPOCHS" --single-cls --workers 4 --device 0 \
    --batch-size "$YOLO_BATCH" --data data/vaipe_pres.yaml --img 640 640 \
    --cfg cfg/training/yolov7x.yaml --weights '' \
    --name yolov7x --exist-ok --hyp data/hyp.scratch.p5.yaml) > "$OUT_DIR/logs/yolov7.log" 2>&1 &
YOLO_PID=$!

# train_vietocr.py uses cuda:0, which maps to physical GPU 1 here
(cd vietocr && CUDA_VISIBLE_DEVICES=1 python train_vietocr.py) > "$OUT_DIR/logs/vietocr.log" 2>&1 &
OCR_PID=$!

# Print progress every 10 minutes so the Kaggle log shows the run is alive.
while kill -0 $YOLO_PID 2>/dev/null || kill -0 $OCR_PID 2>/dev/null; do
    sleep 600
    echo "==== $(date '+%H:%M:%S') ===="
    echo "[yolov7]  $(tail -c 400 "$OUT_DIR/logs/yolov7.log" | tr '\r' '\n' | grep -v '^\s*$' | tail -1)"
    echo "[vietocr] $(tail -n 1 "$OUT_DIR/logs/vietocr.log")"
done

YOLO_RC=0; wait $YOLO_PID || YOLO_RC=$?
OCR_RC=0;  wait $OCR_PID  || OCR_RC=$?

# ---- collect results ---------------------------------------------------------
RUN=yolov7/runs/train/yolov7x
mkdir -p "$OUT_DIR/yolov7" "$OUT_DIR/vietocr"
cp "$RUN"/weights/best.pt "$OUT_DIR/yolov7/yolov7_ocr.pt" 2>/dev/null
cp "$RUN"/weights/last.pt "$OUT_DIR/yolov7/last.pt" 2>/dev/null
cp "$RUN"/*.png "$RUN"/*.txt "$RUN"/*.yaml "$OUT_DIR/yolov7/" 2>/dev/null
cp vietocr/weights/transformerocr.pth "$OUT_DIR/vietocr/vietocr.pt" 2>/dev/null
cp vietocr/config.yml vietocr/train.log "$OUT_DIR/vietocr/" 2>/dev/null

echo
echo "YOLOv7 exit code : $YOLO_RC"
echo "VietOCR exit code: $OCR_RC"
tail -n 15 "$OUT_DIR/logs/yolov7.log"  | tr '\r' '\n' | tail -n 15
tail -n 15 "$OUT_DIR/logs/vietocr.log"
echo
ls -lhR "$OUT_DIR"
[ $YOLO_RC -eq 0 ] && [ $OCR_RC -eq 0 ]
