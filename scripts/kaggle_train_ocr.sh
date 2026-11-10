#!/usr/bin/env bash
# Train the prescription-OCR branch (YOLOv7x + VietOCR) on Kaggle with 2x T4.
# YOLOv7x runs on GPU 0 and VietOCR on GPU 1 at the same time.
#
# Usage (from the repo root, e.g. /kaggle/working/UIT4meow_VAIPE):
#   DATA_ROOT=/kaggle/input/<your-dataset>/public_train bash scripts/kaggle_train_ocr.sh
#
# DATA_ROOT must contain prescription/image/*.png and prescription/label/*.json
set -euo pipefail

: "${DATA_ROOT:?Set DATA_ROOT to the public_train folder}"
export DATA_ROOT
export WANDB_MODE=disabled                  # Kaggle has wandb installed; avoid login prompt
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1   # torch>=2.6 defaults torch.load(weights_only=True)

EPOCHS=${EPOCHS:-50}
YOLO_BATCH=${YOLO_BATCH:-8}

pip install -q vietocr Levenshtein
python -c "from vietocr.model.trainer import Trainer" # fail fast if vietocr deps are broken

# 1. Prepare data
python tools/generate_train_detect_pres.py
python tools/crop_pres.py
mkdir -p vietocr/weights logs

# 2. Train YOLOv7x (GPU 0) and VietOCR (GPU 1) in parallel
(cd yolov7 && CUDA_VISIBLE_DEVICES=0 python train.py \
    --epochs "$EPOCHS" --single-cls --workers 4 --device 0 \
    --batch-size "$YOLO_BATCH" --data data/coco.yaml --img 640 640 \
    --cfg cfg/training/yolov7x.yaml --weights '' \
    --name yolov7x --hyp data/hyp.scratch.p5.yaml) > logs/yolov7.log 2>&1 &
YOLO_PID=$!

# train_vietocr.py uses cuda:0, which maps to physical GPU 1 here
(cd vietocr && CUDA_VISIBLE_DEVICES=1 python train_vietocr.py) > logs/vietocr.log 2>&1 &
OCR_PID=$!

YOLO_RC=0; wait $YOLO_PID || YOLO_RC=$?; echo "YOLOv7 exit: $YOLO_RC (log: logs/yolov7.log)"
OCR_RC=0;  wait $OCR_PID  || OCR_RC=$?;  echo "VietOCR exit: $OCR_RC (log: logs/vietocr.log)"

echo "YOLOv7 weights : yolov7/runs/train/yolov7x*/weights/best.pt"
echo "VietOCR weights: vietocr/weights/transformerocr.pth"
