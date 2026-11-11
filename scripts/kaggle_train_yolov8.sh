#!/usr/bin/env bash
# Train the two models of plan.md on Kaggle, on the 1_vaipe split (train/val/test = 820/176/176):
#   block B: YOLOv8s, text boxes of 6 classes (drugname, quantity, usage, diagnose, date, other)
#   block C: VietOCR vgg_transformer fine-tuned on the crops of all text boxes (option C3),
#            plus the score of the pretrained model on the same test crops (option C2)
# Each model is scored on test at the end. Results go to $OUT_DIR and are zipped to results.zip.
#
# Usage, from anywhere:
#   bash scripts/kaggle_train_yolov8.sh
# Ways to run (GPU T4 x2):
#   one session, both models in parallel (default): YOLO on GPU 0, VietOCR on GPU 1
#   two sessions, as in plan.md:
#     session 1: TRAIN_OCR=0  YOLO_DEVICE=0,1 bash scripts/kaggle_train_yolov8.sh
#     session 2: TRAIN_YOLO=0 OCR_DEVICE=0    bash scripts/kaggle_train_yolov8.sh
# Optional env vars:
#   DATA_ROOT     1_vaipe folder, or public_train of vaipepill2022 (auto-detected under /kaggle/input)
#   MANIFEST      split for public_train            (default splits/pres_split_vaipe1.csv)
#   OUT_DIR       where to put results              (default /kaggle/working/output)
#   SEED          training seed                     (default 42)
#   TRAIN_YOLO    1 = train YOLOv8, 0 = skip        (default 1)
#   TRAIN_OCR     1 = train VietOCR, 0 = skip       (default 1)
#   YOLO_MODEL    pretrained weights                (default yolov8s.pt)
#   IMGSZ         YOLOv8 image size                 (default 1024)
#   EPOCHS        YOLOv8 max epochs                 (default 50)
#   PATIENCE      YOLOv8 early-stop patience        (default 15)
#   YOLO_BATCH    YOLOv8 batch size                 (default 16)
#   YOLO_DEVICE   YOLOv8 GPU: '0', '0,1' or 'cpu'   (default 0)
#   KEEP_LAST     1 = also save YOLOv8 last.pt      (default 0)
#   OCR_ITERS     VietOCR iterations                (default 20000)
#   OCR_VALID_EVERY  VietOCR val check interval     (default 2000)
#   OCR_BATCH     VietOCR batch size                (default 32)
#   OCR_MAX_WIDTH VietOCR image_max_width           (default 1024)
#   OCR_DEVICE    VietOCR GPU: '1', '0' or 'cpu'    (default 1)
#   OCR_CLASSES   classes cropped for VietOCR       (default all 6)
set -uo pipefail
cd "$(dirname "$0")/.."

OUT_DIR=${OUT_DIR:-/kaggle/working/output}
SEED=${SEED:-42}
TRAIN_YOLO=${TRAIN_YOLO:-1}
TRAIN_OCR=${TRAIN_OCR:-1}
YOLO_MODEL=${YOLO_MODEL:-yolov8s.pt}
IMGSZ=${IMGSZ:-1024}
EPOCHS=${EPOCHS:-50}
PATIENCE=${PATIENCE:-15}
YOLO_BATCH=${YOLO_BATCH:-16}
YOLO_DEVICE=${YOLO_DEVICE:-0}
OCR_ITERS=${OCR_ITERS:-20000}
OCR_VALID_EVERY=${OCR_VALID_EVERY:-2000}
OCR_BATCH=${OCR_BATCH:-32}
OCR_MAX_WIDTH=${OCR_MAX_WIDTH:-1024}
OCR_DEVICE=${OCR_DEVICE:-1}
OCR_CLASSES=${OCR_CLASSES:-drugname quantity usage diagnose date other}
mkdir -p "$OUT_DIR/logs"

export WANDB_MODE=disabled                  # Kaggle has wandb installed; avoid the login prompt
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1   # torch>=2.6 defaults torch.load(weights_only=True)

if { [ "$TRAIN_YOLO" = 1 ] && [ "$YOLO_DEVICE" != cpu ]; } || { [ "$TRAIN_OCR" = 1 ] && [ "$OCR_DEVICE" != cpu ]; }; then
    nvidia-smi -L || { echo "ERROR: no GPU. Set Accelerator = GPU T4 x2." >&2; exit 1; }
fi

# ---- locate the dataset ------------------------------------------------------
if [ -z "${DATA_ROOT:-}" ]; then
    DATA_ROOT=$(python tools/vaipe_data.py --find)
fi
if [ -z "${DATA_ROOT:-}" ]; then
    echo "ERROR: no data found under /kaggle/input. Add the 1_vaipe dataset (folders" \
         "train/valid/test with images/ and labels/) or vaipepill2022, or set DATA_ROOT." >&2
    exit 1
fi
export DATA_ROOT
MANIFEST_ARG=(); [ -n "${MANIFEST:-}" ] && MANIFEST_ARG=(--manifest "$MANIFEST")
echo "DATA_ROOT = $DATA_ROOT"
python tools/vaipe_data.py --data-root "$DATA_ROOT" "${MANIFEST_ARG[@]}" || exit 1

# ---- install -----------------------------------------------------------------
set -e
if [ "${SKIP_INSTALL:-0}" != 1 ]; then
    [ "$TRAIN_YOLO" = 1 ] && pip install -q "ultralytics==8.4.209"
    if [ "$TRAIN_OCR" = 1 ]; then
        # vietocr 0.3.13 pins old pillow/einops/imgaug; install it without its pins.
        pip install -q --no-deps vietocr==0.3.13
        pip install -q einops lmdb prefetch_generator albumentations gdown six
    fi
fi
[ "$TRAIN_YOLO" = 1 ] && python -c "import ultralytics; print('ultralytics', ultralytics.__version__)"
[ "$TRAIN_OCR" = 1 ] && (cd vietocr && python -c "import eval_vietocr; from vietocr.model.trainer import Trainer; print('vietocr import OK')")

# ---- prepare data --------------------------------------------------------------
[ "$TRAIN_YOLO" = 1 ] && python tools/prepare_yolo_pres.py --data-root "$DATA_ROOT" "${MANIFEST_ARG[@]}"
# shellcheck disable=SC2086
[ "$TRAIN_OCR" = 1 ] && python tools/prepare_vietocr_pres.py --data-root "$DATA_ROOT" "${MANIFEST_ARG[@]}" \
    --classes $OCR_CLASSES
set +e

# ---- train both models in parallel --------------------------------------------
YOLO_PID=; OCR_PID=
if [ "$TRAIN_YOLO" = 1 ]; then
    python yolov8/train.py --model "$YOLO_MODEL" --imgsz "$IMGSZ" --epochs "$EPOCHS" \
        --patience "$PATIENCE" --batch "$YOLO_BATCH" --device "$YOLO_DEVICE" --workers 4 \
        --seed "$SEED" > "$OUT_DIR/logs/yolov8.log" 2>&1 &
    YOLO_PID=$!
fi
if [ "$TRAIN_OCR" = 1 ]; then
    if [ "$OCR_DEVICE" = cpu ]; then OCR_ENV=""; OCR_DEV=cpu; else OCR_ENV=$OCR_DEVICE; OCR_DEV=cuda:0; fi
    CUDA_VISIBLE_DEVICES=$OCR_ENV python vietocr/train_vietocr_split.py \
        --iters "$OCR_ITERS" --valid-every "$OCR_VALID_EVERY" --batch-size "$OCR_BATCH" \
        --max-width "$OCR_MAX_WIDTH" --device "$OCR_DEV" --seed "$SEED" \
        > "$OUT_DIR/logs/vietocr.log" 2>&1 &
    OCR_PID=$!
fi

clean() { tr '\r' '\n' < "$1" | sed 's/\x1b\[[0-9;]*[A-Za-z]//g'; }
alive() { for p in "$@"; do [ -n "$p" ] && kill -0 "$p" 2>/dev/null && return 0; done; return 1; }
# Print progress every few minutes so the Kaggle log shows the run is alive.
while alive $YOLO_PID $OCR_PID; do
    sleep "${PROGRESS_EVERY:-300}"
    echo "==== $(date '+%H:%M:%S') ===="
    if [ -n "$YOLO_PID" ]; then
        echo "[yolov8]  $(clean "$OUT_DIR/logs/yolov8.log" | grep -E '^ +[0-9]+/[0-9]+ ' | tail -1 | cut -c1-110)"
        echo "[yolov8]  val: $(clean "$OUT_DIR/logs/yolov8.log" | grep -E '^ +all ' | tail -1)"
    fi
    [ -n "$OCR_PID" ] && echo "[vietocr] $(clean "$OUT_DIR/logs/vietocr.log" | grep -v '^\s*$' | tail -1 | cut -c1-160)"
done

YOLO_RC=skipped; [ -n "$YOLO_PID" ] && { YOLO_RC=0; wait $YOLO_PID || YOLO_RC=$?; }
OCR_RC=skipped;  [ -n "$OCR_PID" ]  && { OCR_RC=0;  wait $OCR_PID  || OCR_RC=$?; }

# ---- collect results ---------------------------------------------------------
cp splits/pres_split_vaipe1.csv "$OUT_DIR/split_manifest.csv" 2>/dev/null
if [ -n "$YOLO_PID" ]; then
    RUN=yolov8/runs/vaipe6
    mkdir -p "$OUT_DIR/yolov8/val" "$OUT_DIR/yolov8/test"
    cp "$RUN"/weights/best.pt "$OUT_DIR/yolov8/yolo_vaipe.pt" 2>/dev/null
    [ "${KEEP_LAST:-0}" = 1 ] && cp "$RUN"/weights/last.pt "$OUT_DIR/yolov8/last.pt" 2>/dev/null
    cp "$RUN"/*.png "$RUN"/*.csv "$RUN"/*.yaml "$RUN"/test_metrics.json "$OUT_DIR/yolov8/" 2>/dev/null
    cp "${RUN}_val"/*.png "$OUT_DIR/yolov8/val/" 2>/dev/null
    cp "${RUN}_test"/*.png "$OUT_DIR/yolov8/test/" 2>/dev/null
    cp dataset/pres/yolov8/stats.json dataset/pres/yolov8/data.yaml "$OUT_DIR/yolov8/" 2>/dev/null
    echo; echo "---- yolov8 (tail) ----"
    clean "$OUT_DIR/logs/yolov8.log" | grep -E "^(class|all|drugname|quantity|usage|diagnose|date|other) |conf that|epochs completed|EarlyStopping|TEST RESULT|Error|Traceback" | tail -n 14
fi
if [ -n "$OCR_PID" ]; then
    RUN=vietocr/runs_vaipe
    mkdir -p "$OUT_DIR/vietocr"
    cp "$RUN"/weights/transformerocr.pth "$OUT_DIR/vietocr/vietocr_vaipe.pth" 2>/dev/null
    cp "$RUN"/config.yml "$OUT_DIR/vietocr/vietocr_config.yml" 2>/dev/null
    cp "$RUN"/train.log "$RUN"/C*_metrics.json "$RUN"/C*_predictions.tsv "$OUT_DIR/vietocr/" 2>/dev/null
    cp dataset/pres/vietocr/stats.json "$OUT_DIR/vietocr/" 2>/dev/null
    echo; echo "---- vietocr (tail) ----"
    clean "$OUT_DIR/logs/vietocr.log" | grep -E "^(==|class|all|drugname|quantity|usage|diagnose|date|other) |TEST RESULT|Error|Traceback" | tail -n 20
fi

# ---- zip everything into one file for easy download ---------------------------
# Made even if training failed, so the logs are still easy to get.
ZIP="$OUT_DIR/results.zip"
rm -f "$ZIP"
python - "$OUT_DIR" "$ZIP" <<'EOF'
import sys, zipfile
from pathlib import Path
out, zip_path = Path(sys.argv[1]), Path(sys.argv[2])
with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
    for f in sorted(out.rglob('*')):
        if f.is_file() and f != zip_path and f.suffix != '.zip':
            zf.write(f, f.relative_to(out))
print(f'zip: {zip_path} ({zip_path.stat().st_size / 1e6:.1f} MB)')
for name in zipfile.ZipFile(zip_path).namelist():
    print('   ', name)
EOF

echo
echo "YOLOv8 exit code : $YOLO_RC"
echo "VietOCR exit code: $OCR_RC"
echo "Download: $ZIP"
[ "$YOLO_RC" != skipped ] && [ "$YOLO_RC" != 0 ] && exit 1
[ "$OCR_RC" != skipped ] && [ "$OCR_RC" != 0 ] && exit 1
exit 0
