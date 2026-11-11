#!/usr/bin/env bash
# Train YOLOv8 (default yolov8s, drug-name boxes) + VietOCR (drug-name text) on Kaggle,
# both on the same reproducible train/val/test = 70/15/15 split of the prescriptions.
# YOLOv8 runs on GPU 0 and VietOCR on GPU 1 at the same time; each is scored on test at the end.
# Independent of scripts/kaggle_train_ocr.sh (YOLOv7 + VietOCR, old 90/10 and 80/20 splits).
# Results are copied to $OUT_DIR (kept by Kaggle after "Save Version") and zipped to results.zip.
#
# Usage, from anywhere:
#   bash scripts/kaggle_train_yolov8.sh
# Optional env vars:
#   DATA_ROOT     public_train folder (auto-detected under /kaggle/input if unset)
#   OUT_DIR       where to put results            (default /kaggle/working/output)
#   SEED          split + training seed           (default 42)
#                 the split is read from splits/pres_split_seed<SEED>.json (committed);
#                 if that file is missing it is created with this seed.
#   SPLIT         use another committed split instead: splits/pres_split_<SPLIT>.csv or .json
#                 e.g. SPLIT=vaipe1 -> the 1_vaipe split (grouped by patient, 820/176/176);
#                 SEED is then only the training seed. Unset = the seed split above.
#   TRAIN_YOLO    1 = train YOLOv8, 0 = skip      (default 1)
#   TRAIN_OCR     1 = train VietOCR, 0 = skip     (default 1)
#   YOLO_MODEL    pretrained weights              (default yolov8s.pt)
#   EPOCHS        YOLOv8 max epochs               (default 100)
#   PATIENCE      YOLOv8 early-stop patience      (default 30)
#   YOLO_BATCH    YOLOv8 batch size               (default 16)
#   YOLO_DEVICE   YOLOv8 GPU: '0' or 'cpu'        (default 0)
#   KEEP_LAST     1 = also save YOLOv8 last.pt    (default 0)
#   OCR_ITERS     VietOCR iterations              (default 20000)
#   OCR_VALID_EVERY  VietOCR val check interval   (default 3000)
#   OCR_BATCH     VietOCR batch size              (default 32)
#   OCR_DEVICE    VietOCR GPU: '1' or 'cpu'       (default 1)
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
SEED=${SEED:-42}
SPLIT=${SPLIT:-}
if [ -n "$SPLIT" ]; then
    MANIFEST=$(ls splits/pres_split_"$SPLIT".csv splits/pres_split_"$SPLIT".json 2>/dev/null | head -1)
    if [ -z "$MANIFEST" ]; then
        echo "ERROR: SPLIT=$SPLIT but splits/pres_split_$SPLIT.csv/.json not found." >&2
        exit 1
    fi
    MANIFEST_ARG=(--manifest "$MANIFEST")
else
    MANIFEST=splits/pres_split_seed"$SEED".json   # created by prepare_yolo_pres.py if missing
    MANIFEST_ARG=()
fi
echo "SPLIT manifest = $MANIFEST"
TRAIN_YOLO=${TRAIN_YOLO:-1}
TRAIN_OCR=${TRAIN_OCR:-1}
YOLO_MODEL=${YOLO_MODEL:-yolov8s.pt}
EPOCHS=${EPOCHS:-100}
PATIENCE=${PATIENCE:-30}
YOLO_BATCH=${YOLO_BATCH:-16}
YOLO_DEVICE=${YOLO_DEVICE:-0}
OCR_ITERS=${OCR_ITERS:-20000}
OCR_VALID_EVERY=${OCR_VALID_EVERY:-3000}
OCR_BATCH=${OCR_BATCH:-32}
OCR_DEVICE=${OCR_DEVICE:-1}
mkdir -p "$OUT_DIR/logs"

export WANDB_MODE=disabled                  # Kaggle has wandb installed; avoid the login prompt
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1   # torch>=2.6 defaults torch.load(weights_only=True)

if [ "$YOLO_DEVICE" != cpu ] || [ "$OCR_DEVICE" != cpu ]; then
    nvidia-smi -L || { echo "ERROR: no GPU. Set Accelerator = GPU T4 x2." >&2; exit 1; }
fi

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
if [ "$TRAIN_OCR" = 1 ]; then
    python - <<'EOF'
import sys, types
try:
    import imgaug
except Exception:
    m = types.ModuleType('imgaug'); m.augmenters = types.ModuleType('imgaug.augmenters')
    sys.modules['imgaug'] = m; sys.modules['imgaug.augmenters'] = m.augmenters
from vietocr.model.trainer import Trainer
print('vietocr import OK')
EOF
fi

# ---- prepare data: train/val/test = 70/15/15, fixed by the split manifest -----
# also makes the seed manifest if missing (not when SPLIT is set)
python tools/prepare_yolo_pres.py --data-root "$DATA_ROOT" --seed "$SEED" "${MANIFEST_ARG[@]}"
[ "$TRAIN_OCR" = 1 ] && python tools/prepare_vietocr_pres.py --data-root "$DATA_ROOT" --seed "$SEED" "${MANIFEST_ARG[@]}"
set +e

# ---- train both models in parallel --------------------------------------------
YOLO_PID=; OCR_PID=
if [ "$TRAIN_YOLO" = 1 ]; then
    python yolov8/train.py --model "$YOLO_MODEL" --epochs "$EPOCHS" --patience "$PATIENCE" \
        --batch "$YOLO_BATCH" --device "$YOLO_DEVICE" --workers 4 --seed "$SEED" \
        > "$OUT_DIR/logs/yolov8.log" 2>&1 &
    YOLO_PID=$!
fi
if [ "$TRAIN_OCR" = 1 ]; then
    if [ "$OCR_DEVICE" = cpu ]; then OCR_ENV=""; OCR_DEV=cpu; else OCR_ENV=$OCR_DEVICE; OCR_DEV=cuda:0; fi
    CUDA_VISIBLE_DEVICES=$OCR_ENV python vietocr/train_vietocr_split.py \
        --iters "$OCR_ITERS" --valid-every "$OCR_VALID_EVERY" --batch-size "$OCR_BATCH" \
        --device "$OCR_DEV" --seed "$SEED" > "$OUT_DIR/logs/vietocr.log" 2>&1 &
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
cp "$MANIFEST" "$OUT_DIR/split_manifest.${MANIFEST##*.}" 2>/dev/null
if [ -n "$YOLO_PID" ]; then
    RUN=yolov8/runs/pres
    mkdir -p "$OUT_DIR/yolov8/test"
    cp "$RUN"/weights/best.pt "$OUT_DIR/yolov8/yolov8s_ocr.pt" 2>/dev/null
    [ "${KEEP_LAST:-0}" = 1 ] && cp "$RUN"/weights/last.pt "$OUT_DIR/yolov8/last.pt" 2>/dev/null
    cp "$RUN"/*.png "$RUN"/*.csv "$RUN"/*.yaml "$RUN"/test_metrics.json "$OUT_DIR/yolov8/" 2>/dev/null
    cp "${RUN}_test"/*.png "$OUT_DIR/yolov8/test/" 2>/dev/null
    cp dataset/pres/yolov8/stats.json "$OUT_DIR/yolov8/" 2>/dev/null
    echo; echo "---- yolov8 (tail) ----"
    clean "$OUT_DIR/logs/yolov8.log" | grep -E "^ +all |epochs completed|EarlyStopping|TEST RESULT|Error|Traceback" | tail -n 6
fi
if [ -n "$OCR_PID" ]; then
    RUN=vietocr/runs_split
    mkdir -p "$OUT_DIR/vietocr"
    cp "$RUN"/weights/transformerocr.pth "$OUT_DIR/vietocr/vietocr.pt" 2>/dev/null
    cp "$RUN"/config.yml "$RUN"/train.log "$RUN"/test_metrics.json "$RUN"/test_predictions.tsv \
       "$OUT_DIR/vietocr/" 2>/dev/null
    cp dataset/pres/vietocr_split/stats.json "$OUT_DIR/vietocr/" 2>/dev/null
    echo; echo "---- vietocr (tail) ----"
    clean "$OUT_DIR/logs/vietocr.log" | grep -v '^\s*$' | tail -n 6
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
