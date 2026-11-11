# Train YOLOv8 + VietOCR trên Kaggle (theo `plan.md`)

Luồng này train 2 model học sâu của đồ án, trên **bộ chia `1_vaipe`** (train / val / test = **820 / 176 / 176**, gom theo bệnh nhân, xem `1_vaipe/README.md`):

| Khối | Model | Nhiệm vụ | Kết quả |
|---|---|---|---|
| B (plan 5.2) | YOLOv8s | Phát hiện **và phân lớp** vùng chữ, **6 lớp**: `drugname, quantity, usage, diagnose, date, other` | `output/yolov8/yolo_vaipe.pt` |
| C (plan 5.3) | VietOCR `vgg_transformer` | Đọc chữ của **mọi vùng chữ** (cả 6 lớp), fine-tune từ pretrained = phương án **C3** | `output/vietocr/vietocr_vaipe.pth` |

Cuối mỗi model đều **chấm trên tập test** (176 ảnh, chỉ chấm 1 lần). Script VietOCR còn chấm luôn **model pretrained chưa fine-tune (C2)** trên cùng các vùng cắt test, nên bảng E-C1 có sẵn 2 dòng C2 và C3.

Luồng YOLOv7 cũ (`KAGGLE_TRAIN.md`, `scripts/kaggle_train_ocr.sh`) không bị sửa.

---

## Dữ liệu (nhóm dùng cách B)

Dùng luôn **dataset gốc có sẵn trên Kaggle `tommyngx/vaipepill2022`**, không cần upload gì.
Script tự tìm thư mục `public_train` trong `/kaggle/input` và lấy bộ chia từ file **`splits/pres_split_vaipe1.csv`** (đã commit trong repo, là bản sao y nguyên của `1_vaipe/split_manifest.csv`). Không cần đặt biến nào.

- Danh sách file của `public_train` khớp đúng với CSV: 1.173 đơn thuốc, gồm cả ảnh `VAIPE_P_TRAIN_1172`. Ảnh này bị CSV loại (`split = excluded`) vì trùng với một ảnh test không nhãn, script tự bỏ qua.
- Các dòng `test_unlabeled` trong CSV bị bỏ qua. `valid` trong CSV được hiểu là tập val.
- Nếu dữ liệu không khớp CSV (thiếu hoặc thừa file), script **dừng và báo lỗi**, không âm thầm chia lại.
- Đừng sửa hay xoá `splits/pres_split_vaipe1.csv`: đây là nguồn duy nhất của bộ chia.

Đầu log phải ra đúng như sau:
```
DATA_ROOT = /kaggle/input/.../public_train
... split by pres_split_vaipe1.csv (pres_split_vaipe1.csv, {'train': 820, 'val': 176, 'test': 176})
train: 820 images, drugname=1995, quantity=1995, usage=2809, diagnose=1109, date=820, other=20108
val  : 176 images, drugname=405, quantity=405, usage=583, diagnose=287, date=176, other=4402
test : 176 images, drugname=443, quantity=443, usage=602, diagnose=253, date=176, other=4169
```

> Cách khác (không dùng): upload thư mục `1_vaipe` thành Kaggle dataset riêng. Script cũng nhận dạng này (thư mục `train/`, `valid/`, `test/` có `images/` và `labels/`) và ưu tiên nó nếu gắn cả hai. Bộ chia giống hệt cách B.

## Tham số mặc định (theo plan)

**YOLOv8** (`yolov8/train.py`): `yolov8s.pt`, `imgsz=1024`, `epochs=50`, `batch=16`, `patience=15`, `seed=42`, `fliplr=0` (lật ngang làm hỏng chữ), `mosaic=0.5`.
`best.pt` chọn theo val. Sau khi train, script tìm **ngưỡng conf tối đa F1 trung bình trên val** (`conf_max_f1_on_val`) rồi chấm test.

**VietOCR** (`vietocr/train_vietocr_split.py`): fine-tune từ `vgg_transformer` pretrained, `20000 iter`, `batch 32`, val mỗi `2000 iter`.
- Vùng cắt theo **box chuẩn**, đệm 2 px, chữ chuẩn hoá Unicode NFC. Khoảng 28,8 nghìn vùng train / 6,3 nghìn val / 6,1 nghìn test.
- `image_max_width = 1024` (mặc định của VietOCR là 512). Khoảng 5 % vùng chữ (dòng chẩn đoán, dòng dài) rộng hơn 512 px khi đưa về cao 32 px, nên ở 512 chúng bị bóp méo.
- Train trên **train**, chọn weights theo độ chính xác khớp nguyên dòng trên **val**, chấm trên **test**. Plan ghi "train trên train + valid", nhưng như vậy sẽ không còn tập để chọn checkpoint, nên ở đây val chỉ dùng để chọn.
- Độ đo: **CER, WER, tỉ lệ đúng nguyên dòng**, cho toàn tập và **từng lớp** (`drugname`, `quantity`, ...), cùng thời gian/vùng.

---

## Bước 0: Đẩy code lên GitHub

Notebook Kaggle `git clone` repo từ GitHub, nên mọi thay đổi phải được **commit và push lên `main`** trước khi chạy. Nếu không, Kaggle sẽ chạy code cũ.

## Bước 1: Tạo notebook

1. **Kaggle → Create → New Notebook**.
2. **Add Input** → tìm `vaipepill2022` (của `tommyngx`) → **Add**.
3. **Session options** (bên phải): Accelerator **GPU T4 x2**, Internet **On** (cần để clone repo và tải weights pretrained; tài khoản phải xác minh số điện thoại mới bật được).
4. Một cell code:
   ```python
   !rm -rf /tmp/repo && git clone --depth 1 https://github.com/KhangDaoz/UIT4meow_VAIPE.git /tmp/repo
   !bash /tmp/repo/scripts/kaggle_train_yolov8.sh
   ```

Có 2 cách chạy:

| Cách chạy | Lệnh | Ghi chú |
|---|---|---|
| **1 session**, 2 model song song (mặc định) | `!bash /tmp/repo/scripts/kaggle_train_yolov8.sh` | YOLO dùng GPU 0, VietOCR dùng GPU 1 |
| **2 session** như plan | Session 1: `!TRAIN_OCR=0 YOLO_DEVICE=0,1 bash /tmp/repo/scripts/kaggle_train_yolov8.sh`<br>Session 2: `!TRAIN_YOLO=0 OCR_DEVICE=0 bash /tmp/repo/scripts/kaggle_train_yolov8.sh` | YOLO chạy DDP trên 2 GPU, batch 16 chia đều 2 GPU |

## Bước 2: Chạy thử nhanh (~15 phút)

```python
!EPOCHS=1 OCR_ITERS=400 OCR_VALID_EVERY=200 bash /tmp/repo/scripts/kaggle_train_yolov8.sh
```
Kiểm tra trong log:
- `DATA_ROOT = /kaggle/input/.../public_train`, dòng `split by pres_split_vaipe1.csv` với `{'train': 820, 'val': 176, 'test': 176}`, và số box từng lớp như mục **Dữ liệu**.
- `vietocr import OK`, `train: 28836 crops`.
- Cuối log có bảng từng lớp của YOLO, `TEST RESULT (YOLOv8): ...`, hai bảng `== C2_pretrained` và `== C3_finetune`, `TEST RESULT (VietOCR C3): ...`.
- `YOLOv8 exit code : 0` và `VietOCR exit code: 0`.

Chạy thử xong thì bỏ các biến ở đầu lệnh đi.

## Bước 3: Save Version để treo máy

1. Sửa cell lại, bỏ các biến chạy thử:
   ```python
   !rm -rf /tmp/repo && git clone --depth 1 https://github.com/KhangDaoz/UIT4meow_VAIPE.git /tmp/repo
   !bash /tmp/repo/scripts/kaggle_train_yolov8.sh
   ```
2. **Save Version** → **Save & Run All (Commit)** → *Advanced Settings*: Always save output, Accelerator **GPU T4 x2** → **Save**.
3. Có thể tắt máy, Kaggle vẫn chạy tiếp.

- YOLOv8s 1024: khoảng **1,5–2,5 giờ** (dừng sớm nếu 15 epoch liền mAP val không tăng).
- VietOCR 20k iter: khoảng **3–5 giờ** (lâu hơn bản cũ vì nhiều vùng hơn và ảnh rộng tới 1024).
  Nếu thiếu thời gian, đặt `OCR_ITERS=10000`.
- Cứ 5 phút log in tiến độ của từng model.

## Bước 4: Lấy kết quả

```
output/
├── results.zip                       ← TẢI FILE NÀY (chứa mọi thứ bên dưới)
├── split_manifest.csv
├── yolov8/
│   ├── yolo_vaipe.pt                 ← weights khối B (best theo val)
│   ├── test_metrics.json             ← E-B2: P/R/F1/mAP50/mAP50-95 toàn bộ + từng lớp (test và val), conf tối ưu
│   ├── test/confusion_matrix*.png    ← ma trận nhầm lẫn 6 lớp trên test
│   ├── val/                          ← các đường cong trên val
│   ├── results.csv / results.png     ← loss, mAP theo epoch
│   ├── args.yaml, data.yaml, stats.json
├── vietocr/
│   ├── vietocr_vaipe.pth             ← weights khối C (C3)
│   ├── vietocr_config.yml            ← config để load weights (image_max_width=1024)
│   ├── C3_finetune_metrics.json      ← E-C1 dòng C3: CER/WER/acc toàn bộ + từng lớp, ms/vùng
│   ├── C3_finetune_predictions.tsv   ← từng vùng test: lớp, nhãn đúng, dự đoán
│   ├── C2_pretrained_metrics.json    ← E-C1 dòng C2
│   ├── C2_pretrained_predictions.tsv
│   ├── stats.json, train.log
└── logs/ (yolov8.log, vietocr.log)
```

Cách tải: mở version → tab **Output** → `output/results.zip`. Nếu zip lỗi, tải riêng `yolo_vaipe.pt`, `vietocr_vaipe.pth`, `vietocr_config.yml` và các file `*_metrics.json`.
Bằng CLI: `kaggle kernels output <user>/<notebook-slug> -p ./kaggle_out`.
File zip được tạo kể cả khi train lỗi, để vẫn lấy được log.

## Dùng weights ở máy local

Đặt weights vào `weights/` (đã gitignore). Chấm lại VietOCR trên vùng cắt test (cho E-C1 / E-C2):
```bash
python tools/prepare_vietocr_pres.py --data-root 1_vaipe            # -> dataset/pres/vietocr
python vietocr/eval_vietocr.py --weights pretrained --name C2_pretrained --max-width 1024
python vietocr/eval_vietocr.py --weights weights/vietocr_vaipe.pth \
    --config weights/vietocr_config.yml --name C3_finetune
# -> results/E-C1/<name>_metrics.json, <name>_predictions.tsv
```
Chấm OCR khác (Tesseract C0/C1) bằng cùng độ đo: `from ocr_metrics import evaluate` (`tools/ocr_metrics.py`), đọc nhãn từ `dataset/pres/vietocr/test_meta.tsv`.

---

## Tùy chỉnh

| Biến | Mặc định | Ý nghĩa |
|---|---|---|
| `DATA_ROOT` | tự tìm | Thư mục `1_vaipe` hoặc `public_train` |
| `MANIFEST` | `splits/pres_split_vaipe1.csv` | Bộ chia, chỉ dùng với `public_train` |
| `SEED` | `42` | Seed train |
| `TRAIN_YOLO` / `TRAIN_OCR` | `1` | `0` để bỏ qua model đó |
| `YOLO_MODEL` | `yolov8s.pt` | `yolov8n.pt`, `yolov8m.pt`, ... |
| `IMGSZ` | `1024` | Kích thước ảnh YOLO (E-B1 so 640 / 1024) |
| `EPOCHS` / `PATIENCE` | `50` / `15` | |
| `YOLO_BATCH` | `16` | Giảm nếu *out of memory* |
| `YOLO_DEVICE` | `0` | `0,1` để chạy 2 GPU |
| `KEEP_LAST` | `0` | `1` để lưu thêm `last.pt` |
| `OCR_ITERS` | `20000` | |
| `OCR_VALID_EVERY` | `2000` | Weights chỉ được lưu ở các lần val |
| `OCR_BATCH` | `32` | |
| `OCR_MAX_WIDTH` | `1024` | |
| `OCR_DEVICE` | `1` | `0` khi chạy session riêng |
| `OCR_CLASSES` | cả 6 lớp | VD `"drugname quantity usage"` |
| `OUT_DIR` | `/kaggle/working/output` | |

## Xử lý lỗi

| Lỗi | Cách sửa |
|---|---|
| `ERROR: no data found under /kaggle/input` | Chưa **Add Input** dataset `vaipepill2022` |
| `... does not match ...` | Dataset gắn vào không phải bản `tommyngx/vaipepill2022` gốc (danh sách file khác CSV) |
| Log không có `split by pres_split_vaipe1.csv` | Kaggle đang chạy code cũ: kiểm tra đã push lên `main` chưa |
| `ERROR: no GPU` | Chưa chọn **GPU T4 x2** |
| `CUDA out of memory` (YOLO) | `YOLO_BATCH=8` hoặc `YOLO_DEVICE=0,1` |
| `CUDA out of memory` (VietOCR) | `OCR_BATCH=16` |
