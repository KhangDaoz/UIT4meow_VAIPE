# Train YOLOv8s + VietOCR trên Kaggle (chia train/val/test cố định)

Luồng này train 2 model của nhánh đọc đơn thuốc **trên cùng một bộ chia train / val / test = 70 / 15 / 15**:

| Model | Nhiệm vụ | GPU | Kết quả |
|---|---|---|---|
| YOLOv8s | Phát hiện vùng *tên thuốc* trên đơn | T4 số 0 | `output/yolov8/yolov8s_ocr.pt` |
| VietOCR (`vgg_transformer`) | Đọc chữ tên thuốc | T4 số 1 | `output/vietocr/vietocr.pt` |

Hai model chạy **song song**. Cuối mỗi model đều được **chấm điểm trên tập test**.

Luồng này **độc lập** với luồng cũ trong `KAGGLE_TRAIN.md`. Luồng cũ giữ nguyên, không sửa file nào:

| | Luồng cũ | Luồng này |
|---|---|---|
| Script | `scripts/kaggle_train_ocr.sh` | `scripts/kaggle_train_yolov8.sh` |
| Detector | YOLOv7x, train từ đầu | YOLOv8s, fine-tune từ pretrained COCO |
| VietOCR | `vietocr/train_vietocr.py` | `vietocr/train_vietocr_split.py` |
| Chia dữ liệu | YOLO 90/10, OCR 80/20 (khác nhau, không có test) | **Chung 1 manifest 70/15/15** |

---

## Cách chia dữ liệu (tái lập được)

Tập `public_test` của cuộc thi không có nhãn, nên 1.173 đơn thuốc của `public_train` được chia thành:

| Tập | Số đơn | Dùng để |
|---|---|---|
| train | 821 | Train model |
| val | 176 | Chọn weights tốt nhất (YOLOv8: `best.pt`, VietOCR: lần val có độ chính xác cao nhất) và dừng sớm |
| test | 176 | **Chỉ chấm 1 lần ở cuối** → `test_metrics.json` |

Chia theo **đơn thuốc**, không theo từng dòng chữ hay từng box: mọi box và dòng chữ của cùng một đơn luôn nằm chung một tập.
Cả YOLOv8 và VietOCR dùng **cùng manifest**, nên không đơn nào trong test bị model nào thấy lúc train.

Bộ chia được cố định trong file **`splits/pres_split_seed42.json`** (đã commit vào repo):
- `seed`, `ratios`, `counts`, danh sách tên file của từng tập và `names_sha256` (mã băm danh sách 1.173 tên file gốc).
- Cách tạo: sắp xếp tên file → `random.Random(42).shuffle` → `n_val = round(n·0.15)`, `n_test = round(n·0.15)`, phần còn lại là train.
- Khi train, script **đọc manifest này** chứ không chia lại. Vì vậy bộ chia luôn giống hệt, kể cả khi code hay phiên bản Python thay đổi.
- Nếu dữ liệu không khớp manifest (thiếu hoặc thừa file), script **dừng và báo lỗi** chứ không âm thầm chia lại.
- Chạy lại tool với cùng seed và cùng dữ liệu cho ra đúng file manifest này (đã kiểm tra trên Python 3.11 và 3.14).

Seed khi train:
- **YOLOv8**: `seed=42`, `deterministic=True` của ultralytics. Khi chạy thử trên CPU, 2 lần train cho loss và mAP giống hệt nhau.
- **VietOCR**: đặt seed cho `random`, `numpy` và `torch` trước khi train.
- Trên GPU, kết quả có thể lệch rất nhỏ giữa các lần chạy, do một số phép tính của cuDNN không tất định.

Tạo bộ chia với seed khác, ví dụ `SEED=123` (seed 42 đang có sẵn manifest):
```bash
python tools/split_pres.py --label-dir <public_train>/prescription/label --seed 123
# -> splits/pres_split_seed123.json, commit file này rồi chạy với SEED=123
```
Nếu chạy `SEED=123` mà chưa có manifest, script sẽ tự tạo trên Kaggle và đưa vào file zip kết quả (`split_manifest.json`). Nên tải file đó về và commit vào thư mục `splits/` để lưu lại.

Kiểm tra một manifest có khớp với dữ liệu không:
```bash
python tools/split_pres.py --label-dir <public_train>/prescription/label --check splits/pres_split_seed42.json
```

---

## Bước 1: Tạo notebook

Có thể tạo notebook mới, hoặc dùng lại notebook cũ bằng cách sửa cell.

1. Gắn dataset `tommyngx/vaipepill2022` (**Add Input**).
2. **Session options**: Accelerator **GPU T4 x2**, Internet **On**.
3. Chỉ để lại 1 cell code:
   ```python
   !rm -rf /tmp/repo && git clone --depth 1 https://github.com/KhangDaoz/UIT4meow_VAIPE.git /tmp/repo
   !bash /tmp/repo/scripts/kaggle_train_yolov8.sh
   ```

## Bước 2: Chạy thử nhanh (khoảng 15–20 phút)

```python
!EPOCHS=1 OCR_ITERS=1000 OCR_VALID_EVERY=500 bash /tmp/repo/scripts/kaggle_train_yolov8.sh
```
Kiểm tra trong log:
- Có dòng `using split manifest .../pres_split_seed42.json`.
- Số ảnh `train: 821 / val: 176 / test: 176`, cùng số dòng chữ OCR của từng tập.
- Cuối log có 2 dòng `TEST RESULT: ...` (YOLOv8) và `TEST RESULT (VietOCR): ...`.
- Có `YOLOv8 exit code : 0` và `VietOCR exit code: 0`.

Chạy thử xong thì bỏ các biến ở đầu lệnh đi.

## Bước 3: Save Version để treo máy

**Save Version** → **Save & Run All (Commit)** → *Advanced Settings*: chọn Always save output, Accelerator **GPU T4 x2** → **Save**.

- Thời gian dự kiến khoảng **2–4 giờ**. Hai model chạy song song: YOLOv8s mất khoảng 1–2 giờ, VietOCR 20.000 iter mất khoảng 2–4 giờ.
- YOLOv8 **tự dừng sớm** nếu sau 30 epoch liên tiếp mAP trên val không tăng (`PATIENCE`).
- Cứ 5 phút, tab **Logs** in tiến độ của từng model:
  - YOLOv8: epoch hiện tại và dòng `val: all ... mAP50 mAP50-95`
  - VietOCR: dòng `iter ... train loss` hoặc `valid loss ... acc full seq`
- Cuối log có điểm test của 2 model. Đây là **các con số nên báo cáo**:
  ```
  TEST RESULT: precision=... recall=... mAP50=... mAP50-95=...
  TEST RESULT (VietOCR): lines=... acc_full_sequence=... acc_per_char=...
  ```

## Bước 4: Lấy kết quả

```
output/
├── results.zip                 ← TẢI FILE NÀY: chứa toàn bộ thư mục bên dưới (~170 MB)
├── split_manifest.json         ← bộ chia đã dùng
├── yolov8/
│   ├── yolov8s_ocr.pt          ← weights tốt nhất theo val (~22 MB)
│   ├── test_metrics.json       ← ĐIỂM TEST: precision, recall, mAP50, mAP50-95
│   ├── test/                   ← PR curve, confusion matrix trên tập test
│   ├── results.csv / results.png   (loss, mAP val theo epoch)
│   ├── BoxPR_curve.png, confusion_matrix.png, ...  (trên tập val)
│   ├── stats.json              ← số ảnh, số box của từng tập
│   └── args.yaml               ← toàn bộ tham số train (có seed)
├── vietocr/
│   ├── vietocr.pt              ← weights tốt nhất theo val (~150 MB)
│   ├── test_metrics.json       ← ĐIỂM TEST: acc_full_sequence, acc_per_char
│   ├── test_predictions.tsv    ← từng dòng test: ảnh, nhãn đúng, dự đoán, đúng/sai
│   ├── stats.json              ← số dòng chữ của từng tập
│   ├── config.yml, train.log
└── logs/
    ├── yolov8.log
    └── vietocr.log
```

- **`acc_full_sequence`**: tỉ lệ dòng đọc **đúng hoàn toàn**.
- **`acc_per_char`**: tỉ lệ ký tự đúng vị trí, tính trung bình theo dòng.

**Cách tải:**
- **Trên web**: mở version → tab **Output** → mở thư mục `output` → rê chuột vào `results.zip` → bấm biểu tượng tải bên phải.
  Nếu zip tải không được (vì `vietocr.pt` làm file nặng), hãy tải riêng từng file trong các thư mục `yolov8/` và `vietocr/`.
  Đừng bấm nút tải cả thư mục, vì Kaggle phải nén lại toàn bộ trên server và nút đó hay bị treo.
- **Bằng CLI** (tin cậy nhất):
  ```bash
  kaggle kernels output <user>/<notebook-slug> -p ./kaggle_out
  unzip kaggle_out/output/results.zip -d results
  ```
- **Khi chạy tương tác** (không phải Save Version), thêm 1 cell để hiện link tải trực tiếp:
  ```python
  from IPython.display import FileLink
  FileLink('output/results.zip')
  ```

File zip được tạo **kể cả khi train lỗi**, nên lúc đó bạn vẫn tải được log để xem nguyên nhân.

---

## Tùy chỉnh

Ví dụ chỉ train YOLOv8: `!TRAIN_OCR=0 bash /tmp/repo/scripts/kaggle_train_yolov8.sh`

| Biến | Mặc định | Ý nghĩa |
|---|---|---|
| `SEED` | `42` | Seed chia dữ liệu và seed train. Đọc file `splits/pres_split_seed<SEED>.json` |
| `TRAIN_YOLO` | `1` | `0` để bỏ qua YOLOv8 |
| `TRAIN_OCR` | `1` | `0` để bỏ qua VietOCR |
| `YOLO_MODEL` | `yolov8s.pt` | Có thể đổi sang `yolov8n.pt`, `yolov8m.pt`, ... |
| `EPOCHS` | `100` | Số epoch tối đa của YOLOv8 |
| `PATIENCE` | `30` | YOLOv8 dừng sớm sau N epoch không cải thiện |
| `YOLO_BATCH` | `16` | Có thể tăng lên `32` trên T4. Nếu bị *out of memory* thì giảm |
| `YOLO_DEVICE` | `0` | GPU cho YOLOv8 |
| `KEEP_LAST` | `0` | Đặt `1` để lưu thêm `last.pt` của YOLOv8 |
| `OCR_ITERS` | `20000` | Số iteration VietOCR. Dữ liệu ít nên có thể giảm còn khoảng 10.000 |
| `OCR_VALID_EVERY` | `3000` | Cứ N iter thì chạy val VietOCR một lần. Weights chỉ được lưu ở các lần val này |
| `OCR_BATCH` | `32` | Batch size VietOCR |
| `OCR_DEVICE` | `1` | GPU cho VietOCR |
| `DATA_ROOT` | tự tìm | Thư mục `public_train` |
| `OUT_DIR` | `/kaggle/working/output` | Nơi lưu kết quả |

## Xử lý lỗi

| Lỗi | Cách sửa |
|---|---|
| `ERROR: cannot find public_train/...` | Chưa gắn dataset (**Add Input**) |
| `ERROR: no GPU` | Chưa chọn **GPU T4 x2** |
| `... does not match the data` | Dữ liệu khác với lúc tạo manifest (sai dataset hoặc dataset đã đổi). Kiểm tra lại dataset đã gắn vào notebook |
| Lỗi mạng khi `pip install`, tải `yolov8s.pt` hoặc tải pretrained VietOCR từ `vocr.vn` | Chưa bật **Internet**, hoặc server `vocr.vn` tạm lỗi thì chạy lại sau |
| `CUDA out of memory` | Giảm `YOLO_BATCH` hoặc `OCR_BATCH` |
| `No weights were exported` (VietOCR) | Độ chính xác val không tăng lên trên 0, thường do `OCR_ITERS` quá nhỏ. Tăng `OCR_ITERS` |
| VietOCR lỗi `KeyError` với 1 ký tự lạ | Thêm ký tự đó vào dòng `vocab:` trong `vietocr/vgg_transformer.yml`, push lại |
| Lỗi khác | Xem `output/logs/yolov8.log` và `output/logs/vietocr.log` |

---

## Lưu ý: dùng weights YOLOv8 để suy luận

`run.sh` và `yolov7/detect_ocr.py` hiện chỉ đọc được weights **YOLOv7**.
Muốn dùng `yolov8s_ocr.pt` trong pipeline thì cần một bản `detect_ocr.py` dùng `ultralytics` với cùng định dạng output. Repo hiện chưa có file này.
`vietocr.pt` của luồng này có cùng định dạng với luồng cũ, nên `yolov7/detect_ocr.py --weights_ocr` đọc được.
