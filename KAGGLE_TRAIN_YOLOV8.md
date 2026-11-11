# Train YOLOv8s trên Kaggle (phát hiện vùng tên thuốc trên đơn)

Luồng này **độc lập** với luồng YOLOv7 + VietOCR trong `KAGGLE_TRAIN.md`:

| | Luồng cũ | Luồng này |
|---|---|---|
| Script | `scripts/kaggle_train_ocr.sh` | `scripts/kaggle_train_yolov8.sh` |
| Code train | `yolov7/train.py`, `vietocr/train_vietocr.py` | `yolov8/train.py` |
| Kết quả | `output/yolov7/`, `output/vietocr/` | `output/yolov8/yolov8s_ocr.pt` |

Cả hai luồng dùng chung bước chuẩn bị dữ liệu `tools/generate_train_detect_pres.py`, cùng nhãn và cùng tỉ lệ chia train/val 90/10.
Điểm khác: YOLOv8s **fine-tune từ weights pretrained COCO** thay vì train từ đầu.
Luồng này **không train VietOCR** vì bạn đã có weights VietOCR.

---

## Bước 0: Push code mới lên GitHub

```bash
git add yolov8/ scripts/kaggle_train_yolov8.sh KAGGLE_TRAIN_YOLOV8.md .gitignore
git commit -m "Add YOLOv8s training flow for Kaggle"
git push
```

## Bước 1: Tạo notebook

Có thể tạo notebook mới, hoặc dùng lại notebook cũ bằng cách sửa cell.

1. Gắn dataset `tommyngx/vaipepill2022` (**Add Input**).
2. **Session options**: Accelerator **GPU T4 x2**, Internet **On**.
3. Chỉ để lại 1 cell code (thay `<user>/<repo>`):
   ```python
   !rm -rf /tmp/repo && git clone --depth 1 https://github.com/<user>/<repo>.git /tmp/repo
   !bash /tmp/repo/scripts/kaggle_train_yolov8.sh
   ```

## Bước 2: Chạy thử nhanh (khoảng 10 phút)

```python
!EPOCHS=1 bash /tmp/repo/scripts/kaggle_train_yolov8.sh
```
Cuối log phải có `YOLOv8 exit code: 0` và file `yolov8s_ocr.pt`.
Chạy thử xong thì bỏ `EPOCHS=1` đi.

## Bước 3: Save Version để treo máy

**Save Version** → **Save & Run All (Commit)** → *Advanced Settings*: chọn Always save output, Accelerator **GPU T4 x2** → **Save**.

- Thời gian dự kiến khoảng **1–2 giờ** cho tối đa 100 epoch. Bước copy ảnh mất khoảng 5 phút.
- Train sẽ **tự dừng sớm** nếu sau 30 epoch liên tiếp mAP không tăng (tham số `PATIENCE`).
- Tab **Logs** hiện kết quả từng epoch, gồm loss và dòng `all ... mAP50 mAP50-95` của tập val.

## Bước 4: Lấy kết quả

```
output/
├── yolov8_results.zip          ← TẢI FILE NÀY: chứa toàn bộ thư mục yolov8/ và logs/ (~21 MB)
├── yolov8/
│   ├── yolov8s_ocr.pt          ← weights tốt nhất (~22 MB)
│   ├── results.csv / results.png   (loss, mAP theo epoch)
│   ├── BoxPR_curve.png, confusion_matrix.png, ...
│   └── args.yaml
└── logs/yolov8.log
```

**Cách tải:**
- **Trên web**: mở version → tab **Output** → mở thư mục `output` → rê chuột vào `yolov8_results.zip` → bấm biểu tượng tải bên phải.
  Đừng bấm nút tải cả thư mục, vì Kaggle phải nén lại toàn bộ trên server và nút đó hay bị treo.
- **Bằng CLI** (tin cậy nhất):
  ```bash
  kaggle kernels output <user>/<notebook-slug> -p ./kaggle_out
  unzip kaggle_out/output/yolov8_results.zip -d yolov8_results
  ```
- **Khi chạy tương tác** (không phải Save Version), thêm 1 cell để hiện link tải trực tiếp:
  ```python
  from IPython.display import FileLink
  FileLink('output/yolov8_results.zip')
  ```

File zip được tạo **kể cả khi train lỗi**, nên lúc đó bạn vẫn tải được log để xem nguyên nhân.

---

## Tùy chỉnh

Ví dụ: `!EPOCHS=150 YOLO_BATCH=32 bash /tmp/repo/scripts/kaggle_train_yolov8.sh`

| Biến | Mặc định | Ý nghĩa |
|---|---|---|
| `YOLO_MODEL` | `yolov8s.pt` | Có thể đổi sang `yolov8n.pt`, `yolov8m.pt`, ... |
| `EPOCHS` | `100` | Số epoch tối đa |
| `PATIENCE` | `30` | Dừng sớm sau N epoch không cải thiện |
| `YOLO_BATCH` | `16` | Có thể tăng lên `32` trên T4. Nếu bị *out of memory* thì giảm |
| `YOLO_DEVICE` | `0` | `0,1` để train trên cả 2 T4 (DDP) |
| `KEEP_LAST` | `0` | Đặt `1` để lưu thêm `last.pt` (dùng khi muốn train tiếp) |
| `DATA_ROOT` | tự tìm | Thư mục `public_train` |
| `OUT_DIR` | `/kaggle/working/output` | Nơi lưu kết quả |

## Xử lý lỗi

| Lỗi | Cách sửa |
|---|---|
| `ERROR: cannot find public_train/...` | Chưa gắn dataset (**Add Input**) |
| `ERROR: no GPU` | Chưa chọn **GPU T4 x2** |
| Lỗi mạng khi `pip install` hoặc tải `yolov8s.pt` | Chưa bật **Internet** |
| `CUDA out of memory` | Giảm `YOLO_BATCH` |
| Lỗi khác | Xem `output/logs/yolov8.log` (log đầy đủ) |

---

## Lưu ý: dùng weights YOLOv8 để suy luận

`run.sh` và `yolov7/detect_ocr.py` hiện chỉ đọc được weights **YOLOv7**.
Muốn dùng `yolov8s_ocr.pt` trong pipeline thì cần một bản `detect_ocr.py` dùng `ultralytics` với cùng định dạng output. Repo hiện chưa có file này.
