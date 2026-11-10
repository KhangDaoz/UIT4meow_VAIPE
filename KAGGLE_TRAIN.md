# Train YOLOv7x + VietOCR trên Kaggle (chế độ Save Version)

Hướng dẫn này train **2 model của nhánh đọc đơn thuốc**:

| Model | Nhiệm vụ | File kết quả |
|---|---|---|
| YOLOv7x | Phát hiện vùng *tên thuốc* trên ảnh đơn thuốc | `output/yolov7/yolov7_ocr.pt` |
| VietOCR (`vgg_transformer`) | Đọc chữ tên thuốc từ vùng đã cắt | `output/vietocr/vietocr.pt` |

Hai model chạy **song song trên 2 GPU T4**: YOLOv7x ở GPU 0, VietOCR ở GPU 1.
Toàn bộ quá trình nằm trong script `scripts/kaggle_train_ocr.sh`, gồm cài thư viện, chuẩn bị dữ liệu, train và gom kết quả.

> **Không dùng TPU.** Code viết cho GPU CUDA, TPU v5e-8 không chạy được.

---

## Bước 0: Chuẩn bị một lần

1. **Xác minh số điện thoại cho tài khoản Kaggle**: vào *Settings → Phone verification*.
   Chưa xác minh thì không bật được Internet và GPU trong notebook.
2. **Push repo lên GitHub** (từ máy của bạn):
   ```bash
   git add -A
   git commit -m "Kaggle training for YOLOv7 + VietOCR"
   git remote add origin https://github.com/<user>/<repo>.git
   git branch -M main
   git push -u origin main
   ```
   Repo nên để **Public** để Kaggle clone được mà không cần token.
   Nếu muốn để Private, xem [Phụ lục A](#phụ-lục-a-repo-private).

---

## Bước 1: Tạo notebook

1. Vào https://www.kaggle.com/datasets/tommyngx/vaipepill2022 → bấm **New Notebook**.
   Dataset sẽ được gắn sẵn vào notebook.
   *(Nếu tạo notebook từ trang chủ: ở cột phải chọn **Add Input** → tìm `vaipepill2022` → **Add**.)*
2. Ở cột phải, mục **Session options**:
   - **Accelerator**: chọn **GPU T4 x2**
   - **Internet**: bật **On**
   - **Persistence**: để mặc định
3. Xóa hết các cell mẫu, chỉ để lại **1 cell code** với nội dung sau
   (thay `<user>/<repo>` bằng repo của bạn):

   ```python
   !rm -rf /tmp/repo && git clone --depth 1 https://github.com/<user>/<repo>.git /tmp/repo
   !bash /tmp/repo/scripts/kaggle_train_ocr.sh
   ```

   - Repo được clone vào `/tmp/repo`. Kaggle **không lưu** thư mục này, nên output không bị lẫn hơn 800 file code và ảnh trung gian.
   - Kaggle chỉ lưu `/kaggle/working/output/`, nơi script copy weights và log vào.
   - Script tự tìm thư mục `public_train` trong `/kaggle/input`, bạn không cần sửa đường dẫn.

---

## Bước 2: Chạy thử nhanh (khuyến nghị, khoảng 15–25 phút)

Bước này giúp bắt lỗi môi trường **trước khi** treo máy vài tiếng.

1. Tạm sửa dòng thứ 2 của cell thành:
   ```python
   !EPOCHS=1 OCR_ITERS=1000 bash /tmp/repo/scripts/kaggle_train_ocr.sh
   ```
2. Bấm **Run All** (chạy tương tác).
3. Khi chạy xong, cuối log phải có:
   ```
   YOLOv7 exit code : 0
   VietOCR exit code: 0
   ```
   và danh sách file trong `/kaggle/working/output` phải có `yolov7/yolov7_ocr.pt` và `vietocr/vietocr.pt`.
4. Nếu có lỗi, xem [Xử lý lỗi](#xử-lý-lỗi).
5. Chạy xong thì **sửa dòng đó về như cũ** (bỏ `EPOCHS=1 OCR_ITERS=1000`).

---

## Bước 3: Save Version để treo máy

1. Bấm **Save Version** ở góc trên bên phải.
2. Chọn **Save & Run All (Commit)**.
3. Mở **Advanced Settings**:
   - **Output**: chọn *Always save output, even if the notebook fails*
     (nếu một model lỗi, vẫn giữ được kết quả của model còn lại)
   - **Accelerator**: đảm bảo là **GPU T4 x2**
4. Bấm **Save**.

Giờ bạn có thể **tắt máy hoặc đóng trình duyệt**, Kaggle sẽ tự chạy.

- Thời gian dự kiến khoảng **3–5 giờ**, tối đa cho phép là 12 giờ mỗi lần chạy.
- Mỗi lần chạy trừ vào hạn mức GPU theo tuần của bạn (xem trên trang *Settings → Quotas*).
- Theo dõi tiến độ: trang notebook → **Version** vừa tạo → tab **Logs**. Cứ mỗi 10 phút script in 1 dòng tiến độ của từng model.

---

## Bước 4: Lấy kết quả

Khi version chạy xong (trạng thái *Successful*), mở version đó → tab **Output**:

```
output/
├── yolov7/
│   ├── yolov7_ocr.pt      ← weights tốt nhất (đúng tên mà run.sh dùng)
│   ├── last.pt
│   ├── results.png, results.txt, confusion_matrix.png, ...   (biểu đồ mAP/loss)
├── vietocr/
│   ├── vietocr.pt         ← weights tốt nhất (đúng tên mà run.sh dùng)
│   ├── config.yml
│   └── train.log
└── logs/
    ├── yolov7.log
    └── vietocr.log
```

- **Tải về**: bấm **Download** trên tab Output.
- **Dùng tiếp trên Kaggle**: bấm **New Dataset** từ output. Sau đó gắn dataset này vào notebook inference qua **Add Input**.

Để chạy pipeline OCR (`run.sh`), copy 2 file vào thư mục `weights/` của repo:
```
weights/yolov7_ocr.pt
weights/vietocr.pt
```

---

## Tùy chỉnh

Đặt biến trước lệnh `bash`, ví dụ `!EPOCHS=80 YOLO_BATCH=8 bash /tmp/repo/scripts/kaggle_train_ocr.sh`

| Biến | Mặc định | Ý nghĩa |
|---|---|---|
| `EPOCHS` | `50` | Số epoch YOLOv7x |
| `YOLO_BATCH` | `8` | Batch size YOLOv7x. Giảm xuống `4` nếu bị *CUDA out of memory* |
| `OCR_ITERS` | `20000` | Số iteration VietOCR |
| `DATA_ROOT` | tự tìm | Đường dẫn tới thư mục `public_train` |
| `OUT_DIR` | `/kaggle/working/output` | Nơi lưu kết quả |

---

## Xử lý lỗi

Log chi tiết nằm ở `output/logs/yolov7.log` và `output/logs/vietocr.log`.

| Lỗi | Nguyên nhân / Cách sửa |
|---|---|
| `fatal: could not read Username` khi clone | Repo đang Private → chuyển Public hoặc làm theo Phụ lục A |
| `Could not resolve host` / pip lỗi mạng | Chưa bật **Internet** trong Session options |
| `ERROR: no GPU` | Chưa chọn Accelerator **GPU T4 x2** |
| `ERROR: cannot find public_train/prescription/label` | Chưa gắn dataset `vaipepill2022` vào notebook (**Add Input**) |
| `CUDA out of memory` trong `yolov7.log` | Thêm `YOLO_BATCH=4` |
| VietOCR lỗi khi tải `vgg_transformer.pth` từ `vocr.vn` | Server của VietOCR tạm lỗi, chạy lại sau |
| VietOCR lỗi `KeyError` với 1 ký tự lạ | Tên thuốc có ký tự không có trong `vocab` → thêm ký tự đó vào dòng `vocab:` trong `vietocr/vgg_transformer.yml`, push lại |
| Chạy quá 12 giờ bị dừng | Giảm `EPOCHS` hoặc `OCR_ITERS` |

---

## Phụ lục A: Repo private

1. Tạo GitHub token: *GitHub → Settings → Developer settings → Personal access tokens → Fine-grained*.
   Chỉ cấp quyền **Contents: Read-only** cho repo này.
2. Trong notebook Kaggle: **Add-ons → Secrets → Add Secret**, đặt tên `GITHUB_TOKEN`, dán token vào, rồi tick chọn để gắn secret vào notebook.
3. Thay cell bằng:
   ```python
   from kaggle_secrets import UserSecretsClient
   token = UserSecretsClient().get_secret("GITHUB_TOKEN")
   !rm -rf /tmp/repo && git clone --depth 1 https://{token}@github.com/<user>/<repo>.git /tmp/repo
   !bash /tmp/repo/scripts/kaggle_train_ocr.sh
   ```

---

## Phụ lục B: Script đã làm gì

1. Tìm dataset trong `/kaggle/input` và kiểm tra có GPU.
2. Cài `vietocr==0.3.13` bỏ qua các bản ghim cũ (`--no-deps`), rồi cài các thư viện nó cần.
   Nếu thư viện `imgaug` không tương thích với NumPy 2 thì được thay bằng một module rỗng; lúc train chỉ dùng augment của `albumentations`.
3. Chuẩn bị dữ liệu:
   - `tools/generate_train_detect_pres.py`: tạo nhãn YOLO cho vùng `drugname`, chia train/val theo tỉ lệ 90/10
   - `tools/crop_pres.py`: cắt từng dòng tên thuốc kèm text để train VietOCR, chia train/val theo tỉ lệ 80/20
4. Train song song:
   - YOLOv7x từ đầu, ảnh 640, `data/vaipe_pres.yaml`, `hyp.scratch.p5.yaml`
   - VietOCR fine-tune từ pretrained `vgg_transformer`
5. Copy weights tốt nhất, biểu đồ và log vào `/kaggle/working/output`.
