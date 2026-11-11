# PLAN v2 — Trích xuất thông tin đơn thuốc VAIPE bằng học sâu, đối chứng với xử lý ảnh cổ điển

> Tài liệu độc lập: đọc xong là bắt tay làm được, không cần đọc kế hoạch cũ (`team/KeHoach.md`).
> Ngày lập: 10/11/2026, sửa 11/11 theo chốt nhóm (00:40): deadline 11/11/2026 không bàn lùi, YOLOv8s + VietOCR fine-tune song song trên Kaggle T4, VLM qua FPT AI API.
> Thay thế phạm vi cũ (bộ 1–5, phiếu xét nghiệm, phân loại tài liệu). T0 = 00:40 11/11, nộp ≤23:59 — xem mục 8.

---

## 0. Tóm tắt một trang

| Hạng mục | Nội dung |
| :--- | :--- |
| **Bài toán** | Ảnh đơn thuốc → JSON: danh sách thuốc (tên, số lượng, cách dùng) + chẩn đoán + ngày |
| **Dữ liệu** | Chỉ bộ VAIPE (đơn thuốc BHYT in máy, sạch, mẫu đồng bộ). Split đã chốt: train 820 / valid 176 / test 176 |
| **Phương pháp chính** | YOLOv8 phát hiện + phân loại vùng chữ (6 lớp) → VietOCR **fine-tune** đọc chữ → luật ghép khoản thuốc + so khớp từ điển |
| **Đối chứng** | Cổ điển (hình thái học + chiếu, Tesseract, regex) · VLM zero-shot (Qwen2.5-VL qua FPT AI API) đầu-cuối |
| **Phần xử lý ảnh (môn học)** | Tiền xử lý cổ điển (ch. 2, 3, 5, 7) làm đối chứng + thí nghiệm **độ bền** trên ảnh test bị làm suy giảm |
| **Độ đo chính** | F1 khoản thuốc (tên + số lượng đúng) trên 176 ảnh test; phụ: F1@IoU0.5 phát hiện, CER nhận dạng |
| **Người** | Hiếu (A: tiền xử lý, độ bền, tích hợp, demo) · Trường (B: phát hiện) · Khang (C: nhận dạng) · Thắng (D: ghép khoản + VLM) |
| **Phần cứng** | 2 session Kaggle T4 song song (YOLO T4×2 + VietOCR T4×1); local RTX 3050 4GB dọn code/chạy pipeline; FPT AI API chạy VLM |
| **Cắt vì 1 ngày** | Bỏ E-B3, E-C2, E-D3 (thí nghiệm phụ). YOLO (2h30' nền) + VietOCR fine-tune 20k iter (4–6h nền) song song 2 session Kaggle |

**Lý do đổi hướng** (để viết vào báo cáo, mục Giới thiệu/Thảo luận): pipeline cũ chỉ dùng kỹ thuật cổ điển, đạt số liệu tốt khi
đo từng bước có nhãn chuẩn hỗ trợ, nhưng chạy đầu-cuối thì lỗi cộng dồn (phân loại sai 0,39 trên ảnh thật, gom dòng sụp
trên bảng kẻ cong, khoanh nhầm vùng thuốc). Hướng mới: thu hẹp về một loại tài liệu, dùng học sâu làm phương pháp chính để
sản phẩm dùng được, giữ kỹ thuật của môn ở vai trò **tiền xử lý và đối chứng có đo đạc**.

---

## 1. Đối chiếu yêu cầu đồ án (Lựa chọn 2 — Dự án thực tế)

| Yêu cầu của thầy | Plan đáp ứng ở đâu |
| :--- | :--- |
| Dùng ≥ 1 chương học phần, ghi rõ | Ch. 2 (lấy mẫu, nội suy), 3 (CLAHE, lọc trung vị/Gauss), 5 (Canny, chiếu, Otsu/Sauvola), 7 (giãn, đóng, mở) — mục 5 |
| Pipeline nhiều bước, có phương pháp đối chứng | 4 khối A–D, mỗi khối có đối chứng cổ điển; đầu-cuối có đối chứng VLM — mục 4, 5 |
| Đánh giá định lượng, độ đo chuẩn | F1@IoU, mAP50, CER/WER, F1 khoản thuốc — mục 6 |
| ≥ 50 ảnh đánh giá | 176 ảnh test VAIPE (+ 176 × 5 mức suy giảm) |
| Mỗi bước ghi tham số + lý do | Bảng tham số trong mục 5, ghi lại `results/<mã>/config.json` |
| Nộp: slide 10–15 (.pptx + .pdf), `code/` + `requirements.txt` + 5–10 ảnh `samples/`, README, báo cáo 4–10 trang (.pdf, có phân công + khai báo AI), video ≤ 3 phút (khuyến khích) | Mục 10 |

---

## 2. Dữ liệu

### 2.1 Vị trí và split (đã chốt, KHÔNG chia lại)

```
data/1_vaipe/
├── train/images/*.png  train/labels/*.json   # 820 ảnh
├── valid/images/       valid/labels/         # 176 ảnh — chọn tham số, early stopping
├── test/images/        test/labels/          # 176 ảnh — CHỈ dùng cho số liệu cuối
├── test_unlabeled/                           # 172 ảnh test gốc, không nhãn — chỉ demo định tính
└── split_manifest.csv                        # image_id, split, group_id, n_drugname, width, height
```

Split theo nhóm ảnh cùng bệnh nhân/lần khám (seed 42) nên không rò rỉ giữa train và test. Ảnh 760×1070, 770×1420 hoặc
1150×1600. Tải từ Drive nhóm, giải nén vào `data/` (không commit).

### 2.2 Định dạng nhãn

Mỗi ảnh một file JSON là **danh sách vùng chữ**:

```json
{"id": 23, "text": "1) RENAPRIL 5MG 5mg", "label": "drugname", "box": [46, 386, 294, 413], "mapping": 47}
{"id": 24, "text": "SL: 28 Viên",          "label": "quantity", "box": [595, 387, 708, 409]}
{"id": 26, "text": "Sáng 1 Viên",          "label": "usage",    "box": [289, 416, 376, 436]}
{"id": 21, "text": "Chấn đoán: T10 - Tăng huyết áp ...", "label": "diagnose", "box": [28, 272, 732, 294]}
{"id": 38, "text": "Ngày 03/07/2019",      "label": "date",     "box": [484, 633, 621, 652]}
```

- `box = [xmin, ymin, xmax, ymax]` pixel trên ảnh gốc. **Lưu ý:** khác quy ước `[x, y, w, h]` của code cũ.
- 6 lớp: `drugname`, `quantity`, `usage`, `diagnose`, `date`, `other`. `mapping` (chỉ có ở `drugname`) là mã thuốc nội bộ VAIPE.
- Vùng chữ ở mức **cụm/cột chữ**, không phải cả dòng: tên thuốc và "SL: 28 Viên" cùng hàng nhưng là 2 vùng.
- Thống kê train: 1.995 lượt `drugname`, **318 tên khác nhau**; tên có tiền tố số thứ tự `"1) "`.
  `quantity` phổ biến: `SL: 20 Viên` (649), `SL: 10 Viên` (434).
- Một số `id` bị khuyết (vùng thông tin cá nhân đã bị nguồn xóa/che) — không coi là lỗi.

### 2.3 Đáp án khoản thuốc (dẫn xuất, dùng cho độ đo đầu-cuối)

Script `tools/build_items_gt.py` sinh `data/1_vaipe/<split>/items_gt.json` từ nhãn theo **cùng luật ghép của khối D**
(mục 5.4) áp trên box + text chuẩn, rồi **soát bằng mắt 30 ảnh test** (ghi `results/gt_review.csv`). Mỗi ảnh:

```json
{"image_id": "VAIPE_P_TRAIN_1",
 "items": [{"name": "RENAPRIL 5MG 5mg", "quantity": 28, "unit": "Viên", "usage": ["Sáng 1 Viên"]}],
 "diagnose": "I10 - Tăng huyết áp ...", "date": "03/07/2019"}
```

### 2.4 Bộ test suy giảm (khối A tạo, Ngày 2)

176 ảnh test × 5 mức, sinh **một lần**, seed 42, lưu `data/1_vaipe/test_degraded/L{0..4}/` + `params.csv`.
Dùng lại `code/src/synthetic/degrade.py` (`rotate`, `perspective`, `gaussian_blur`, `motion_blur`, `shadow`, `jpeg`,
`gaussian_noise`).

| Mức | Biến đổi (cộng dồn ngẫu nhiên trong khoảng) |
| :---: | :--- |
| L0 | Ảnh gốc |
| L1 | Xoay ±1–2°, JPEG q 60–80 |
| L2 | L1 + Gauss blur σ 1–1,5, bóng nhẹ |
| L3 | Xoay ±2–5°, phối cảnh nhẹ, bóng vừa, nhiễu Gauss σ 5–10, JPEG q 40–60 |
| L4 | L3 + motion blur, bóng đậm, JPEG q 25–40 |

Ảnh suy giảm hình học không còn khớp box gốc → **chỉ đo độ đo đầu-cuối** (F1 khoản thuốc, không cần box).
Kiểm tra bằng mắt 10 ảnh L4: chữ vẫn đọc được bằng mắt người (nếu không, giảm mức).

---

## 3. Môi trường

```bash
cd code
python3.11 -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt            # lõi cũ: numpy, opencv, scikit-image, pandas, matplotlib, pytest, rapidfuzz, gradio
pip install -r requirements-dl.txt         # cho local: torch+cu121, ultralytics, vietocr, pytesseract, openai
pip install -e .
sudo apt install tesseract-ocr tesseract-ocr-vie   # đối chứng OCR (Windows: bản UB-Mannheim + gói vie)
```

`code/requirements-dl.txt` (Python 3.11; ghim phiên bản khi cài xong):

```
torch                # bản +cu121 cho RTX 3050 local; Kaggle có sẵn CUDA
ultralytics          # YOLOv8 (train ở Kaggle, infer local)
vietocr              # nhận dạng chữ tiếng Việt (pretrained dùng ngay; fine-tune chỉ nếu còn thời gian)
pytesseract
rapidfuzz             # so khớp chuỗi gần đúng
gradio               # demo
openai               # client OpenAI-compatible cho FPT AI VLM API
```

**Kaggle (Trường, Khang):** notebook `code/notebooks/{train_yolo,finetune_vietocr}.ipynb` đã setup sẵn
internet on + GPU T4/P100 + dataset VAIPE upload kèm. Train xong export weights về `code/weights/`
(không commit; Drive nhóm + `scripts/download_weights.sh`).

**VLM qua FPT AI (Thắng):** client OpenAI-compatible, dùng biến môi trường:

```
VLM_API_BASE=https://api.fpt.ai/morse/v1   # hoặc URL FPT AI broker cấp cho nhóm
VLM_API_KEY=<Thắng điền>
VLM_MODEL=qwen2.5-vl-32b                   # hoặc model FPT AI cung cấp
```

Nếu FPT AI không phục vụ Qwen-VL, fallback theo thứ tự: (a) FPT AI mô hình VLM hiện có của họ,
(b) OpenRouter `qwen/qwen2.5-vl-32b` (cần key openrouter.ai/keys). Code ghi hai case trong `assemble/vlm.py`.

---

## 4. Kiến trúc

```
                ┌──────────────────────────── Đối chứng đầu-cuối: VLM zero-shot (D) ─────────────────────────┐
ảnh ─► [A] Tiền xử lý ─► [B] Phát hiện vùng chữ ─► [C] Nhận dạng ─► [D] Ghép khoản + so khớp từ điển ─► JSON
       (tùy chọn,         YOLOv8 6 lớp               VietOCR            luật vị trí, rapidfuzz
        cổ điển)          | đối chứng: giãn +        | đối chứng:       | phân lớp: theo YOLO hoặc regex
                          |   chiếu (ch. 5, 7)       |   Tesseract vie  |   trên chữ (đối chứng)
```

### 4.1 Cấu trúc thư mục code mới

Giữ `code/src/common/` (`io`, `paths`, `seed`, `experiment`) và `code/src/ocr/metrics.py`. Các module khác của
pipeline cũ chuyển vào `code/legacy/` (không xóa — dùng lại hàm làm đối chứng).

```
code/
├── src/
│   ├── common/          # giữ nguyên; thêm common/boxes.py (xyxy↔xywh, IoU, ghép cặp tham lam)
│   ├── vaipe/           # Hiếu — đọc nhãn, items_gt, chuyển định dạng YOLO / VietOCR
│   ├── enhance/         # Hiếu — khối A: deskew, clahe, denoise, shadow, pipeline
│   ├── detect/          # Trường — khối B: yolo.py, classical.py
│   ├── recognize/       # Khang — khối C: vietocr_engine.py, tesseract_engine.py, crop.py
│   ├── assemble/        # Thắng — khối D: classify_rules.py, group.py, lexicon.py, vlm.py
│   ├── metrics/         # dùng chung: detection.py, ocr.py (= ocr/metrics.py cũ), items.py
│   └── pipeline.py      # Hiếu — ghép A→B→C→D, chọn biến thể bằng tham số
├── experiments/         # E-A*, E-B*, E-C*, E-D*, E-X* (mỗi thí nghiệm 1 file, ghi results/<mã>/)
├── notebooks/           # Colab
├── tools/               # build_items_gt.py, make_degraded.py, to_yolo.py, to_vietocr.py
├── tests/
├── samples/             # 8 ảnh: 6 VAIPE test + 2 ảnh L3/L4
├── run_pipeline.py      # CLI
└── app.py               # Gradio demo
```

### 4.2 Hợp đồng dữ liệu giữa các khối (`src/common/contracts.py`, viết lại Ngày 1)

```python
@dataclass
class TextBox:            # đầu ra B, đầu vào C
    box: list[int]        # [x0, y0, x1, y1] — CÙNG quy ước nhãn VAIPE
    label: str | None     # một trong 6 lớp, hoặc None nếu bộ phát hiện không phân lớp (đối chứng cổ điển)
    score: float
    text: str = ""        # C điền
    text_conf: float = 0.0

@dataclass
class Item:               # đầu ra D
    name: str             # đã bỏ tiền tố "1) "
    name_raw: str         # chuỗi OCR gốc
    quantity: int | None
    unit: str | None
    usage: list[str]
    lexicon_match: str | None   # tên chuẩn trong từ điển, None nếu không khớp
    box: list[int]

@dataclass
class PrescriptionResult:
    image_id: str
    items: list[Item]
    diagnose: str | None
    date: str | None
    boxes: list[TextBox]  # để vẽ overlay
    timing: dict          # giây cho từng khối
```

Quy ước: ảnh BGR `uint8`; đọc/ghi qua `common.io`; seed 42; mọi thí nghiệm ghi `results/<mã>/{config.json, table.csv}`.

---

## 5. Từng khối: việc cần làm, tham số, thí nghiệm

### 5.1 Khối A — Tiền xử lý và độ bền (Hiếu, kiêm tích hợp)

**Việc:**
1. `vaipe/`: đọc nhãn, `tools/build_items_gt.py` (dùng `assemble.group` của Thắng — phối hợp Ngày 2), `tools/make_degraded.py`.
2. `enhance/`, mỗi bước là một hàm `np.ndarray -> np.ndarray` bật/tắt được:

| Bước | Kỹ thuật | Chương | Tham số khởi đầu | Lấy từ code cũ |
| :--- | :--- | :---: | :--- | :--- |
| A1 Nắn phối cảnh | Canny → contour 4 góc → `warpPerspective`; không tìm được thì bỏ qua | 5 | Canny 50/150 | `legacy/preprocessing/corners.py`, `rectify.py` |
| A2 Chỉnh nghiêng | Chiếu ngang, tối đa phương sai P(y) theo góc | 5 | ±10°, bước 0,25° | `legacy/preprocessing/deskew.py::skew_projection` |
| A3 Khử bóng | Ước lượng nền bằng phép đóng, chia nền | 7 | SE 81 (≈ 4× chiều cao chữ) | `legacy/cleaning/background.py` |
| A4 Khử nhiễu | Lọc trung vị / song phương | 3 | k = 3 | — |
| A5 Tăng tương phản | CLAHE | 3 | clip 2,0, lưới 8×8 | `legacy/preprocessing/enhance.py` |

**Thí nghiệm:**

| Mã | Câu hỏi | Cách làm | Độ đo | Tiêu chí xong |
| :--- | :--- | :--- | :--- | :--- |
| **E-A1** | Tiền xử lý cổ điển có giúp pipeline DL khi ảnh xấu đi? | Pipeline tốt nhất (B=YOLO, C=VietOCR, D=luật) trên L0–L4, có/không A1–A5 (và từng bước bật riêng trên L3) | F1 khoản thuốc theo mức | Biểu đồ đường 5 mức × 2–3 biến thể |
| **E-A2** | Chỉnh nghiêng đúng không? | Xoay ảnh test ±0–10° biết trước, ước lượng góc | Sai số góc tuyệt đối trung vị | Bảng + 1 hình |
| **E-A3** | Thời gian chạy | 50 ảnh, CPU máy cá nhân, từng khối | giây/ảnh | Bảng |

**Kết quả mong đợi (giả thuyết, kiểm chứng chứ không ép):** ở L0 tiền xử lý không giúp hoặc hại nhẹ; ở L3–L4 chỉnh
nghiêng + khử bóng giúp rõ. Kết quả ngược lại vẫn báo cáo trung thực.

### 5.2 Khối B — Phát hiện + phân loại vùng chữ (Trường)

**Phương pháp chính — YOLOv8:**
- `tools/to_yolo.py`: nhãn → định dạng YOLO (`class cx cy w h` chuẩn hóa), 6 lớp theo thứ tự
  `[drugname, quantity, usage, diagnose, date, other]`; `data.yaml` trỏ train/valid.
- Huấn luyện trên Colab T4:
  `yolo detect train model=yolov8s.pt data=vaipe.yaml imgsz=1024 epochs=50 batch=16 patience=15 seed=42 fliplr=0 mosaic=0.5`
  - `fliplr=0`: lật ngang làm hỏng chữ. `imgsz=1024`, `batch=16`: T4×2 32GB dư VRAM. Chữ VAIPE cao 15–25 px, 1024 giữ đủ vùng nhỏ.
  - Kaggle T4×2 ≈ 2h30'. Xuất `weights/yolo_vaipe.pt`. Ngưỡng `conf` chọn trên valid (tối đa F1), mặc định 0,25.

**Đối chứng cổ điển** (`detect/classical.py`, không phân lớp → `label=None`):
1. Xám → Sauvola (cửa sổ 31, k 0,2) (ch. 5).
2. Ước lượng chiều cao chữ h = trung vị chiều cao thành phần liên thông.
3. Giãn với SE (0,5h × 1,5h) nối ký tự thành cụm, nhưng không nối qua khoảng trống giữa các cột ("1) RENAPRIL" và "SL: 28" cách xa) (ch. 7).
4. Thành phần liên thông → box; lọc theo diện tích/tỉ lệ (hậu xử lý).
   Dùng lại `legacy/table_extraction/geometry.py::words_from_binary`; chọn hệ số SE trên valid.

**Thí nghiệm:**

| Mã | Câu hỏi | So sánh | Độ đo | Tiêu chí xong |
| :--- | :--- | :--- | :--- | :--- |
| **E-B1** | DL hơn cổ điển bao nhiêu ở việc tìm vùng chữ? | Cổ điển vs YOLO (bỏ nhãn lớp) vs YOLO 640/1024 | P/R/F1 @IoU 0,5 không phân lớp, trên test | Bảng 4 dòng |
| **E-B2** | YOLO phân lớp đúng không? | YOLO 6 lớp | P/R/F1 từng lớp @IoU 0,5, mAP50, ma trận nhầm lẫn | Bảng + ma trận |
| **E-B3** | Cần bao nhiêu dữ liệu? | YOLO train trên 10 / 25 / 50 / 100 % train | F1 `drugname` | Đường cong (bỏ trước nếu thiếu thời gian) |

Ghép cặp box: tham lam theo IoU giảm dần, mỗi box GT ghép tối đa 1 box dự đoán (`metrics/detection.py`).

### 5.3 Khối C — Nhận dạng chữ (Khang)

**Dữ liệu:** `tools/to_vietocr.py` cắt mọi vùng chữ có nhãn (đệm 2 px) → `data/1_vaipe/crops/{train,valid,test}/` +
file nhãn `ảnh<TAB>text` theo định dạng VietOCR. Khoảng 30 nghìn vùng train (cả lớp `other`).

**Các phương án** (đánh giá trên vùng cắt theo **box chuẩn** của test để cô lập khối C):

| Phương án | Chi tiết |
| :--- | :--- |
| C0 Tesseract `vie` | `--psm 7`, vùng cắt phóng ×2 nội suy bậc ba (ch. 2). Đối chứng cổ điển |
| C1 Tesseract `vie` + nhị phân | Như C0 nhưng đầu vào Otsu / Sauvola (ch. 5) — kiểm tra lại kết quả cũ "ảnh xám tốt hơn nhị phân" |
| C2 VietOCR pretrained | `vgg_transformer` pretrained, không huấn luyện — đối chứng |
| **C3 VietOCR fine-tune (PHƯƠNG ÁN CHÍNH)** | Từ C2, train trên crops train + valid; **Kaggle T4 session 2 song song YOLO**, ~20k iter batch 32, ≈ 4–6 h nền. Dùng cho pipeline chính |
| C4 (đã cắt) | `vgg_seq2seq` fine-tune — bỏ |

**Thí nghiệm:**

| Mã | Câu hỏi | Độ đo | Tiêu chí xong |
| :--- | :--- | :--- | :--- |
| **E-C1** | Bộ OCR nào tốt nhất trên vùng chữ chuẩn? | CER, WER toàn tập (`corpus_cer`), **CER riêng lớp drugname, quantity**, thời gian/vùng | Bảng 3 dòng (C0, C1, C2) |
| **E-C2** | Tiền xử lý vùng cắt (xám/nhị phân/CLAHE) ảnh hưởng thế nào? | CER của C0 và C2 với 3 kiểu đầu vào | Bảng nhỏ |
| ~~E-C3~~ | ~~Phân tích lỗi~~ | Cắt vì 1 ngày | — |

Chuẩn hóa trước khi tính CER: Unicode NFC, gộp khoảng trắng (đã có trong `ocr/metrics.py::_norm`). Phân biệt hoa/thường.

### 5.4 Khối D — Ghép khoản thuốc, so khớp từ điển, đối chứng VLM (Thắng)

**D1 Phân lớp vùng chữ bằng luật** (`assemble/classify_rules.py`), dùng khi B là cổ điển (không có nhãn lớp):

| Lớp | Luật trên chữ OCR (không phân biệt hoa/thường, bỏ dấu khi so) |
| :--- | :--- |
| drugname | `^\s*\d+\s*\)` |
| quantity | `^SL\s*:` |
| usage | bắt đầu bằng `Sáng|Trưa|Chiều|Tối|Uống|Ngày uống` và chứa số |
| diagnose | `^Ch[aẩ]n\s*đo[aá]n` (dòng kế tiếp nối vào nếu thẳng cột trái và sát dưới) |
| date | `^Ngày\s*\d` |
| other | còn lại |

**D2 Ghép khoản** (`assemble/group.py`) — dùng cho cả pipeline lẫn sinh `items_gt`:
1. Sắp `drugname` theo y. Mỗi `drugname` mở một khoản; khoảng dọc của khoản = từ đỉnh box này đến đỉnh `drugname` kế tiếp
   (khoản cuối: đến box `date` hoặc đáy ảnh).
2. `quantity` của khoản = box `quantity` chồng theo y nhiều nhất với `drugname` (cùng hàng), nằm bên phải.
3. `usage` = mọi box `usage` có tâm y nằm trong khoảng dọc của khoản, đọc trái → phải, trên → dưới.
4. Tách trường: `name` = bỏ tiền tố `^\d+\)\s*`; `quantity` = số nguyên trong `SL: (\d+)`; `unit` = từ sau số.

**D3 So khớp từ điển** (`assemble/lexicon.py`): từ điển = tập tên thuốc (đã bỏ tiền tố) trong **train**.
`rapidfuzz.process.extractOne(name, lexicon, scorer=ratio)`; điểm ≥ ngưỡng τ (chọn trên valid, khởi đầu 90) thì thay
bằng tên chuẩn. Báo cáo **tỉ lệ tên thuốc test không có trong từ điển (OOV)** — nếu cao, so khớp chỉ giúp phần trong từ điển.

**D4 Đối chứng VLM zero-shot** (`assemble/vlm.py`, qua FPT AI API):
- Gọi endpoint OpenAI-compatible của FPT AI (`VLM_API_BASE`, `VLM_API_KEY`, `VLM_MODEL` từ biến môi trường).
  Mô hình mặc định `qwen2.5-vl-32b`; nếu FPT AI không phục vụ Qwen-VL, đổi `VLM_MODEL` sang model FPT AI có (hoặc fallback OpenRouter).
  Ảnh base64 kèm prompt (OpenAI vision API format) — **ảnh VAIPE đã che thông tin cá nhân từ nguồn** nên gửi API được.
- Prompt cố định (lưu `results/E-D2/prompt.txt`): yêu cầu trả JSON đúng schema `items[{name, quantity, unit, usage[]}], diagnose, date`;
  `temperature=0`. JSON hỏng → thử parse lại một lần, vẫn hỏng thì tính là đơn rỗng.
- Tốc độ ước tính: 176 ảnh test × ~3 s/ảnh ≈ 9 phút; chạy song song 4 luồng = ~3 phút.

**Thí nghiệm:**

| Mã | Câu hỏi | So sánh | Độ đo | Tiêu chí xong |
| :--- | :--- | :--- | :--- | :--- |
| **E-D1** | Luật ghép đúng không khi đầu vào hoàn hảo? | D2 trên box + text chuẩn vs `items_gt` đã soát | F1 khoản (kỳ vọng ≈ 1,0 — kiểm tra luật) | Con số + danh sách ca sai |
| **E-D2** | Pipeline vs VLM đầu-cuối | (i) Cổ điển: B-cổ điển + C0 + D1 + D2 · (ii) Lai: YOLO + C0 + D2 · **(iii) DL: YOLO + C3(fine-tune) + D2** (pipeline chính) · (iv) DL-C2-fallback (nếu C3 trễ) · (v) VLM | F1 khoản, F1 tên thuốc, đúng số lượng %, tỉ lệ đơn đúng hoàn toàn, giây/ảnh | **Bảng chính của báo cáo** |
| ~~E-D3~~ | ~~So khớp từ điển~~ | Cắt vì 1 ngày | — | — |

### 5.5 Thí nghiệm liên khối

| Mã | Người | Nội dung |
| :--- | :--- | :--- |
| **E-X1** Ablation | Hiếu | Thay từng khối của pipeline DL bằng đáp án chuẩn (B chuẩn, C chuẩn) để biết lỗi đến từ khối nào: F1 khoản khi (B chuẩn, C chuẩn) / (B chuẩn, C dự đoán) / (B dự đoán, C chuẩn) |
| **E-X2** Độ bền (co lại) | Hiếu | = E-A1, chỉ 3 mức L0/L2/L4 (bỏ L1/L3) để kịp thời gian |

---

## 6. Độ đo (định nghĩa chung, code ở `src/metrics/`)

| Độ đo | Định nghĩa |
| :--- | :--- |
| **IoU** | diện tích giao / diện tích hợp của 2 box `xyxy` |
| **P/R/F1 phát hiện** | Ghép tham lam IoU ≥ 0,5 (cùng lớp nếu đo có lớp). TP = cặp ghép được; P = TP/#dự đoán; R = TP/#GT. Tính trên **toàn tập** (cộng dồn TP/FP/FN), không lấy trung bình từng ảnh |
| **mAP50** | Theo `ultralytics` (`model.val`) — báo cáo kèm, không dùng để chọn mô hình |
| **CER / WER** | Levenshtein ký tự / từ chia độ dài đáp án, sau chuẩn hóa NFC + gộp khoảng trắng; toàn tập = tổng lỗi / tổng ký tự |
| **F1 khoản thuốc** (chính) | Khoản dự đoán khớp khoản GT khi **CER(tên) ≤ 0,1 và quantity bằng nhau**. Ghép tham lam theo CER tên tăng dần. Cộng dồn TP/FP/FN cả tập |
| F1 tên thuốc | Như trên nhưng chỉ xét tên |
| Đơn đúng hoàn toàn | % ảnh mà mọi khoản đều khớp và không có khoản thừa |
| Thời gian | giây/ảnh, CPU máy cá nhân (ghi cấu hình CPU) — VLM ghi riêng GPU Colab |

Khoảng tin cậy: với bảng chính E-D2, bootstrap 1.000 lần theo ảnh → báo cáo F1 kèm khoảng 95 % (`metrics/items.py::bootstrap_ci`).

**Kỷ luật đánh giá:** mọi lựa chọn (ngưỡng, τ, kích thước SE, `conf`, epoch) làm trên **valid**. Test chạy **một lần**
khi đã chốt cấu hình (Ngày 10). Ghi lại commit hash trong `config.json`.

---

## 7. Phân công chi tiết theo subtask

> Ngôn ngữ tự nhiên, không liệt tên file. Bắt đầu từ 00:40 11/11, nộp ≤23:59.
> **AI** = agent viết code skeleton, **HV** = người làm tay (đẩy data, train, chạy số liệu, soát, viết §4/§5), **TT** = chạy nền.
> Mỗi người ghi trạng thái vào `team/TienDo.md` khi xong subtask.

### 7.1. Hiếu — khối A + tích hợp + nộp (~8h10')

| Việc | Ước tính | Chờ | Loại |
| :--- | :---: | :--- | :---: |
| Dọn repo các module cũ sang `legacy/`, tạo thư mục mới | 10' | — | AI |
| Viết hợp đồng dữ liệu mới (box, contract, đường dẫn) | 15' | repo | AI |
| Viết bộ đọc nhãn VAIPE 6 lớp cho cả nhóm | 20' | hợp đồng | AI |
| Sinh nhãn YOLO + crops VietOCR từ nhãn gốc | 20' | bộ đọc | AI |
| Sinh `items_gt`+ Thắng soát 10 ảnh test | 25' | luật ghép (D) | AI+HV |
| Sinh ảnh test suy biến 3 mức L0/L2/L4 | 15' | bộ đọc | AI |
| Viết khối A: nắn nghiêng, CLAHE, khử nhiễu, khử bóng | 20' | repo | AI |
| Viết bộ đo: IoU, CER, F1 khoản, bootstrap CI | 25' | hợp đồng | AI |
| Ghép pipeline A→B→C→D và CLI chạy | 25' | A,B,C,D | AI |
| App Gradio demo tải ảnh xem JSON+overlay | 25' | pipeline | AI |
| README mới + requirements-dl + script tải weights | 15' | — | AI |
| Viết test tối thiểu (5 file) | 25' | hợp đồng + D | AI |
| Chạy thí nghiệm ablation trên 176 ảnh test | 30' | B,C,D xong | HV |
| Chạy thí nghiệm độ bền 3 mức suy biến | 30' | ảnh suy biến, B,C,D | HV |
| Viết báo cáo LaTeX 4–10 trang | 1h15' | có số liệu | HV+AI |
| Slide 12 trang | 45' | báo cáo | HV+AI |
| Quay video demo ≤3 phút | 30' | app xong | HV |
| pytest xanh + đóng gói zip + nộp email | 25' | mọi thứ | HV |

### 7.2. Trường — khối B YOLO (~4h45')

| Việc | Ước tính | Chờ | Loại |
| :--- | :---: | :---: | :---: |
| Đẩy VAIPE lên Kaggle Datasets (~700MB) | 20' | — | HV |
| Sinh nhãn YOLO + `data.yaml` | 10' | nhãn YOLO | AI |
| Viết notebook train YOLOv8s T4×2 | 15' | Kaggle | AI |
| **Train YOLOv8s 50 epoch** | 2h30' | notebook | **TT** |
| Viết khối B YOLO: load weights, infer | 25' | train xong | AI |
| Viết khối B đối chứng cổ điển (dilation+chiếu) | 25' | A xong | AI |
| Chạy E-B1: classical vs YOLO trên test | 20' | weights | HV |
| Chạy E-B2: F1 từng lớp + mAP50 + confusion | 15' | weights | HV |
| Viết báo cáo phần B + 3 slide | 40' | E-B xong | HV |

### 7.3. Khang — khối C OCR (~3h25' trực tiếp + 4–6h train nền)

| Việc | Ước tính | Chờ | Loại |
| :--- | :---: | :---: | :---: |
| Cài Tesseract 5 + gói `vie` | 15' | — | HV |
| Cài VietOCR pretrained `vgg_transformer` | 15' | — | HV |
| Sinh crops cho OCR (train+valid+test) | 10' | bộ crops | AI |
| Viết khối C: cắt vùng + Tesseract (C0/C1) | 35' | — | AI |
| Viết khối C: VietOCR engine (C2 pretrained + C3 fine-tune) | 25' | env | AI |
| Viết notebook fine-tune VietOCR Kaggle T4 | 15' | crops | AI |
| **Mở Kaggle session 2, chạy fine-tune VietOCR 20k iter** | **4–6h** | notebook | **TT** song song YOLO |
| Chạy E-C1: so 4 engine (C0/C1/C2/C3) trên test crops GT | 25' | C3 xong | HV |
| Viết báo cáo phần C + 3 slide | 40' | E-C xong | HV |

### 7.4. Thắng — khối D luật + VLM API (~4h35')

| Việc | Ước tính | Chờ | Loại |
| :--- | :--- | :---: | :---: |
| Viết khối D luật regex gán 6 lớp trên text | 25' | — | AI |
| Viết luật gom khoản (drugname+quantity+usage) | 35' | bộ đọc | AI |
| Viết so khớp tên thuốc với từ điển train | 20' | bộ đọc | AI |
| Viết client VLM gọi FPT AI API | 40' | — | AI |
| Soát 10 ảnh `items_gt` test | 20' | items_gt | HV |
| Thử VLM 10 ảnh valid, chốt prompt | 15' | VLM client | HV |
| Chạy E-D1: luật trên box chuẩn | 10' | luật ghép | HV |
| Chạy E-D2: 4 pipeline + bootstrap CI 176 test | 40' | B,C,D xong | HV |
| Chạy VLM 176 test (8 luồng) | 10' | VLM client | **TT** |
| Viết báo cáo phần D + Thảo luận + 3 slide | 1h | E-D xong | HV |

### 7.5. Tóm tắt từng người

| Người | Tổng trực tiếp | Bottleneck | Chạy nền |
| :--- | :---: | :--- | :---: |
| Hiếu | ~8h10' | báo cáo cuối | — |
| Trường | ~4h45' | train YOLO 2h30' | ✅ YOLO |
| Khang | ~3h25' (+4–6h nền) | E-C1 chờ crops + C3 fine-tune | ✅ VietOCR fine-tune |
| Thắng | ~4h35' | E-D2 chờ B+C xong | ✅ VLM 176 test |

### 7.6. Phụ thuộc và chốt mốc

- B và C độc lập với nhau (C dùng box chuẩn, không cần B).
- D luật ghép dùng box + text chuẩn được ngay, không cần YOLO.
- E-D2 (bảng chính) cần B + C + D xong; E-X1/E-A1 cũng vậy.
- Đổi hợp đồng dữ liệu (`contracts`) phải báo cả nhóm và sửa test trong cùng commit.

| Mốc | Ước tính từ T0 | Số liệu trên tay |
| :--- | :---: | :--- |
| YOLO weights xong | ~3h15' | — |
| Bảng chính E-D2 xong | ~5h40' | — |
| Báo cáo đầy đủ | ~7h | — |
| Nộp | ~7h20' | dư ~16h buffer tới 23:59 |

---

## 8. Lịch (1 ngày — Hôm nay 11/11/2026)

> Nộp đầy đủ v2 **trong hôm nay** không bàn lùi. Phân công theo khối, song song tối đa.
> Ai xong phần mình sớm thì hỗ trợ đứa khác. Cắt theo bảng mục 8.5 nếu trễ.

### 8.1. Bảng thời gian công việc (kèm AI hỗ trợ viết code)

> **AI** = opencode/agent viết code, **HV** = người thực hiện bằng tay (đẩy data, train, chạy số liệu, review),
> **TT** = chạy nền không cần chờ (train / API call).
> Nhóm 4 người: Hiếu (A+ tích hợp) · Trường (B) · Khang (C) · Thắng (D).
> YOLO model cuối chốt: **YOLOv8s** trên Kaggle **T4 ×2** (batch 16, imgsz 640, 50 epoch ≈ 2,5h).
> Khai báo: với AI hỗ trợ, AI dùng để viết code skeleton + thí nghiệm, *không thay người ra quyết định/thực thi/train/chấm số liệu* — viết rõ trong §7 Khai báo AI của báo cáo.

#### Giai đoạn 1 — Dọn repo + data + notebook (~1h, song song)

| # | Việc | Người | Loại | Thời gian |
| :-- | :--- | :---: | :---: | :---: |
| 1.1 | Dọn v1 → `code/legacy/`, tạo `src/{vaipe,enhance,detect,recognize,assemble,metrics}/` | AI | AI | 10' |
| 1.2 | `common/boxes.py` (xyxy↔xywh, IoU, ghép cặp) + viết lại `contracts.py` + sửa `paths.py` | AI | AI | 15' |
| 1.3 | `vaipe/dataset.py` (đọc nhãn VAIPE 6 lớp, lists boxes/text theo ảnh) | AI | AI | 20' |
| 1.4 | `tools/to_yolo.py` + `tools/to_vietocr.py` (chuyển nhãn, sinh crops) | AI | AI | 20' |
| 1.5 | `tools/build_items_gt.py` chạy + Thắng soát 10 ảnh test | Thắng | HV | 25' |
| 1.6 | Đẩy dataset VAIPE lên Kaggle Datasets (~700 MB) | Trường | HV | 20' |
| 1.7 | Cài venv + `requirements-dl.txt` + thử `torch.cuda.is_available()` trên RTX 3050 | Cả nhóm | HV | 15' |

→ max-track ~50–60'.

#### Giai đoạn 2 — Code logic tất cả khối (~3h, AI liên tục, song song Kaggle train)

| # | Việc | Người | Loại | Thời gian |
| :-- | :--- | :---: | :---: | :---: |
| 2.1 | `enhance/{deskew,clahe,denoise,shadow,pipeline}.py` (wrap legacy) | AI | AI | 20' |
| 2.2 | `detect/classical.py` (wrap legacy) + `detect/yolo.py` (load + infer) | AI | AI | 25' |
| 2.3 | `recognize/{crop, tesseract_engine, vietocr_engine}.py` | AI | AI | 30' |
| 2.4 | `assemble/{classify_rules, group, lexicon, vlm}.py` (`vlm.py` = client OpenAI-compatible cho FPT AI) | AI | AI | 40' |
| 2.5 | `metrics/{detection, ocr, items}.py` (F1 khoản + `bootstrap_ci`) | AI | AI | 25' |
| 2.6 | `pipeline.py` + `run_pipeline.py` + `app.py` Gradio + README mới + `requirements-dl.txt` | AI | AI | 30' |
| 2.7 | `tools/make_degraded.py` (sinh L0/L2/L4) + `notebooks/{train_yolo, vlm_eval}.ipynb` | AI | AI | 20' |
| 2.8 | Push `train_yolo.ipynb` lên Kaggle, bật GPU T4 ×2, chạy | Trường | HV | 10' push + **TT 2,5h** |
| 2.9 | VietOCR pretrained C2 `.predict()` trên 10 crops valid | Khang | HV | 10' |

→ max-track ~3h (song song Kaggle train 2,5h nền).

#### Giai đoạn 3 — Chạy số liệu (~2h sau khi YOLO xong, mục tiêu xong ~17:30–18:00)

| # | Việc | Người | Loại | Thời gian |
| :-- | :--- | :---: | :---: | :---: |
| 3.1 | Tải `weights/yolo_vaipe.pt` về local | Hiếu | HV | 5' |
| 3.2 | `run_pipeline.py samples/` kiểm tra đầu-cuối | Hiếu | HV | 10' |
| 3.3 | E-X1 ablation 3 dòng trên 176 ảnh test | Hiếu | HV | 30' |
| 3.4 | E-A1 độ bền 3 mức L0/L2/L4 × 176 | Hiếu | HV | 30' |
| 3.5 | E-B1, E-B2 trên test | Trường | HV | 20' |
| 3.6 | E-C1 (C0/C1/C2) trên test crops GT | Khang | HV | 25' |
| 3.7 | E-D1 luật D2 trên box+text chuẩn | Thắng | HV | 10' |
| 3.8 | **E-D2** (4 pipeline + bootstrap CI) trên 176 test | Thắng | HV | 40' |
| 3.9 | **E-D2 VLM** 176 ảnh qua FPT AI API, 8 luồng | Thắng | **TT** | 5–10' |

→ max-track 1,5–2h (4 người song song, VLM nền).

#### Giai đoạn 4 — Báo cáo + slide + demo + nộp (~3h, kết thúc ~22:00)

| # | Việc | Người | Loại | Thời gian |
| :-- | :--- | :---: | :---: | :---: |
| 4.1 | Viết lại `report/latex/main.tex` + sections theo số liệu mới (giữ §1, §2 chuyển từ v1) | Hiếu + AI | HV+AI | 1h15' |
| 4.2 | Viết §3.1–3.4 phương pháp + phụ lục | AI | AI | 30' |
| 4.3 | 12 slide PPTX | AI draft + Hiếu sửa | HV+AI | 45' |
| 4.4 | Demo `app.py` Gradio + quay video ≤ 3 phút | Hiếu | HV | 30' |
| 4.5 | Khai báo AI (mỗi người 1–2 câu) + title LopXX_NhomYY_MSSV | Hiếu | HV | 20' |
| 4.6 | `pytest` xanh + đóng gói `LopXX_NhomYY_TrichXuatDonThuoc.zip` | Hiếu | HV | 20' |
| 4.7 | 📮 Nộp email `uyenvnb@ptit.edu.vn` trước 23:59 | Hiếu | HV | 5' |

→ max-track 3h.

### 8.2. Tổng thời gian (kèm AI)

| Track | Thời gian | Ghi chú |
| :--- | :---: | :--- |
| AI viết code (1.1–1.4, 2.1–2.7) | ~3 h liên tục | không cần nhóm rảnh, tôi làm chính |
| HV push + train YOLO (1.6, 2.8) | ~2 h 45' | chạy nền 2,5h không chờ |
| Chạy số liệu (3) | 1,5–2 h | 4 người song song sau khi YOLO xong ~16:30 |
| Báo cáo + slide + nộp (4) | ~3 h | Hiếu + AI draft |
| **Tổng từ T0 đến nộp** | **~11 h** | T0=00:40 → nộp ~11:40 (dư ~12h buffer tới 23:59) |

**Điểm nghẽn + buffer:**
- YOLO train trễ → 3h+ : fallback `yolov8n` (1,5–2h) hoặc `yolov8s` epochs=30 (2h).
- VietOCR fine-tune trễ → > 6h: fallback C2 pretrained (cờ đã code sẵn), báo cáo ghi "C3 đang train, dùng C2 tạm".
- FPT AI rate-limit : tăng số luồng + retry, không ảnh hưởng deadline (3' nếu 8 luồng).
- Báo cáo viết lại : tái dùng §1 (giữ nguyên), §2 (giữ VAIPE + split), thêm 1 đoạn "lý do chuyển hướng" viện dẫn số liệu v1 (0,39 / 4 % / 0/18).
- Kaggle upload chậm : chỉ upload `train/` + `valid/`, bỏ `test_unlabeled/` ra ngoài.
- Cài torch CUDA lỗi : infer CPU tạm, train Kaggle là chính.

### 8.3. Lịch giờ cụ thể (giả định bắt đầu 12:00 hôm nay)

| Giờ | Việc chính | Người | Ghi chú |
| :---: | :--- | :---: | :--- |
| 12:00–13:00 | Dọn repo + box/contracts + vaipe/dataset + to_yolo + đẩy Kaggle | AI + Trường | song song |
| 13:00–14:00 | enhance + detect + recognize skeleton; Kaggle bắt đầu train YOLOv8s | AI + Trường | train nền 2,5h |
| 14:00–15:30 | assemble (vlm) + metrics + pipeline + app + make_degraded + notebooks | AI | AI liên tục; Khang test C2 valid |
| 15:30–16:30 | Vietnam YOLO weights về local + `run_pipeline.py samples/` smoke test | Hiếu | YOLO xong ~16:00 |
| 16:30–18:30 | E-B1/B2, E-C1, E-D1, E-X1 ablation, E-A1 3 mức | 4 người song song | VLM nền 10' |
| 18:30–19:30 | E-D2 đầu-cuối 4 pipeline + bootstrap + VLM trên test | Thắng + Hiếu | bảng chính |
| 19:30–22:00 | Báo cáo LaTeX + slide 12 + demo Gradio + video + khai báo AI | Hiếu + AI | draft số liệu sẵn |
| 22:00–22:30 | `pytest` + đóng gói zip | Hiếu | |
| 22:30 | 📮 Nộp email | Hiếu | dư 1h30' buffer |

### 8.4. Thứ tự cắt khi trễ (cắt từ trên xuống, giữ phần dưới)

| Ưu tiên | Cắt khi nào | Thay bằng |
| :--- | :--- | :--- |
| 1 (cắt trước) | `app.py` Gradio | CLI `run_pipeline.py` + ảnh overlay (đủ cho demo) |
| 2 | Độ bền E-A1 3 mức → 2 mức (L0, L4) | bỏ L2 |
| 3 | Ablation E-X1 | bỏ dòng "B chuẩn C chuẩn", giữ 2 dòng còn lại |
| 4 | E-B3, E-C2, E-A2, E-A3 (tham số/thời gian) | ghi tham số phụ lục, không vẽ hình |
| 5 | VietOCR fine-tune (C3) 20k→10k iter, hoặc bỏ về C2 pretrained (đã code sẵn cờ) | dùng C2 làm pipeline chính |
| **Không cắt** | B-YOLO, C-VietOCR pretrained, D2-VLM, E-B1, E-C1, E-D1, pipeline + ghép báo cáo + nộp | |

---

## 9. Kiểm thử tối thiểu (`code/tests/`)

| File | Kiểm tra |
| :--- | :--- |
| `test_boxes.py` | IoU, đổi `xyxy↔xywh`, ghép tham lam (ví dụ tay 3 box) |
| `test_metrics.py` | CER/WER ví dụ biết trước; F1 khoản: khớp hoàn hảo = 1, thiếu 1 khoản, thừa 1 khoản |
| `test_group.py` | D2 trên nhãn chuẩn của 3 ảnh train cho đúng số khoản = số `drugname` |
| `test_rules.py` | D1 trên 20 chuỗi mẫu (mỗi lớp vài chuỗi) |
| `test_pipeline.py` | `run_pipeline.py` chạy 1 ảnh mẫu không lỗi (bỏ qua nếu thiếu trọng số: `pytest.skip`) |

`pytest` phải xanh trước mỗi lần merge vào `main`.

---

## 10. Sản phẩm nộp

```
LopXX_NhomYY_TrichXuatDonThuoc.zip
├── slides/   slide.pptx, slide.pdf            # 10–15 slide, 10 phút
├── code/     src/, experiments/, tools/, notebooks/, tests/, samples/ (8 ảnh), run_pipeline.py, app.py,
│             requirements.txt, requirements-dl.txt, README.md, scripts/download_weights.sh
├── report/   report.pdf                       # 4–10 trang
└── video/    demo.mp4 (≤ 3 phút) hoặc link
```

**README code** phải có: môi trường (Python 3.11, Tesseract 5 + `vie`), lệnh cài, lệnh tải trọng số, lệnh chạy:

```bash
python run_pipeline.py samples/ --out out/                 # pipeline DL mặc định
python run_pipeline.py samples/ --detector classical --ocr tesseract   # pipeline cổ điển
python run_pipeline.py samples/ --enhance deskew,shadow     # bật tiền xử lý
python app.py                                               # demo Gradio
```

Đầu ra mong đợi: `out/<id>.json` (PrescriptionResult) + `out/<id>_overlay.png` (box tô màu theo lớp, đánh số khoản),
và link tải dữ liệu (Drive nhóm).

**Dàn ý báo cáo (4–10 trang):**
1. Giới thiệu: bài toán, đầu vào/đầu ra, khó khăn, đóng góp; lý do chuyển từ thuần cổ điển sang DL + cổ điển đối chứng (½ trang)
2. Dữ liệu: VAIPE, split theo nhóm, `items_gt`, bộ suy giảm (¾ trang)
3. Phương pháp: sơ đồ + A, B, C, D; mỗi bước ghi chương học phần, tham số, lý do (2½ trang)
4. Kết quả: E-B1/B2, E-C1, **E-D2 (bảng chính)**, E-X1 ablation, E-A1 độ bền (biểu đồ), thời gian (3 trang)
5. Thảo luận: khi nào cổ điển đủ, khi nào DL cần, VLM mạnh/yếu ở đâu, giới hạn (ảnh VAIPE sạch, mẫu đồng bộ — chưa kiểm chứng trên ảnh chụp thật), hướng phát triển (½–1 trang)
6. Phân công (bảng mục 7) · 7. Khai báo sử dụng AI (mỗi người 1–2 câu)

**Slide (≈ 12):** bài toán (1) · dữ liệu (1) · sơ đồ pipeline (1) · A (2) · B (2) · C (2) · D + VLM (2) · bảng chính + độ bền (1) · kết luận (1), mỗi người trình bày phần mình.

**Video:** chạy `app.py`, tải 1 ảnh test sạch → JSON + overlay; tải ảnh L3 → so có/không tiền xử lý; 1 ca VLM đúng/sai.

---

## 11. Rủi ro và cách xử lý

| Rủi ro | Dấu hiệu | Xử lý |
| :--- | :--- | :--- |
| **Hạn nộp hôm nay** | 11/11/2026 đã trôi qua nửa ngày | Co lịch mục 8 theo bảng cắt. Khả thi nếu cắt Gradio + E-B3/E-C2/E-D3 |
| **YOLO chưa train xong** | Kaggle bị ngắt hoặc > 3h | Dùng YOLO `yolov8n` (không `s`), imgsz=640, batch=4, epochs=30 (1h30'). Hoặc dùng classical detection + Tesseract (E-D2 dòng (i)) làm "pipeline chính" tạm, ghi rõ |
| **VietOCR fine-tune trễ / CER cao** | Quá 08:00 chưa có weights | Fallback C2 pretrained (đã code sẵn cờ); báo cáo ghi "C3 đang train, dùng C2 tạm" — vẫn nộp được |
| **FPT AI không phục vụ Qwen-VL** | 404 / model not found | Đổi `VLM_MODEL` sang model FPT AI có; hoặc fallback OpenRouter `qwen/qwen2.5-vl-32b` (cần key) |
| **API key FPT AI chưa có** | Thắng chưa nhận key | Dùng OpenRouter tạm (free $1 credit đủ 176 ảnh), đổi FPT AI sau |
| **Kaggle hết quota GPU** | YOLO hoặc VietOCR bị kill | YOLO `n` imgsz 640 epochs 30; VietOCR giảm 20k→10k iter; hoặc train local RTX 3050 (batch=2, lâu 3×) |
| **Báo cáo trống số liệu đến 09:00** | YOLO/C3 trễ | Nộp với pipeline C2 pretrained (số liệu C2 có sẵn ~03:10); ghi "C3 đang train, phụ lục bổ sung sau" |
| **FPT AI không phục vụ Qwen-VL** | API trả 404 / model not found | Đổi `VLM_MODEL` sang model hiện có của FPT AI, hoặc fallback OpenRouter `qwen/qwen2.5-vl-32b` (cần key) |
| **API key FPT AI chưa có** | Thắng chưa nhận key | Dùng OpenRouter tạm (free credit $1 cho tài khoản mới đủ 176 ảnh test), đổi sang FPT AI sau |
| Kết quả DL quá cao, "nhạt" | F1 khoản ≥ 0,98 ở L0 | Kết luận hợp lệ; trọng tâm sang E-A1 (độ bền) và E-X1 (ablation). Ghi rõ "ảnh VAIPE sạch, mẫu đồng bộ" |
| Kaggle hết quota GPU | Bị ngắt phiên | Lưu checkpoint Kaggle Output mỗi epoch; YOLO `n` nếu thiếu; fallback train local RTX 3050 (batch=2, imgsz=640, lâu gấp 3×) |
| VietOCR khó cài | Lỗi phụ thuộc torch | Venv riêng Python 3.11; nếu vẫn lỗi dùng PaddleOCR rec (`lang=vi`) làm C2 |
| `items_gt` sai do luật D2 | E-D1 < 0,98 | Sửa luật hoặc sửa tay `items_gt` cho các ca soát được; ghi số ca sửa |
| Tiền xử lý làm hỏng ảnh sạch | E-A1 L0 giảm | Bật tiền xử lý có điều kiện: chỉ chỉnh nghiêng khi \|góc\| > 0,5°, chỉ khử bóng khi độ lệch sáng nền lớn |
| **Thông tin cá nhân + VLM API** | Ảnh có tên/mã BN | VAIPE đã che từ nguồn → gửi API được. Nếu mở rộng sang ảnh tự chụp thì local-only, không gửi API |
| **Báo cáo trống số liệu** | До 09:00 chưa có YOLO weights | Nộp với số liệu Tesseract-only + VLM zero-shot; ghi rõ "phiên bản YOLO đang train, báo cáo bổ sung" |

---

## 12. Định nghĩa "xong" cho cả dự án (1 ngày)

- [ ] `pytest` xanh (test tối thiểu mục 9); `run_pipeline.py samples/` chạy được trên máy local (cần `weights/yolo_vaipe.pt` từ Kaggle, fallback classical nếu chưa có).
- [ ] Mỗi thí nghiệm không bị cắt có `results/<mã>/{config.json, table.csv}` và số liệu khớp báo cáo.
- [ ] Bảng chính E-D2 có **4–5 dòng** (cổ điển, lai YOLO+Tesseract, DL YOLO+VietOCR-C3, fallback C2 nếu C3 trễ, VLM) với F1 khoản + khoảng tin cậy 95 % trên **176 ảnh test**.
- [ ] Biểu đồ E-A1 có **3 mức** suy giảm (L0, L2, L4) thay vì 5 mức.
- [ ] Ablation E-X1 có **3 dòng** (gốc, B chuẩn, C chuẩn).
- [ ] Báo cáo 4–10 trang, ghi chương học phần ở từng bước, có phân công + khai báo AI, có lý do chuyển từ v1 sang v2 (số liệu v1: phân loại 0,39 trên ảnh thật, bảng 4% dòng đúng).
- [ ] 10–15 slide (.pptx + .pdf), video ≤ 3 phút (khuyến khích), `.zip` đặt tên đúng quy định.
- [ ] **Nộp trước 23:59 ngày 11/11/2026** qua email `uyenvnb@ptit.edu.vn`.
