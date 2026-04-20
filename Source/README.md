# Đồ Án 1 — Tiền Xử Lý Dữ Liệu
**Môn học:** Khai thác dữ liệu và ứng dụng (CSC14004) — HK2 2025/2026  
**Trường:** Đại học Khoa học Tự nhiên, ĐHQG TP.HCM

---

## Thành viên nhóm

| MSSV | Họ và Tên | Email | Vai trò |
|------|-----------|-------|---------|
| 23122048 | Nguyễn Lâm Phú Quý | 23122048@student.hcmus.edu.vn | Nhóm trưởng |
| 23122030 | Phạm Phú Hòa | 23122030@student.hcmus.edu.vn | Thành viên |
| 23122041 | Đào Sỹ Duy Minh | 23122041@student.hcmus.edu.vn | Thành viên |
| 23122044 | Trần Chí Nguyên | 23122044@student.hcmus.edu.vn | Thành viên |
| 23122039 | Huỳnh Trung Kiệt | 23122039@student.hcmus.edu.vn | Thành viên |

---

## Mô tả tập dữ liệu

### Phần 1 — Ảnh số: NWPU-RESISC45
- **Nguồn:** Kaggle — [aqibrehmanpirzada/nwpuresisc45](https://www.kaggle.com/datasets/aqibrehmanpirzada/nwpuresisc45)
- **Kích thước:** 31,500 ảnh (27,000 train / 4,500 test), 45 lớp, mỗi ảnh 256×256×3 RGB
- **Bài toán:** Phân loại cảnh viễn thám (remote sensing scene classification)
- **Đặc điểm:** Hoàn toàn cân bằng lớp (600 train / 100 test mỗi lớp), ảnh trùng lặp rất thấp (~0.05%)

### Phần 2 — Dữ liệu bảng: IEEE-CIS Fraud Detection
- **Nguồn:** Kaggle — [ieee-fraud-detection](https://www.kaggle.com/competitions/ieee-fraud-detection/data)
- **Kích thước:** ~590,540 giao dịch, ~430 thuộc tính (số + phân loại), nhãn nhị phân `isFraud`
- **Bài toán:** Phát hiện giao dịch gian lận (binary classification)
- **Đặc điểm:** Mất cân bằng lớp nghiêm trọng (fraud ≈ 3.5%, ratio 28:1), tỉ lệ giá trị thiếu cao (Identity group: 84.5%)

### Phần 3 — Văn bản: RAGTruth
- **Nguồn:** HuggingFace — [wandb/RAGTruth-processed](https://huggingface.co/datasets/wandb/RAGTruth-processed)
- **Kích thước:** ~17,790 mẫu, 2 nhãn (Supported / Hallucinated)
- **Bài toán:** Phát hiện hallucination trong văn bản do LLM sinh ra
- **Đặc điểm:** Văn bản tiếng Anh, binary classification, lexical boundary giữa 2 lớp thấp

---

## Cấu trúc thư mục

```
Lab1DataMining/
├── README.md
└── Source/
    ├── requirements.txt
    ├── data/
    │   ├── raw/          # Dữ liệu gốc (xem hướng dẫn tải bên dưới)
    │   └── processed/    # Dữ liệu đã xử lý (được tạo khi chạy notebook)
    ├── notebooks/
    │   ├── 01_EDA_image.ipynb          # Phần 1: EDA ảnh
    │   ├── 02_preprocessing_image.ipynb # Phần 1: Pipeline tiền xử lý ảnh
    │   ├── 03_advanced_image.ipynb     # Phần 1 [Nâng cao]: PCA + Edge Detection
    │   ├── 04_tabular_preprocessing.ipynb # Phần 2: Tiền xử lý bảng
    │   └── 05_text_preprocessing.ipynb # Phần 3: Tiền xử lý văn bản
    └── docs/
        ├── main.tex
        └── main.pdf      # Báo cáo PDF
```

---

## Hướng dẫn cài đặt và chạy

### 1. Yêu cầu hệ thống
- Python **3.13.12** (tested); Python 3.10+ có thể tương thích
- RAM: ≥ 16 GB (NB03 dùng IncrementalPCA trên 27,000 ảnh; NB04 xử lý ~590K hàng)
- Disk: ≥ 15 GB (dataset ảnh ~3.5 GB sau giải nén)

### 2. Cài đặt môi trường

```bash
cd Lab1DataMining/Source
python -m venv .venv

# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

pip install -r requirements.txt

# Tải thêm model ngôn ngữ cho spaCy
python -m spacy download en_core_web_sm

# Tải NLTK data
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('punkt')"
```

### 3. Tải dữ liệu

**Phần 1 — NWPU-RESISC45:**
```bash
# Cần tài khoản Kaggle và kaggle CLI
kaggle datasets download -d aqibrehmanpirzada/nwpuresisc45
unzip nwpuresisc45.zip -d Source/data/raw/image/
```
Hoặc tải thủ công từ: https://www.kaggle.com/datasets/aqibrehmanpirzada/nwpuresisc45  
Giải nén vào `Source/data/raw/image/` sao cho có đường dẫn `Source/data/raw/image/train/<class_name>/*.jpg`

**Phần 2 — IEEE-CIS Fraud Detection:**
```bash
kaggle competitions download -c ieee-fraud-detection
unzip ieee-fraud-detection.zip -d Source/data/raw/tabular/
```
Hoặc tải thủ công từ: https://www.kaggle.com/competitions/ieee-fraud-detection/data  
Đặt các file `train_transaction.csv`, `train_identity.csv`, `test_transaction.csv`, `test_identity.csv` vào `Source/data/raw/tabular/`

**Phần 3 — RAGTruth:**
```python
# Tự động tải khi chạy notebook 05 (dùng datasets của HuggingFace)
from datasets import load_dataset
ds = load_dataset("wandb/RAGTruth-processed")
```

### 4. Chạy notebook

```bash
cd Source/notebooks

# Chạy theo thứ tự:
jupyter nbconvert --to notebook --execute 01_EDA_image.ipynb --inplace
jupyter nbconvert --to notebook --execute 02_preprocessing_image.ipynb --inplace
jupyter nbconvert --to notebook --execute 03_advanced_image.ipynb --inplace
jupyter nbconvert --to notebook --execute 04_tabular_preprocessing.ipynb --inplace
jupyter nbconvert --to notebook --execute 05_text_preprocessing.ipynb --inplace
```

Hoặc mở JupyterLab và chạy từng notebook theo thứ tự (Kernel → Restart & Run All).

> **Lưu ý:** NB03 cần ~30 phút (IncrementalPCA trên 27K ảnh). NB04 cần ~45 phút (MICE imputation + feature selection trên 590K hàng). NB05 cần ~15 phút (Sentence Transformer encoding).

---

## Phân công công việc

| Thành viên | Công việc chính | Hoàn thành |
|-----------|-----------------|------------|
| **Phạm Phú Hòa** (Nhóm trưởng) | Kiểm tra và review toàn bộ code/notebook, chỉnh sửa tổng thể, viết và hoàn thiện báo cáo PDF (main.tex), tích hợp kết quả các phần | 100% |
| **Nguyễn Lâm Phú Quý** | Phần 2 (Tabular): EDA bảng, xử lý giá trị thiếu, phát hiện ngoại lai (NB04) | 100% |
| **Huỳnh Trung Kiệt** | Phần 2 (Tabular): Chuẩn hóa, mã hóa phân loại, lựa chọn đặc trưng, xử lý mất cân bằng lớp (NB04) | 100% |
| **Trần Chí Nguyên** | Phần 1 (Ảnh): EDA ảnh, pipeline preprocessing, ablation study (NB01, NB02, NB03) | 100% |
| **Đào Sỹ Duy Minh** | Phần 3 (Văn bản): Toàn bộ text preprocessing, vectorization, phân tích (NB05) | 100% |

---

## Link tài nguyên

| Tài nguyên | Link |
|-----------|------|
| Dataset NWPU-RESISC45 | https://www.kaggle.com/datasets/aqibrehmanpirzada/nwpuresisc45 |
| Dataset IEEE-CIS Fraud | https://www.kaggle.com/competitions/ieee-fraud-detection |
| Dataset RAGTruth | https://huggingface.co/datasets/wandb/RAGTruth-processed |
| Sentence Transformer model | https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2 |
| Báo cáo PDF | `Source/docs/main.pdf` |
