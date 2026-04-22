# Đồ Án 1 — Tiền Xử Lý Dữ Liệu

**Môn học:** Khai thác dữ liệu và ứng dụng (CSC14004) — HK2 2025/2026
**Trường:** Đại học Khoa học Tự nhiên, ĐHQG TP.HCM

---

## Thành viên nhóm

| MSSV     | Họ và Tên           | Email                         | Vai trò       |
| -------- | ---------------------- | ----------------------------- | -------------- |
| 23122048 | Nguyễn Lâm Phú Quý | 23122048@student.hcmus.edu.vn | Thành viên   |
| 23122030 | Phạm Phú Hòa        | 23122030@student.hcmus.edu.vn | Nhóm trưởng |
| 23122041 | Đào Sỹ Duy Minh     | 23122041@student.hcmus.edu.vn | Thành viên   |
| 23122044 | Trần Chí Nguyên     | 23122044@student.hcmus.edu.vn | Thành viên   |
| 23122039 | Huỳnh Trung Kiệt     | 23122039@student.hcmus.edu.vn | Thành viên   |

---

## Mô tả tập dữ liệu

### Phần 1 — Ảnh số: NWPU-RESISC45

- **Nguồn:** [Kaggle](https://www.kaggle.com/datasets/aqibrehmanpirzada/nwpuresisc45)
- **Kích thước:** 31,500 ảnh (27,000 train / 4,500 test), 45 lớp, mỗi ảnh 256×256×3 RGB
- **Bài toán:** Phân loại cảnh viễn thám (Remote Sensing Scene Classification)

### Phần 2 — Dữ liệu bảng: IEEE-CIS Fraud Detection

- **Nguồn:** [Kaggle](https://www.kaggle.com/competitions/ieee-fraud-detection/data)
- **Kích thước:** ~590,540 giao dịch, ~430 thuộc tính, nhãn nhị phân `isFraud`
- **Bài toán:** Phát hiện giao dịch gian lận (Binary Classification)

### Phần 3 — Văn bản: RAGTruth

- **Nguồn:** [HuggingFace](https://huggingface.co/datasets/wandb/RAGTruth-processed)
- **Kích thước:** ~17,790 mẫu, 2 nhãn (Supported / Hallucinated)
- **Bài toán:** Phát hiện ảo giác (Hallucination) trong văn bản do LLM sinh ra

---

## Hướng dẫn tải và cài đặt dữ liệu 

Để đảm bảo tính đồng bộ và tiết kiệm thời gian, toàn bộ dữ liệu (Raw Data) đã được tổng hợp và cấu trúc sẵn trên Google Drive.

1. **Link tải dữ liệu**: [**Google Drive - Data Mining Project**](https://drive.google.com/file/d/1N2Ru-ylEtyK_4LEXSjtFZrzgemTpwN6s/view?usp=drive_link)
2. **Cách cài đặt**:
   * Tải file dữ liệu từ link trên.
   * Giải nén và đặt thư mục `data` vào ngay thư mục gốc của dự án (cùng cấp với thư mục `notebooks`).
   * **Cấu trúc thư mục chuẩn**:
     ```
     Source/
     ├── README.md
     ├── requirements.txt
     ├── data/
     │   ├── raw/
     │   │   ├── image/      # Dữ liệu gốc NWPU-RESISC45
     │   │   ├── tabular/    # Dữ liệu gốc IEEE-CIS Fraud
     │   │   └── text/       # Dữ liệu gốc RAGTruth
     │   └── processed/      # Dữ liệu sau tiền xử lý (.npy, .csv, .json)
     │       ├── image/
     │       ├── tabular/
     │       └── text/
     ├── outputs/             # Hình ảnh, biểu đồ kết quả
     │   ├── image/
     │   ├── tabular/
     │   └── text/
     ├── notebooks/
     │   ├── 01_EDA_image.ipynb
     │   ├── 02_preprocessing_image.ipynb
     │   ├── 03_advanced_image.ipynb
     │   ├── 04_tabular_preprocessing.ipynb
     │   └── 05_text_preprocessing.ipynb
     └── docs/
         └── report/main.pdf  # Báo cáo PDF
     ```

---

## 🛠️ Hướng dẫn chạy đồ án

### 1. Yêu cầu hệ thống
- **Ngôn ngữ**: Python **3.12.9**
- **RAM**: Khuyến nghị ≥ 16 GB
- **Thư viện**: Xem file `requirements.txt`

### 2. Cài đặt thư viện

```bash
cd Source
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 3. Chạy Notebook

> **Lưu ý**: Dự án sử dụng **Jupytext** để đồng bộ giữa file Python (`.py`) và Notebook (`.ipynb`).

Mở JupyterLab/VS Code và chạy các file trong thư mục `notebooks/` theo thứ tự:

- `01_EDA_image.ipynb` -> `02_preprocessing_image.ipynb` -> `03_advanced_image.ipynb`
- `04_tabular_preprocessing.ipynb`
- `05_text_preprocessing.ipynb`

### 4. Lưu ý về đường dẫn (Path)

Tất cả các file code đều sử dụng đường dẫn tương đối (Relative Path). Chỉ cần bạn đặt thư mục `data` đúng vị trí như hướng dẫn ở trên, code sẽ tự động nhận diện và chạy được trên mọi hệ điều hành (Windows/Linux/macOS) mà không cần chỉnh sửa bất kỳ dòng code nào.

---

## Phân công công việc

| Thành viên                     | Công việc chính                                                                       | Hoàn thành |
| -------------------------------- | ---------------------------------------------------------------------------------------- | ------------ |
| **Phạm Phú Hòa**        | Nhóm trưởng, Review code, Chỉnh sửa tổng thể, Viết báo cáo PDF (LaTeX)         | 100%         |
| **Trần Chí Nguyên**     | Phần 1 (Ảnh): EDA, Pipeline preprocessing, Ablation study                              | 100%         |
| **Nguyễn Lâm Phú Quý** | Phần 2 (Bảng): EDA, Xử lý giá trị thiếu, Phát hiện ngoại lai                   | 100%         |
| **Huỳnh Trung Kiệt**     | Phần 2 (Bảng): Chuẩn hóa, Mã hóa, Lựa chọn đặc trưng, Xử lý mất cân bằng | 100%         |
| **Đào Sỹ Duy Minh**     | Phần 3 (Văn bản): Toàn bộ text preprocessing, Vectorization, Phân tích            | 100%         |
