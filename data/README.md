# data/

## Cấu trúc thư mục

`
data/
├── image/          ← Dataset ảnh (NWPU-RESISC45 hoặc tập dữ liệu ảnh khác)
│   ├── train/      ← Đặt các folder lớp vào đây (train set)
│   └── test/       ← Đặt các folder lớp vào đây (test set)
│
├── tabular/        ← Dataset dạng bảng (IEEE-CIS Fraud Detection)
│   ├── raw/        ← Đặt file CSV gốc vào đây (train_transaction.csv, ...)
│   └── processed/  ← Output sau tiền xử lý
│
└── text/           ← Dataset văn bản (RAGTruth)
    ├── raw/        ← Parquet files từ HuggingFace (tự động tải qua script)
    └── processed/  ← Output sau tiền xử lý
`

## Hướng dẫn

### 1. Ảnh (NWPU-RESISC45)
Đặt 45 folder lớp (airplane/, airport/, ...) vào data/image/train/ và data/image/test/.

### 2. Bảng (IEEE-CIS Fraud)
Tải từ Kaggle: https://www.kaggle.com/c/ieee-fraud-detection/data
Đặt 	rain_transaction.csv, 	rain_identity.csv, 	est_transaction.csv, 	est_identity.csv vào data/tabular/raw/.

### 3. Văn bản (RAGTruth)
Chạy script tải tự động:
`
python download_text_dataset.py
`
File 
agtruth_full.parquet sẽ được tạo trong data/text/raw/.
