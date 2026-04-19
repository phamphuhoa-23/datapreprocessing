# PDF Highlights & Notes — CSC14004 Đồ Án 1

> Trích xuất từ `CSC14004 - Data Mining - P1.pdf` kết hợp với `FEEDBACK.MD` và `REQUIREMENTS_CHECKLIST.md`.  
> Cập nhật: 06/04/2026

---

## §1 — Mục tiêu đồ án (highlights)

- Tiền xử lý không chỉ là làm sạch — phải **giải thích bằng công cụ thống kê**, **so sánh có kiểm soát**, **thiết kế pipeline tái sử dụng**.
- **[Bắt buộc]** = bắt buộc toàn nhóm; **[Nâng cao]** = điểm thưởng.
- Mỗi phần yêu cầu: **phân tích ablation** (thay đổi từng thành phần & đo ảnh hưởng) + **kiểm định thống kê**.

---

## §2.1 — Phần 1: Tiền xử lý ảnh số

### §2.1.1 Dataset (highlighted)
- ≥ **5 lớp**, ≥ **5.000 ảnh** — NWPU-RESISC45 dùng 45 lớp / 31.500 ảnh ✅

### §2.1.2 — Phân tích thống kê (highlighted — toàn bộ mục a→d bắt buộc)

| Mục | Yêu cầu |
|-----|---------|
| a | Histogram + KDE phân phối pixel **theo từng kênh màu** (R, G, B) |
| b | Class imbalance: tỷ lệ mỗi lớp, kiểm tra lớp nào > **3× lớp nhỏ nhất** |
| c | pHash duplicate — báo cáo tỷ lệ trùng, xử lý (thêm lý thuyết Hamming nếu near-dup) |
| d | Mean intensity + std theo lớp → **boxplot phân lớp** |

**Feedback mới (FEEDBACK.MD):**
- **[FIX]** §b: Dùng **pie chart** thay vì bar chart cho class imbalance
- **[FIX]** §a/pixel: Giải thích rõ tại sao chọn 5 lớp đại diện (không nói rõ)
- **[FIX]** §d boxplot: Đừng sort theo median, fix `rotation=90 fontsize=7`; thêm per-class brightness/contrast report dạng DataFrame

### §2.1.3 — Kỹ thuật tiền xử lý (highlighted — toàn bộ bắt buộc)

| Mục | Yêu cầu |
|-----|---------|
| a | Resize ≥ **3 kích thước** (32×32, 64×64, 128×128), SSIM + PSNR so với gốc, đường cong SSIM, **biện hộ kích thước chọn** |
| b | ≥ **3 không gian màu** (RGB, Grayscale, HSV, LAB), PCA explained variance k=50, thảo luận không gian màu nào tốt nhất |
| c | **4 phương pháp chuẩn hóa**: Min-Max[0,1], Min-Max[-1,1], Z-score global, Z-score per-channel; **KS test** p-value + giải thích |
| d | **[Bắt buộc] Augmentation** ≥ 5 phép (flip H/V, rotate, crop, noise Gaussian, brightness); **t-SNE** tập gốc vs augmented |
| e | **[Nâng cao]** PCA scree plot, số component cho **90%, 95%, 99%** variance; 2D/3D tô màu theo lớp; silhouette |
| f | **[Nâng cao]** Sobel/Prewitt/Canny ≥ 2 bộ siêu tham số mỗi loại; **Edge Density** theo lớp; ANOVA một chiều |

**Feedback mới (FEEDBACK.MD):**
- **[FIX §a]** Chọn **128×128** làm kích thước cuối cùng (KNN cao + mất mát trung bình)
- **[FIX §b]** Phân tích màu sắc nên dùng **ảnh 128×128** (như đã chọn ở §a), không dùng 256×256 gốc
- **[OPT §c]** Có thể bỏ cell visualize ảnh trước/sau chuẩn hóa (bình thường, ít giá trị)
- **[🔴 CRITICAL §f]** Edge detection **SAI HOÀN TOÀN** — Đọc lại đề: siêu tham số chính là **ngưỡng T** (threshold), không phải kernel_size hay gaussian_blur
  - Prewitt: áp dụng đúng kernel Prewitt (không chỉ có Sobel)
  - Sobel: không có ngưỡng T → edge density = 0 (sai)
  - Cần ablation ngưỡng T: `T_values = [30, 50, 80, 120]`
  - Canny: ablation `sigma + T1/T2` (4 configs)
  - **XÓA code cũ, đọc lại PDF mục f, làm lại từ đầu**

---

## §2.2 — Phần 2: Tiền xử lý dạng bảng (highlighted)

### §2.2.1 Dataset
- ≥ **10 thuộc tính**, ≥ **10.000 records**, cả số lẫn phân loại
- **Tỷ lệ giá trị thiếu ≥ 5%** trên ít nhất 1 thuộc tính

### §2.2.2 — EDA chuyên sâu (highlighted)
| Mục | Yêu cầu |
|-----|---------|
| a | Shapiro-Wilk (n≤5000) hoặc **D'Agostino-Pearson** (n>5000); phân loại chuẩn/không chuẩn |
| b | Heatmap **Pearson + Spearman**; cặp đa cộng tuyến \|r\|>0.9 → đề xuất xử lý |
| c | Missingno matrix; **Little's MCAR test**; phân loại MCAR/MAR/MNAR |

### §2.2.3 — Kỹ thuật tiền xử lý (highlighted)
| Mục | Yêu cầu |
|-----|---------|
| a | **5 chiến lược** điền khuyết: Mean, Median, Mode, kNN(k∈{3,5,10}), MICE; 10% MCAR nhân tạo → RMSE |
| b | **[Bắt buộc]** 4 outlier methods: IQR+Z-score, Isolation Forest (contamination∈{0.01,0.05,0.1}), LOF (k∈{10,20,50}), DBSCAN; Jaccard overlap; KS test tác động |
| c | Min-Max, Z-score, Robust, Quantile (uniform+normal); **Levene's test** homoscedasticity; violin plot |
| d | One-Hot, Ordinal + **Target Encoding** (CV) + **Binary Encoding** (high-cardinality) + **Frequency Encoding**; VIF |
| e | **[Bắt buộc]** Lọc: ANOVA F-test + Chi-square + MI; Model: RF/GB importance + **RFE với CV**; Giảm chiều: PCA + t-SNE + UMAP; **CV F1-score chart** |
| f | **[Nâng cao]** SMOTE + ADASYN + Undersampling; P/R/F1-macro/AUC-ROC; giải thích không resampling trước split |

**Feedback mới (FEEDBACK.MD):**
- **[FIX §4b]** Chưa giải thích rõ tại sao chọn method cụ thể (DBSCAN ít thay đổi phân phối nhất so với IQR nhiều nhất — phải giải thích nguyên nhân)
- **[FIX §4d/§4e]** code heuristic, subsample quá đơn giản — kiểm tra lại §d và §e
- **[🔴 §4e]** RFE với cross-validation bị **COMMENT OUT** → phải bật lên
- **[🔴 §4e]** CV F1-score chart cũng bị **COMMENT OUT** → phải bật lên

---

## §2.3 — Phần 3: Tiền xử lý văn bản (highlighted)

### §2.3.1 Dataset
- ≥ **10.000 mẫu**, ≥ **2 nhãn phân loại**

### §2.3.2 — Text EDA (highlighted)
| Mục | Yêu cầu |
|-----|---------|
| a | Phân phối độ dài (số từ, **số ký tự**) theo nhãn; **Mann-Whitney U test** |
| b | Word cloud + **top-50 từ phổ biến** theo lớp; **TTR** |
| c | Log-log plot Zipf; kiểm tra tuân theo định luật |

### §2.3.3 — Kỹ thuật (highlighted)
| Mục | Yêu cầu |
|-----|---------|
| a | Pipeline: lowercase, HTML, URL, mention, hashtag, special chars, whitespace; **báo cáo tỷ lệ từ vựng thay đổi** và phân phối độ dài **mỗi bước riêng** |
| b | **4 chiến lược tokenization**: word-level (NLTK+spaCy), sentence, character, **subword BPE** (HuggingFace); vocab size + OOV + avg length |
| c | Stop words; vocab trước/sau; **MI trung bình** trước/sau; NB hiệu năng; thảo luận |
| d | Porter, Snowball, WordNet; **collision rate**; LR **5-fold CV** |
| e | BoW, TF-IDF (n=1/2/3), **Word2Vec** (train on data); sparsity + cosine similarity + t-SNE + **silhouette score** |
| f | **[Nâng cao]** Sentence Transformer `all-MiniLM-L6-v2`; K-Means silhouette + Linear SVM vs TF-IDF |

**Feedback mới (FEEDBACK.MD):**
- **[🔴 CRITICAL §6a]** Bảng **per-step vocab thay đổi** chưa có — đang gộp tất cả bước vào 1 hàm. Cần bảng:  
  `| Bước | Vocab trước | Vocab sau | % thay đổi |`  
  `| original → lowercase → remove_html → remove_url → remove_punct → normalize_ws |`
- **[FIX §5a]** Đã có số từ / số câu, nhưng **chưa vẽ biểu đồ số ký tự** — phải thêm vào subplot 2×2

---

## §3 — Yêu cầu triển khai (highlights)

- Python + Jupyter Notebooks
- Thư viện: numpy, pandas, matplotlib, seaborn, scikit-learn, scipy, statsmodels, opencv/Pillow, nltk/spacy, missingno, imbalanced-learn
- **Tất cả kết quả số phải in ra, đặt trong biến Python** để dễ kiểm tra
- Mỗi kỹ thuật: **(i) markdown lý thuyết** → **(ii) code** → **(iii) markdown phân tích**

---

## §5 — Tiêu chí chấm điểm (highlights)

| Tiêu chí | Điểm |
|----------|------|
| Phần 1 – Ảnh (bắt buộc) | **25%** |
| — pHash, class imbalance, KS test | 7% |
| — SSIM/PSNR đúng | 10% |
| — Ablation + PCA/t-SNE | 8% |
| Phần 2 – Bảng (bắt buộc) | **25%** |
| — EDA (phân phối, MCAR test) | 7% |
| — Outlier + encoding | 10% |
| — Feature selection + CV F1 | 8% |
| Phần 3 – Văn bản | **20%** |
| — Text EDA + Zipf | 5% |
| — Pipeline + tokenization (OOV) | 8% |
| — Vectorization + silhouette + ST | 7% |
| Chất lượng code + báo cáo | **30%** |
| — Code sạch, AI ≤ 30% | 10% |
| — Cấu trúc + README | 5% |
| — Báo cáo PDF phân tích, thảo luận | 15% |
| Điểm thưởng | 10% |
| **Tổng (1)+(2)+(5)+(3 HOẶC 4)** | **110%** |

---

## Tóm tắt các vấn đề CẦN SỬA (ưu tiên cao → thấp)

### 🔴 CRITICAL (phải sửa)
1. **03_advanced_image** — Edge detection (§2.1.3.f): **Xóa và làm lại** — siêu tham số chính là ngưỡng T, không phải kernel_size. Thêm Prewitt đúng cách.
2. **03_advanced_image** — PCA: Hiện đang ghi ">100" cho số components 90/95/99% — phải in **con số thực tế**.
3. **03_advanced_image** — t-SNE/PCA 2D: Đang chỉ dùng 10 lớp / 50 mẫu — phải dùng **tất cả 45 lớp**, sample đủ lớn. Thêm **3D PCA**.
4. **04_tabular_preprocessing** — §4e: **Bỏ comment** RFE with CV + CV F1-score chart.
5. **05_text_preprocessing** — §6a: Thêm **bảng per-step vocab change** (từng bước riêng biệt).
6. **05_text_preprocessing** — §5a: Thêm **subplot số ký tự** vào biểu đồ phân phối độ dài.

### 🟡 MEDIUM (nên sửa)
7. **01_EDA_image** — Class imbalance: Đổi từ bar chart → **pie chart**.
8. **01_EDA_image** — Boxplot brightness: Bỏ sort theo median, fix `rotation=90 fontsize=7`.
9. **02_preprocessing_image** — Ghi rõ **128×128 là kích thước được chọn** trong phân tích.
10. **02_preprocessing_image** — §b Color space: dùng **ảnh 128×128** thay vì 256×256.
11. **04_tabular_preprocessing** — Giải thích rõ **tại sao chọn DBSCAN vs IQR** dựa trên kết quả KS.
