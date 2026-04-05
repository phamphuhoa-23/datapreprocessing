# YÊU CẦU ĐỒ ÁN — CHECKLIST ĐẦY ĐỦ

> Đọc thẳng từ `CSC14004 - Data Mining - P1.pdf`. Tick theo trạng thái **thực tế trong code** (`Source/notebooks/`).
> Cập nhật ngày: 04/04/2026

**Ký hiệu:**
- ✅ Hoàn thiện đầy đủ (code + phân tích dynamic)
- ⚠️ Có nhưng chưa đủ sâu / chưa dynamic / còn hardcode
- ❌ Thiếu hoàn toàn hoặc bị comment out
- 🔶 Nâng cao (bonus) — không bắt buộc

---

## PHẦN 1 — Tiền xử lý ảnh (25%) — `01_EDA_image.py`, `02_preprocessing_image.py`, `03_advanced_image.py`

### §2.1.2 — Phân tích thống kê tập dữ liệu (7%)

| # | Yêu cầu | File | Trạng thái | Ghi chú |
|---|---------|------|------------|---------|
| 1a | Histogram + KDE phân phối pixel theo từng kênh màu (R, G, B) | 01_EDA | ✅ | line ~244; gaussian_kde đúng |
| 1b | Class imbalance: tỷ lệ mỗi lớp, kiểm tra lớp nào > 3× lớp nhỏ nhất | 01_EDA | ✅ | line ~118; imbalance_ratio dynamic |
| 1b+ | Chi-square test phân phối lớp đồng đều | 01_EDA | ✅ | Added — line ~125 |
| 1c | pHash duplicate detection — báo cáo tỷ lệ trùng, xử lý | 01_EDA | ✅ | line ~465; exact + near-dup (Hamming ≤ 4) |
| 1c- | Near-duplicate: có detection nhưng chỉ **sample 2000** ảnh, chưa toàn bộ tập | 01_EDA | ⚠️ | Cân nhắc chạy toàn bộ |
| 1d | Mean intensity + std deviation theo lớp → boxplot | 01_EDA | ✅ | line ~667–738; 45 lớp sorted |
| 1d+ | Kruskal-Wallis + ANOVA + η² cho Brightness & Contrast | 01_EDA | ✅ | line ~747–764; dynamic kết luận |
| 1d- | Kết luận markdown **Phân bố pixel / Brightness** dùng số hardcode | 01_EDA | ⚠️ | line ~443, ~771: hardcode F/H values |

### §2.1.3 — Các kỹ thuật tiền xử lý (10% + 8%)

| # | Yêu cầu | File | Trạng thái | Ghi chú |
|---|---------|------|------------|---------|
| 2a | Resize ≥3 kích thước, SSIM + PSNR so với ảnh gốc | 02_pre | ✅ | 64×64, 128×128, 224×224 |
| 2a | Đường cong SSIM theo kích thước, chọn và biện hộ | 02_pre | ✅ | errorbar plot done |
| 2a | kNN ablation theo resize | 02_pre | ✅ | line ~182–194 |
| 2a | ANOVA + Kruskal + η² + post-hoc Bonferroni trên SSIM | 02_pre | ✅ | Added Bonferroni p_bonf + stars |
| 2a- | Kết luận markdown Resize hardcode SSIM/PSNR values | 02_pre | ⚠️ | line ~259: static table |
| 2b | ≥3 color spaces (RGB, Grayscale, HSV, Lab) | 02_pre | ✅ | 4 không gian |
| 2b | PCA explained variance k=50 cho mỗi color space | 02_pre | ✅ | scree + cumulative plot |
| 2b | kNN ablation theo color space | 02_pre | ✅ | line ~338–358 |
| 2b- | Kết luận markdown Color Space hardcode accuracy values | 02_pre | ⚠️ | line ~376: static table |
| 2b- | Không có kiểm định thống kê so sánh kNN fold scores giữa color spaces | 02_pre | ⚠️ | Thêm Friedman/Wilcoxon thì tốt hơn |
| 2c | 4 normalization (Min-Max[0,1], Min-Max[-1,1], Z-score global, Z-score per-ch) | 02_pre | ✅ | đủ 4 |
| 2c | KS test phân phối trước/sau chuẩn hóa + p-value | 02_pre | ✅ | line ~450–462 |
| 2c | Diễn giải KS effect size D (nhỏ/trung bình/lớn) | 02_pre | ✅ | Added — `ks_effect_size()` |
| 2c | Levene test variance giữa 5 phương pháp | 02_pre | ✅ | line ~465–466 |
| 2c | kNN ablation theo normalization | 02_pre | ✅ | line ~502–519 |
| 2c | Wilcoxon signed-rank (best norm vs Original) + Cohen's d | 02_pre | ✅ | Added — dynamic kết luận |
| 2c- | Kết luận markdown Normalization hardcode các số | 02_pre | ⚠️ | line ~530: static TODO table |
| 2d | Pipeline augmentation ≥5 phép biến đổi (flip H/V, rotate, crop, noise, brightness) | 02_pre | ✅ | 6 phép |
| 2d | t-SNE visualization tập gốc vs tập augmented | 02_pre | ✅ | số lớp tăng dần 5→45 |
| 2d | **Per-augmentation kNN ablation** (mỗi technique riêng biệt) | 02_pre | ✅ | Added — loop qua AUGMENTATIONS |
| 2d | Wilcoxon mỗi technique vs baseline | 02_pre | ✅ | Added — trong loop |
| 2d | Paired t-test + Wilcoxon + Cohen's d variance trước/sau aug | 02_pre | ✅ | line ~763–771 |
| 2d+ | Dynamic kết luận variance test | 02_pre | ✅ | Added — print "LÀM TĂNG / KHÔNG" |
| 2d- | Kết luận markdown Augmentation hardcode values | 02_pre | ⚠️ | line ~811–831: static block lớn |
| 2e 🔶 | PCA scree plot, 90%/95%/99% variance | 03_adv | ✅ | line ~127–163 |
| 2e 🔶 | PCA 2D + 3D projection tô màu theo lớp | 03_adv | ✅ | line ~200–245 |
| 2e 🔶 | t-SNE 2D tô màu theo lớp | 03_adv | ✅ | line ~252–302 |
| 2e 🔶 | Thảo luận mức độ tách biệt lớp | 03_adv | ✅ | line ~308–311 |
| 2e 🔶 | **ARI/Purity metric** thay vì chỉ nhận xét bằng mắt | 03_adv | ⚠️ | Commented out trong code (line ~192) |
| 2f 🔶 | Sobel với ≥2 bộ siêu tham số (ksize=3, ksize=5) | 03_adv | ✅ | line ~500–543 |
| 2f 🔶 | **Prewitt** với ≥2 bộ siêu tham số (raw, σ=1 smooth) | 03_adv | ✅ | line ~500–543 |
| 2f 🔶 | Canny với ≥2 bộ siêu tham số (50/80, 100/200) | 03_adv | ✅ | line ~543 |
| 2f 🔶 | Edge Density (pixel cạnh / tổng pixel) theo lớp | 03_adv | ✅ | line ~549–584 |
| 2f 🔶 | ANOVA một chiều kiểm định khác biệt edge density | 03_adv | ✅ | line ~589, ~628–637 |
| 2f 🔶 | Kết luận: thông tin cạnh có phân biệt lớp? | 03_adv | ✅ | line ~673–674 dynamic |
| 2f- 🔶 | Kết luận markdown tổng kết line ~687 hardcode values | 03_adv | ⚠️ | |

---

## PHẦN 2 — Tiền xử lý bảng (25%) — `04_tabular_preprocessing.py`

### §2.2.2 — EDA thống kê (7%)

| # | Yêu cầu | File | Trạng thái | Ghi chú |
|---|---------|------|------------|---------|
| 3a | D'Agostino-Pearson (n>5000) cho mỗi thuộc tính số | 04_tab | ✅ | line ~315; dynamic classification |
| 3a | Phân loại từng cột: chuẩn / không chuẩn dựa trên p-value | 04_tab | ✅ | dynamic count + kết luận |
| 3a | Dùng kết quả để lựa chọn scaler/correlation method | 04_tab | ✅ | markdown line ~356 |
| 3b | **Heatmap Pearson** | 04_tab | ✅ | Added — subplot 2 heatmaps |
| 3b | **Heatmap Spearman** | 04_tab | ✅ | line ~409–422 |
| 3b | Dynamic Pearson vs Spearman comparison (max|Δr|, mean|Δr|) | 04_tab | ✅ | Added |
| 3b | Phát hiện đa cộng tuyến mạnh |r|>0.9 | 04_tab | ✅ | line ~427–441 |
| 3b | Đề xuất xử lý đa cộng tuyến | 04_tab | ✅ | dynamic print |
| 3c | Missingno visualization (missing matrix, heatmap) | 04_tab | ✅ | line ~574+ |
| 3c | **Little's MCAR test** | 04_tab | ✅ | line ~620–694; per-group |
| 3c | Phân loại cơ chế MCAR/MAR/MNAR + giải thích | 04_tab | ✅ | dynamic verdict line ~692 |

### §2.2.3 — Kỹ thuật tiền xử lý (10% + 8%)

| # | Yêu cầu | File | Trạng thái | Ghi chú |
|---|---------|------|------------|---------|
| 4a | 5 chiến lược: Mean, Median, Mode, kNN(k∈{3,5,10}), MICE | 04_tab | ✅ | line ~1029–1035 |
| 4a | 10% MCAR nhân tạo + RMSE benchmark | 04_tab | ✅ | `benchmark_imputation()` |
| 4a | Bảng so sánh + lựa chọn có lý giải | 04_tab | ✅ | sort by RMSE + dynamic best |
| 4a | **Friedman test + pairwise Wilcoxon + Bonferroni** trên per-column RMSE | 04_tab | ✅ | Added |
| 4b | IQR + Z-score | 04_tab | ✅ | line ~1236–1249 |
| 4b | Isolation Forest contamination ∈ {0.01, 0.05, 0.1} | 04_tab | ✅ | line ~1252–1258 |
| 4b | LOF n_neighbors ∈ {10, 20, 50} | 04_tab | ✅ | line ~1260–1265 |
| 4b | DBSCAN | 04_tab | ✅ | line ~1267–1269 |
| 4b | Tỷ lệ phát hiện + Jaccard similarity giữa methods | 04_tab | ✅ | line ~1288–1305 |
| 4b | KS test đánh giá tác động đến phân phối sau loại bỏ ngoại lai | 04_tab | ✅ | line ~1319–1323 |
| 4b- | Kết quả KS test per-method chưa có dynamic kết luận rõ | 04_tab | ⚠️ | chỉ print stat/p, thiếu "=> phân phối thay đổi/không" |
| 4c | Min-Max, Z-score, Robust Scaling, Quantile Transform (uniform + normal) | 04_tab | ✅ | ≥4 methods |
| 4c | Levene's test homoscedasticity sau chuẩn hóa | 04_tab | ✅ | line ~1390–1404 |
| 4c | Violin plot theo thuộc tính | 04_tab | ✅ | line ~1408–1423 |
| 4c- | Violin plot: chỉ 1 cột đại diện, không phải đa cột | 04_tab | ⚠️ | Đề yêu cầu "từng thuộc tính" |
| 4d | One-Hot + Ordinal encoding cơ bản | 04_tab | ✅ | |
| 4d | Target Encoding với cross-validation (tránh leakage) | 04_tab | ✅ | line ~1449 |
| 4d | Binary Encoding (high-cardinality) | 04_tab | ✅ | |
| 4d | Frequency Encoding | 04_tab | ✅ | |
| 4d | VIF đo đa cộng tuyến mới sau encoding | 04_tab | ✅ | |
| 4e | ANOVA F-test (thuộc tính số) | 04_tab | ✅ | line ~1823 |
| 4e | Chi-square test (thuộc tính phân loại) | 04_tab | ✅ | line ~1874 |
| 4e | Mutual Information | 04_tab | ✅ | line ~1831 |
| 4e | RF feature importance | 04_tab | ✅ | line ~1901 |
| 4e | GradientBoosting feature importance | 04_tab | ✅ | line ~1910 |
| 4e | **RFE với cross-validation** | 04_tab | ❌ | **COMMENTED OUT** line ~1935–1971 |
| 4e | **CV F1-score chart theo số lượng đặc trưng** | 04_tab | ❌ | **COMMENTED OUT** line ~1961–1977 |
| 4e | PCA | 04_tab | ✅ | |
| 4e | t-SNE visualization | 04_tab | ✅ | |
| 4e | UMAP | 04_tab | ⚠️ | try/except — cài hay không tùy môi trường |
| 4f 🔶 | SMOTE + ADASYN + Random Under-sampling | 04_tab | ✅ | line ~2062+ |
| 4f 🔶 | P/R/F1-macro/AUC-ROC trên tập test chưa resampled | 04_tab | ✅ | |
| 4f 🔶 | Giải thích không resampling trước chia train/test | 04_tab | ✅ | |
| Cấu trúc | Pipeline **sequential** (impute/outlier/scale nối nhau) thay vì **independent benchmark** | 04_tab | ⚠️ | Đề yêu cầu mỗi bước benchmark độc lập, hiện tại vẫn còn phụ thuộc nhau một phần |

---

## PHẦN 3 — Tiền xử lý văn bản (20%) — `05_text_preprocessing.py`

### §2.3.2 — Text EDA (5%)

| # | Yêu cầu | File | Trạng thái | Ghi chú |
|---|---------|------|------------|---------|
| 5a | Phân phối độ dài (số từ, số ký tự) theo nhãn | 05_text | ✅ | |
| 5a | Mann-Whitney U test khác biệt độ dài giữa lớp | 05_text | ✅ | line ~338 |
| 5a | Effect size r (Z/√N) + interpret_r() | 05_text | ✅ | Added |
| 5b | Word cloud theo từng lớp | 05_text | ✅ | |
| 5b | Bảng top-50 từ phổ biến | 05_text | ✅ | |
| 5b | TTR (Type-Token Ratio) | 05_text | ✅ | line ~479 |
| 5b | Mann-Whitney TTR giữa lớp | 05_text | ✅ | line ~494 |
| 5c | Log-log plot Zipf | 05_text | ✅ | |
| 5c | Kiểm tra mức độ tuân theo Zipf (slope α so với lý thuyết) | 05_text | ✅ | line ~527 dynamic |

### §2.3.3 — Kỹ thuật tiền xử lý (8%)

| # | Yêu cầu | File | Trạng thái | Ghi chú |
|---|---------|------|------------|---------|
| 6a | Pipeline: lowercase, HTML, URL, mention, hashtag, special chars, whitespace | 05_text | ✅ | line ~596–605 |
| 6a | **Bảng per-step vocab thay đổi** (MỖI BƯỚC riêng: lowercase → remove_html → remove_url → ...) | 05_text | ❌ | **CRITICAL GAP** — code hiện tại chỉ đo Raw→Normalized→NoStop→Lemmatized, KHÔNG phải per individual step |
| 6a | Tác động đến phân phối độ dài qua từng bước | 05_text | ⚠️ | Đo theo stage tổng hợp, không phải từng bước pipeline |
| 6b | Word-level tokenization (NLTK + spaCy) | 05_text | ✅ | |
| 6b | Sentence-level tokenization | 05_text | ✅ | |
| 6b | Character-level tokenization | 05_text | ✅ | |
| 6b | Subword BPE (HuggingFace tokenizers) | 05_text | ✅ | line ~677–698 |
| 6b | Bảng: vocab size, OOV ratio, avg token length cho 4 methods | 05_text | ✅ | line ~702–706 |
| 6c | Loại bỏ stop words | 05_text | ✅ | |
| 6c | Vocab size trước/sau | 05_text | ✅ | |
| 6c | MI trung bình trước/sau (from/after stopword removal) | 05_text | ✅ | |
| 6c | NB performance trước/sau + so sánh | 05_text | ✅ | line ~864 |
| 6c | Wilcoxon signed-rank fold scores + Cohen's d | 05_text | ✅ | Added |
| 6c | Thảo luận: bỏ stop words có luôn cải thiện? | 05_text | ✅ | line ~886 |
| 6d | Porter Stemmer + collision rate | 05_text | ✅ | |
| 6d | Snowball Stemmer + collision rate | 05_text | ✅ | |
| 6d | WordNet Lemmatizer + collision rate | 05_text | ✅ | |
| 6d | LR 5-fold CV đánh giá tác động | 05_text | ✅ | |
| 6d | Friedman test + pairwise Wilcoxon + Bonferroni (stem/lemma methods) | 05_text | ✅ | Added |
| 6e | BoW | 05_text | ✅ | |
| 6e | TF-IDF n-gram (n=1, 2, 3) | 05_text | ✅ | |
| 6e | Word2Vec (huấn luyện trên tập) | 05_text | ✅ | line ~1245 |
| 6e | Sparsity ratio cho mỗi biểu diễn | 05_text | ✅ | line ~1229–1242 |
| 6e | Cosine similarity intra-class vs inter-class | 05_text | ✅ | line ~1276–1300 |
| 6e | t-SNE 2D tô màu theo nhãn | 05_text | ✅ | |
| 6e | Silhouette score đánh giá tách lớp | 05_text | ✅ | line ~1366–1383 |
| 6e | Bootstrap silhouette (uncertainty estimate) | 05_text | ✅ | Added — 10 rounds |
| 6e | Friedman test so sánh silhouette giữa 5 methods | 05_text | ✅ | Added |
| 6e | Pairwise Wilcoxon + Bonferroni vectorization | 05_text | ✅ | Added |
| 6f 🔶 | Sentence Transformer (all-MiniLM-L6-v2) | 05_text | ✅ | line ~1462–1571 |
| 6f 🔶 | K-Means + silhouette: TF-IDF vs ST | 05_text | ✅ | |
| 6f 🔶 | Linear SVM so sánh TF-IDF vs ST | 05_text | ✅ | |

---

## YÊU CẦU KỸ THUẬT CHUNG (§3.1)

| Yêu cầu | Trạng thái | Ghi chú |
|---------|------------|---------|
| Tất cả kết quả số phải được **in ra** (không chỉ trong markdown) | ⚠️ | Nhiều kết luận still hardcode trong `# markdown cells` |
| **Mỗi kỹ thuật**: (i) markdown lý thuyết; (ii) code; (iii) markdown phân tích kết quả | ⚠️ | (iii) nhiều chỗ hardcode → cần dùng `print()` dynamic |
| Tất cả kết quả đặt trong **biến Python** để dễ kiểm tra | ⚠️ | Nhiều nơi print trực tiếp, không gán biến |
| Notebook **Restart & Run All** chạy được | ⚠️ | Chưa test — dataset path cần đúng |
| Sử dụng numpy, pandas, matplotlib, seaborn, scikit-learn, scipy, statsmodels, opencv, nltk/spacy, missingno | ✅ | Đủ thư viện |

---

## YÊU CẦU BÁO CÁO PDF (§3.2) — **CHƯA LÀM**

| Yêu cầu | Trạng thái | Ghi chú |
|---------|------------|---------|
| Tối thiểu 20 trang | ❌ | Chưa làm |
| Tóm tắt điều hành (1 trang) | ❌ | |
| Bảng so sánh tổng hợp tất cả kỹ thuật có chỉ số định lượng | ❌ | |
| Thảo luận: lý do lựa chọn, hạn chế, hướng cải tiến | ❌ | |
| Phân công công việc | ❌ | |
| Toàn bộ bằng tiếng Việt | ❌ | |

---

## YÊU CẦU NỘP BÀI (§6)

| Yêu cầu | Trạng thái | Ghi chú |
|---------|------------|---------|
| README.md: tên/MSSV, dataset, cài đặt môi trường, chạy notebook, phân công | ✅ | Có README trong repo |
| requirements.txt với version cụ thể | ⚠️ | Chưa kiểm tra có đủ version không |
| Cấu trúc thư mục chuẩn (data/raw, data/processed, notebooks/) | ✅ | |
| Notebook Restart & Run All không lỗi | ⚠️ | Cần test |

---

## TÓM TẮT — CÁC VIỆC CẦN LÀM NGAY (ưu tiên cao → thấp)

### 🔴 CRITICAL — Điểm bị trừ nặng nếu thiếu

| STT | Việc cần làm | File | Trạng thái |
|-----|-------------|------|-------------|
| C1 | **Bảng per-step vocab** trong pipeline chuẩn hóa §2.3.3a | 05_text | ✅ DONE – PIPELINE_STEPS 9 bước + PIPELINE_STEP_TABLE |
| C2 | **RFE + CV F1 chart** — uncomment block RFE + dual chart | 04_tab | ✅ DONE – N_FEATS=[5,10,15,20], dual subplot RFE+RF |
| C3 | **Báo cáo PDF ≥20 trang** — toàn bộ phần này chưa làm | Docs | ❌ CHƯA LÀM |

### 🟡 MODERATE — Ảnh hưởng đến "Xuất sắc" vs "Tốt"

| STT | Việc cần làm | File | Trạng thái |
|-----|-------------|------|-------------|
| M1 | **Thay hardcode markdown kết luận** bằng `print()` dynamic | 02_pre | ✅ DONE – 4 kết luận: Resize, ColorSpace, Norm, Augmentation |
| M2 | Thêm **dynamic kết luận KS test** per outlier method | 04_tab | ✅ DONE – `ks_method_results` dict, dynamic best_ks |
| M3 | **Violin plot đa thuộc tính** §2.2.3c | 04_tab | ✅ DONE – 3 cột đại diện side-by-side |
| M4 | **ARI metric** đánh giá cluster quality PCA 2D | 03_adv | ✅ DONE – KMeans + adjusted_rand_score, dynamic interpretation |
| M5 | **pHash near-duplicate** chạy toàn bộ dataset thay vì sample 2000 | 01_EDA | ⚠️ PENDING |
| M6 | Kết luận dynamic Color Space + quantile REPR_CLASSES | 02_pre / 03_adv | ✅ DONE (03_adv quantile REPR_CLASSES, 02_pre dynamic print) |

### 🟢 MINOR — Tăng điểm trình bày / giải thích

| STT | Việc cần làm | File | Ghi chú |
|----|-------------|------|---------|
| m1 | Hardcode values trong markdown kết luận 01_EDA (pixel/brightness) | 01_EDA | ⚠️ MINOR |
| m2 | Hardcode tổng kết 03_advanced (line ~687) → dynamic | 03_adv | ⚠️ MINOR |
| m3 | Kiểm tra `requirements.txt` đầy đủ version cụ thể | Root | ⚠️ MINOR |
| m4 | **Test Restart & Run All** toàn notebook trước nộp | All | Vẫn cần làm cuối cùng |
