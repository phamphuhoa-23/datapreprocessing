# So sánh số liệu: Report.tex vs Notebook Outputs

> Tạo ngày 2026-04-22. Mỗi mục ghi rõ **giá trị trong Report** và **giá trị thực tế từ cell output notebook**.
> - ✅ = Khớp (hoặc sai lệch làm tròn chấp nhận được)
> - ⚠️ = Sai lệch nhỏ (cần xem lại)
> - ❌ = Sai lệch nghiêm trọng (cần sửa ngay)

---

## PHẦN 1 — Ảnh (NB01 + NB02 + NB03)

### NB01 — EDA Ảnh

#### Pixel Mean
| Kênh | Report | Notebook (output) | Trạng thái |
|------|--------|-------------------|------------|
| R    | 93,8   | 93,84             | ✅ |
| G    | 97,1   | 97,12             | ✅ |
| B    | 87,6   | 87,59             | ✅ |

#### Kruskal-Wallis η² theo kênh màu
| Kênh | Report | Notebook (output) | Trạng thái |
|------|--------|-------------------|------------|
| R    | 0,500  | 0,500             | ✅ |
| G    | 0,437  | 0,437             | ✅ |
| B    | 0,455  | 0,455             | ✅ |

#### Duplicate Detection
| Chỉ số | Report | Notebook (output) | Trạng thái |
|--------|--------|-------------------|------------|
| Exact duplicates | 7 nhóm / 14 ảnh (0,05%) | 7 nhóm (14 ảnh), 0,05% | ✅ |
| Near-duplicates | "≈8 triệu cặp so sánh" (số cặp scan) | 1 cặp near-duplicate trong kết quả | ✅ (con số trong report là số cặp được scan, không phải kết quả) |

#### ANOVA — Brightness & Contrast ❌ SAI LỆCH NGHIÊM TRỌNG
| Chỉ số | Report (F / η²) | Notebook output (F / η²) | Trạng thái |
|--------|-----------------|--------------------------|------------|
| Brightness (mean L) | F = **38,37** / η² = **0,434** | F = **499,14** / η² = **0,449** | ❌ |
| Contrast (std L)    | F = **48,62** / η² = **0,492** | F = **560,48** / η² = **0,478** | ❌ |

> **Ghi chú**: Report đang dùng số liệu cũ. Notebook output cho F lớn hơn ~13 lần và η² cũng khác. Cả hai chỉ số F và η² đều cần cập nhật.

---

### NB02 — Preprocessing Ảnh

#### Resize Ablation — SSIM / PSNR / k-NN ❌ SAI LỆCH NGHIÊM TRỌNG (TẤT CẢ GIÁ TRỊ)

| Kích thước | SSIM (Report) | SSIM (NB output) | PSNR dB (Report) | PSNR dB (NB output) | k-NN Acc (Report) | k-NN Acc (NB output) |
|-----------|--------------|-----------------|-----------------|---------------------|------------------|---------------------|
| 32×32     | 0,459 ± 0,141 | **0,531 ± 0,176** | 22,86 ± 3,31 | **24,89 ± 4,74** | — | — |
| 64×64     | 0,640 ± 0,112 | **0,688 ± 0,132** | 25,29 ± 3,45 | **27,37 ± 4,69** | 0,2556 ± 0,0263 | **0,352 ± 0,028** |
| 128×128   | 0,847 ± 0,064 | **0,870 ± 0,069** | 29,20 ± 3,70 | **31,44 ± 4,91** | 0,2600 ± 0,0229 | **0,358 ± 0,029** |
| **224×224** | 0,947 ± 0,027 | **0,957 ± 0,026** | 33,64 ± 4,05 | **36,13 ± 5,30** | 0,2600 ± 0,0295 | **0,350 ± 0,027** |
| 256×256   | 1,000 ± 0,000 | 1,000 ± 0,000 | ∞ | ∞ | 0,2489 ± 0,0206 | **0,338 ± 0,026** |

> **Ghi chú**: Toàn bộ SSIM, PSNR và k-NN accuracy đều sai. k-NN accuracy trong report (~0.25-0.26) khác hẳn notebook output (~0.35). SSIM và PSNR cũng đều thấp hơn so với giá trị thực.

| Thống kê | Report | Notebook output | Trạng thái |
|----------|--------|-----------------|------------|
| ANOVA F (SSIM) | **1140,82** | **739,70** | ❌ |
| ANOVA η² (SSIM) | **0,739** | **0,623** | ❌ |
| Kết luận chọn 224×224 | "SSIM=0,947, mất mát ≤5,3%" | SSIM=0,957, mất mát ≤4,3% | ⚠️ (percentages differ) |

#### Color Space Ablation
| Không gian màu | PCA Var@50 (Report) | PCA Var@50 (NB output) | k-NN Acc (Report) | k-NN Acc (NB output) | Trạng thái |
|---------------|---------------------|------------------------|------------------|----------------------|------------|
| RGB           | 0,714               | **0,727**              | 0,0898 ± 0,0030  | **0,0889 ± 0,0058**  | ⚠️ |
| Grayscale     | 0,718               | **0,732**              | 0,0667 ± 0,0040  | **0,0698 ± 0,0071**  | ⚠️ |
| HSV           | 0,661               | **0,669**              | 0,1138 ± 0,0101  | **0,1133 ± 0,0037**  | ⚠️ |
| CIE Lab       | 0,739               | **0,753**              | 0,0978 ± 0,0044  | **0,0916 ± 0,0086**  | ⚠️ |

> **Ghi chú**: PCA variance đều thấp hơn thực tế ~0.013-0.014. k-NN của Lab sai lệch đáng kể (0.0978 vs 0.0916).

#### Normalization Ablation
| Phương pháp | k-NN Acc (Report) | k-NN Acc (NB output) | KS D-stat (Report) | KS D-stat (NB output) | Trạng thái |
|-------------|------------------|----------------------|--------------------|-----------------------|------------|
| Original    | 0,0898 ± 0,0030  | **0,0889 ± 0,0058**  | 0,0000             | 0,0000                | ⚠️ |
| Min-Max [0,1]    | 0,0898 ± 0,0030  | **0,0889 ± 0,0058**  | 0,9980             | **0,9981**            | ⚠️ |
| Min-Max [-1,1]   | 0,0898 ± 0,0030  | **0,0889 ± 0,0058**  | 0,9980             | **0,9981**            | ⚠️ |
| **Z-score global** | 0,1004 ± 0,0060 | **0,1062 ± 0,0074**  | 0,9961             | **0,9963**            | ❌ |
| Z-score per-ch  | 0,0440 ± 0,0087  | **0,0498 ± 0,0067**  | 0,9961             | **0,9962**            | ⚠️ |

| Thống kê | Report | Notebook output | Trạng thái |
|----------|--------|-----------------|------------|
| Wilcoxon p (Z-score global vs Original) | 0,0625 | 0,0625 | ✅ |
| Cohen's d (Z-score global vs per-ch) | **1,730** | **1,315** | ❌ |

#### Augmentation Ablation ❌ SAI LỆCH NGHIÊM TRỌNG (TẤT CẢ GIÁ TRỊ)

| Kỹ thuật | k-NN Acc (Report) | k-NN Acc (NB output) | Wilcoxon p (Report) | Wilcoxon p (NB output) | Trạng thái |
|---------|------------------|----------------------|---------------------|------------------------|------------|
| Baseline (no aug) | 0,0948 ± 0,0151 | **0,0978 ± 0,0130** | ref | ref | ⚠️ |
| H-Flip      | 0,0941 ± 0,0095 | **0,0859 ± 0,0103** | 0,750 | **0,1250** | ❌ |
| **V-Flip**  | **0,0956 ± 0,0108** | **0,0911 ± 0,0060** | 0,750 | **0,3125** | ❌ |
| Rotation    | 0,0933 ± 0,0101 | **0,0952 ± 0,0091** | 0,625 | **0,6250** | ⚠️ |
| Random Crop | 0,0904 ± 0,0095 | **0,0930 ± 0,0113** | 0,625 | **0,8125** | ⚠️ |
| Gaussian Noise | 0,0937 ± 0,0091 | **0,0993 ± 0,0113** | 1,000 | **0,7500** | ❌ |
| Brightness/Contrast | 0,0778 ± 0,0126 | **0,0763 ± 0,0092** | 0,125 | **0,1250** | ⚠️ |

> **Ghi chú QUAN TRỌNG**: Report nói kỹ thuật tốt nhất là **V-Flip (0,0956)**, nhưng thực tế notebook output cho thấy kỹ thuật tốt nhất là **Gaussian Noise (0,0993)**. Đây là kết luận sai trong report. Tất cả k-NN accuracy và Wilcoxon p đều khác.

---

### NB03 — Advanced Image (PCA / Edge Detection)

| Chỉ số | Report | Notebook output | Trạng thái |
|--------|--------|-----------------|------------|
| Max cumulative variance tại n=800 | 88,2% | 88,2% | ✅ |
| n₉₀, n₉₅, n₉₉ | ">800" | ">800" | ✅ |
| ARI (PCA 2D, 45 lớp) | "thấp" (không nêu số) | 0,022 | ✅ (nhất quán) |
| Sobel: lớp Edge Density cao nhất | dense_residential, chaparral | dense_residential (0.793), chaparral (0.799) | ✅ |
| Edge Detection: ANOVA p | p≈0 | p≈0 | ✅ |

---

## PHẦN 2 — Bảng (NB04 + NB05)

### NB04 — EDA Bảng

#### Normality Test
| Chỉ số | Report | Notebook output | Trạng thái |
|--------|--------|-----------------|------------|
| Cột không chuẩn / tổng | 399/400 | 399/400 | ✅ |
| Cột chuẩn duy nhất | id_25, p=0,627 | id_25, K²=0,9335, p=0,627049 | ✅ |
| TransactionAmt K² | 25.679 | *Không có trong cell output (chỉ trong markdown)* | ⚠️ Không thể xác minh |

#### Missing Rate theo nhóm
| Nhóm | Report | Notebook output | Trạng thái |
|------|--------|-----------------|------------|
| Identity | 84,5% | 84,5% | ✅ |
| D-timedelta | 58,2% | 58,2% | ✅ |
| M-match | 49,9% | 49,9% | ✅ |
| Email | 46,4% | 46,4% | ✅ |
| V-Vesta | 43,0% | 43,0% | ✅ |
| Card | 0,5% | 0,5% | ✅ |
| Transaction | 0% | 0,0% | ✅ |

#### Fraud Rate
| Chỉ số | Report | Notebook output (NB05) | Trạng thái |
|--------|--------|------------------------|------------|
| Tỉ lệ fraud | ~3,5% | 3,50% (20,663 / 590,540) | ✅ |
| Imbalance ratio | ~27,6:1 | 27,6× | ✅ |

---

### NB05 — Preprocessing Bảng

#### Imputation RMSE
| Chiến lược | Report | Notebook output | Trạng thái |
|-----------|--------|-----------------|------------|
| Mean      | 120,03 | 120,0285 | ✅ |
| Median    | 124,50 | 124,4956 | ✅ |
| Mode      | 154,76 | 154,7576 | ✅ |
| kNN-3     | 112,23 | 112,2291 | ✅ |
| kNN-5     | 106,84 | 106,8398 | ✅ |
| kNN-10    | 102,40 | 102,3987 | ✅ |
| MICE      | 97,35  | 97,3548  | ✅ |

#### Outlier Detection Rates ❌ SAI LỆCH
| Phương pháp | Report | Notebook output | Trạng thái |
|-------------|--------|-----------------|------------|
| IQR         | **54,38%** | **53,77%** | ⚠️ |
| Z-score     | 10,11% | 10,11% | ✅ |
| IF c=0,01   | 1,00%  | 1,00% | ✅ |
| IF c=0,05   | 5,00%  | 5,00% | ✅ |
| IF c=0,10   | 10,00% | 10,00% | ✅ |
| **LOF k=10** | **9,28%** | **4,87%** | ❌ |
| **LOF k=20** | **8,17%** | **4,26%** | ❌ |
| **LOF k=50** | **8,13%** | **4,71%** | ❌ |
| DBSCAN      | **0,71%** | **0,75%** | ⚠️ |

> **Ghi chú**: LOF rates trong report (~8-9%) gần gấp đôi so với notebook output (~4-5%). Đây là sai lệch đáng kể, có vẻ là số liệu từ lần chạy cũ.

#### Scaling — Levene F statistics ❌ SAI LỆCH NGHIÊM TRỌNG

| Phương pháp | Report (Levene F) | Notebook output (Levene F) | Trạng thái |
|-------------|-----------------|---------------------------|------------|
| Min-Max [0,1]    | **1.653,08** | **2.278,95** | ❌ |
| Z-score          | **1.626,30** | **2.106,98** | ❌ |
| **Robust Scaling** | **1.527,92** | **1.979,22** | ❌ |
| Quantile-Uniform | **2.293,60** | **2.963,73** | ❌ |
| Quantile-Normal  | **2.152,29** | **1.341,16** | ❌ |

> **Ghi chú QUAN TRỌNG**: Tất cả 5 giá trị Levene F đều sai. Đặc biệt: Report nói **Robust Scaling có F thấp nhất (1.527,92)** → chứng minh Robust tốt nhất. Nhưng notebook output cho thấy **Quantile-Normal có F thấp nhất (1.341,16)**, còn Robust có F = 1.979,22. Thứ tự ranking bị thay đổi!

#### Categorical Encoding — VIF ❌ SAI LỆCH (One-Hot)
| Phương pháp | Report (mean VIF / max VIF) | Notebook output (mean VIF / max VIF) | Trạng thái |
|-------------|---------------------------|--------------------------------------|------------|
| **One-Hot** | **9,99×10¹¹ / 1,14×10¹³** | **5,85×10¹³ / 2,25×10¹⁵** | ❌ |
| Ordinal     | 9,21 / 15,91               | 9,21 / 15,91 | ✅ |
| Target (CV) | 7,38 / 11,54               | 7,38 / 11,54 | ✅ |
| Binary      | 9,94 / 123,33              | 9,94 / 123,33 | ✅ |
| Frequency   | 9,14 / 15,16               | 9,14 / 15,16 | ✅ |

> **Ghi chú**: One-Hot VIF trong report thấp hơn thực tế ~60 lần (mean) và ~200 lần (max).

#### Feature Selection
| Chỉ số | Report | Notebook output | Trạng thái |
|--------|--------|-----------------|------------|
| Số features chọn (Best Tầng 2) | 30 | 30 | ✅ |
| AUC (Best Tầng 2) | 0,884 | **0,8833** | ✅ (làm tròn) |
| F1-macro (Best Tầng 2) | 0,805 | **0,8059** | ✅ (làm tròn) |

#### RFE Results ⚠️ Giá trị k=10 và k=20 có vẻ bị đổi chỗ

| n_features | Report (F1) | Notebook output (F1) | Trạng thái |
|-----------|------------|----------------------|------------|
| 5  | 0,7214 | 0,7214 | ✅ |
| **10** | **0,7502** | **0,7529** | ❌ |
| **20** | **0,7506** | **0,7502** | ❌ |
| **30** | **0,7612** | **0,7603** | ⚠️ |
| 50 | 0,7571 | 0,7568 | ⚠️ |

> **Ghi chú**: Giá trị k=10 (0,7502) và k=20 (0,7506) trong report trông giống bị **đổi chỗ** cho nhau so với notebook output (k=10: 0,7529, k=20: 0,7502).

#### Class Imbalance ❌ SAI LỆCH CỰC KỲ NGHIÊM TRỌNG

| Chiến lược | Precision (Report) | Precision (NB) | Recall (Report) | Recall (NB) | F1-macro (Report) | F1-macro (NB) | AUC-ROC (Report) | AUC-ROC (NB) |
|-----------|-------------------|----------------|-----------------|-------------|-------------------|---------------|-----------------|--------------|
| Không resampling | **0,1158** | **0,7876** | **0,1783** | **0,2342** | **0,5502** | **0,6731** | **0,4828** | **0,8513** |
| SMOTE    | **0,0468** | **0,1252** | **0,8747** | **0,7450** | **0,3048** | **0,5528** | **0,7376** | **0,8560** |
| ADASYN   | **0,0466** | **0,1035** | **0,8756** | **0,7820** | **0,3031** | **0,5194** | **0,7274** | **0,8447** |
| RUS      | **0,0451** | **0,1269** | **0,9030** | **0,7467** | **0,2774** | **0,5549** | **0,7214** | **0,8545** |

> **Ghi chú**: Đây là sai lệch nghiêm trọng nhất trong toàn bộ report. **Tất cả 16 giá trị đều hoàn toàn khác**. Report đang dùng số liệu từ một lần chạy cũ với cấu hình khác (có vẻ như không có train/val split đúng, hoặc model/data khác). Đặc biệt:
> - No Resampling: Report AUC=0,4828 (dưới random!) vs NB AUC=0,8513
> - SMOTE: Report Recall=0,875 vs NB Recall=0,745
> - Toàn bộ kết luận nghiệp vụ trong phần thảo luận cũng bị ảnh hưởng bởi sai lệch này

---

## PHẦN 3 — Văn bản (NB06)

### NB06 — Text Preprocessing

#### Zipf Distribution
| Chỉ số | Report | Notebook output | Trạng thái |
|--------|--------|-----------------|------------|
| α (toàn corpus) | ~1,63 | 1,6306 | ✅ |
| α (Supported) | 1,505 | 1,505 | ✅ |
| α (Hallucinated) | 1,534 | 1,534 | ✅ |
| R² | 0,936 | 0,9357 | ✅ |

#### Pipeline 9 bước — Vocab Size ⚠️ Chênh lệch nhỏ nhất quán (+3 từ vựng ở steps 0-6)

| Bước | Vocab (Report) | Vocab (NB output) | ΔVocab % (Report) | ΔVocab % (NB) | Mean Tokens (Report) | Mean Tokens (NB) |
|------|--------------|-------------------|-------------------|---------------|---------------------|-----------------|
| 0 raw       | **40.616** | **40.613** | 0,00 | 0,00 | 150,0 | 150,0 |
| 1 lowercase | **34.563** | **34.560** | -14,90 | -14,90 | 149,2 | 149,2 |
| 2 remove_html | **34.557** | **34.554** | -0,02 | -0,02 | 149,2 | 149,2 |
| 3 remove_url | **34.538** | **34.535** | -0,05 | -0,05 | 149,2 | 149,2 |
| 4 remove_email | **34.537** | **34.534** | -0,00 | -0,00 | 149,2 | 149,2 |
| 5 remove_mention | **34.535** | **34.532** | -0,01 | -0,01 | 149,2 | 149,2 |
| 6 remove_hashtag | **34.513** | **34.510** | -0,06 | -0,06 | 149,2 | 149,2 |
| 7 remove_number | 31.558 | 31.558 | **-8,56** | **-8,55** | 147,8 | 147,8 |
| 8 remove_punct | 31.946 | 31.946 | +1,23 | +1,23 | 127,4 | 127,4 |
| 9 normalize_ws | 31.946 | 31.946 | 0,00 | 0,00 | 127,4 | 127,4 |

> **Ghi chú**: Có chênh lệch nhất quán +3 token trong steps 0-6 (report số cao hơn 3). Đây có thể do phiên bản dataset nhỏ khác. Từ step 7 trở đi khớp hoàn toàn. Sai lệch -8.56% vs -8.55% ở step 7 là không đáng kể (làm tròn).

#### Tokenization Comparison
| Chiến lược | Vocab (Report) | Vocab (NB) | Mean Tokens (R) | Mean Tokens (NB) | OOV (R) | OOV (NB) | Trạng thái |
|-----------|--------------|------------|----------------|-----------------|---------|---------|------------|
| Word-level | 31.946 | 31.946 | 127,37 | 127,367510 | 0,0264 | 0,0264 | ✅ |
| Sentence   | — | — | 7,45 | 7,446824 | N/A | N/A | ✅ |
| Character  | 27 | 27 | 764,75 | 764,745812 | 0,0000 | 0,0000 | ✅ |
| Subword BPE | 9.695 | 9.695 | 138,45 | 138,451152 | 0,0023 | 0,0023 | ✅ |

#### Stop Words
| Chỉ số | Report | Notebook output | Trạng thái |
|--------|--------|-----------------|------------|
| Số stop words NLTK | 198 | 198 | ✅ |
| NB F1 có stop words | 0,6831 ± 0,0052 | 0,6831 ± 0,0052 | ✅ |
| NB F1 không stop words | 0,6832 ± 0,0057 | 0,6832 ± 0,0057 | ✅ |
| Wilcoxon W | 7,00 | 7,00 | ✅ |
| Wilcoxon p | 1,0000 | 1,0000 | ✅ |
| Cohen's d | 0,0173 | 0,0173 | ✅ |

#### Stemming & Lemmatization
| Phương pháp | Collision Rate (Report) | Collision Rate (NB) | LR F1 (Report) | LR F1 (NB) | Trạng thái |
|-------------|------------------------|---------------------|---------------|------------|------------|
| Porter Stemmer | 35,43% | 35,43% | 0,7274 | 0,7274 | ✅ |
| Snowball Stemmer | 35,68% | 35,68% | 0,7284 | 0,7284 | ✅ |
| WordNet Lemmatizer | 13,20% | 13,20% | 0,7285 | 0,7285 | ✅ |

#### Vectorization
| Biểu diễn | Dims (R) | Sparsity (R) | Silhouette (R) | SVM F1 (R) | Silhouette (NB) | SVM F1 (NB) | Trạng thái |
|----------|---------|-------------|---------------|-----------|----------------|------------|------------|
| BoW unigram | 10.000 | 99,45% | 0,0187 | — | 0,0187 | — | ✅ |
| TF-IDF unigram | 10.000 | 99,45% | 0,0039 | 0,7223 | 0,0039 | 0,7223 | ✅ |
| TF-IDF bigram | 10.000 | >99% | 0,0034 | — | 0,0034 | — | ✅ |
| TF-IDF trigram | 10.000 | >99% | 0,0033 | — | 0,0033 | — | ✅ |
| Word2Vec | 100 | 0% | 0,0779 | **0,6965** | 0,0780 | **0,6953** | ⚠️ |
| Sentence Transformer | 384 | 0% | 0,0826 | 0,6948 | 0,0826 | 0,6948 | ✅ |

> **Ghi chú**: Word2Vec SVM F1 sai nhẹ: Report 0,6965 vs NB 0,6953.

#### Sentence Transformer (Advanced)
| Chỉ số | Report | Notebook output | Trạng thái |
|--------|--------|-----------------|------------|
| TF-IDF K-Means silhouette | 0,1834 | 0,1834 | ✅ |
| ST K-Means silhouette | 0,0826 | 0,0826 | ✅ |
| Linear SVM TF-IDF F1 | 0,7223 | 0,7223 | ✅ |
| Linear SVM Word2Vec F1 | **0,6965** | **0,6953** | ⚠️ |
| Linear SVM ST F1 | 0,6948 | 0,6948 | ✅ |

#### Classification Leaderboard
| Mô hình | F1-macro (Report) | F1-macro (NB) | Trạng thái |
|---------|------------------|---------------|------------|
| NB + BoW (có stop words) | 0,6831 | 0,6831 | ✅ |
| NB + BoW (không stop words) | 0,6832 | 0,6832 | ✅ |
| NB + TF-IDF unigram | 0,6825 | 0,6825 | ✅ |
| NB + TF-IDF bigram | 0,6838 | 0,6838 | ✅ |
| LR + TF-IDF unigram | 0,7295 | 0,7295 | ✅ |
| **LR + TF-IDF bigram** | **0,7331** | **0,7331** | ✅ |
| SVM + TF-IDF | 0,7223 | 0,7223 | ✅ |
| SVM + Word2Vec | **0,6965** | **0,6953** | ⚠️ |
| SVM + ST | 0,6948 | 0,6948 | ✅ |

#### Mann-Whitney & TTR
| Chỉ số | Report | Notebook output | Trạng thái |
|--------|--------|-----------------|------------|
| Effect size r (range) | 0,22–0,28 | 0,2225–0,2775 | ✅ |
| Số mẫu n | 17.790 | 17.790 | ✅ |
| TTR corpus | 0,0125 | 0,0125 | ✅ |
| TTR Supported | 0,0215 | 0,0215 | ✅ |
| TTR Hallucinated | 0,0202 | 0,0202 | ✅ |

---

## TÓM TẮT CÁC SAI LỆCH CẦN SỬA

### ❌ Nghiêm trọng (cần sửa ngay):

| # | Vị trí trong Report | Nội dung sai | Giá trị đúng (NB output) |
|---|---------------------|-------------|--------------------------|
| 1 | §3.2.4 (Brightness/Contrast ANOVA, NB01) | Brightness: F=38,37 / η²=0,434 | **F=499,14 / η²=0,449** |
| 2 | §3.2.4 (Brightness/Contrast ANOVA, NB01) | Contrast: F=48,62 / η²=0,492 | **F=560,48 / η²=0,478** |
| 3 | §3.3.1 Resize table (NB02) | Toàn bộ SSIM / PSNR / k-NN accuracy | Xem bảng chi tiết |
| 4 | §3.3.1 ANOVA resize (NB02) | F=1140,82, η²=0,739 | **F=739,70, η²=0,623** |
| 5 | §3.3.4 Augmentation table (NB02) | Best technique: V-Flip (0,0956) | **Best: Gaussian Noise (0,0993)** |
| 6 | §3.3.4 Augmentation all values (NB02) | Tất cả k-NN và Wilcoxon p | Xem bảng chi tiết |
| 7 | §4.3.2 Scaling Levene F (NB05) | Tất cả 5 Levene F statistic | Xem bảng chi tiết |
| 8 | §4.3.2 Scaling — Robust có F nhỏ nhất | Report: Robust=1.527,92 (nhỏ nhất) | **Nhỏ nhất là Quantile-Normal=1.341,16** |
| 9 | §4.3.4 One-Hot Encoding VIF (NB05) | mean=9,99×10¹¹ / max=1,14×10¹³ | **mean≈5,85×10¹³ / max≈2,25×10¹⁵** |
| 10 | §4.3.5 Class Imbalance table (NB05) | **TẤT CẢ 16 giá trị đều sai** | Xem bảng chi tiết |

### ⚠️ Cần xem lại (sai lệch nhỏ hơn):

| # | Vị trí | Nội dung | Report | NB output |
|---|--------|---------|--------|-----------|
| 11 | §3.3.2 Color space PCA (NB02) | RGB PCA Var@50 | 0,714 | 0,727 |
| 12 | §3.3.2 Color space PCA (NB02) | Grayscale PCA Var@50 | 0,718 | 0,732 |
| 13 | §3.3.2 Color space PCA (NB02) | Lab PCA Var@50 | 0,739 | 0,753 |
| 14 | §3.3.2 Color space k-NN (NB02) | Lab k-NN accuracy | 0,0978 ± 0,0044 | 0,0916 ± 0,0086 |
| 15 | §3.3.3 Normalization (NB02) | Z-score global k-NN | 0,1004 ± 0,0060 | 0,1062 ± 0,0074 |
| 16 | §3.3.3 Normalization (NB02) | Cohen's d | 1,730 | 1,315 |
| 17 | §3.3.3 Normalization (NB02) | Z-score per-ch k-NN | 0,0440 ± 0,0087 | 0,0498 ± 0,0067 |
| 18 | §4.3.1 Outlier LOF (NB05) | LOF k=10 rate | 9,28% | 4,87% |
| 19 | §4.3.1 Outlier LOF (NB05) | LOF k=20 rate | 8,17% | 4,26% |
| 20 | §4.3.1 Outlier LOF (NB05) | LOF k=50 rate | 8,13% | 4,71% |
| 21 | §4.3.1 Outlier IQR (NB05) | IQR rate | 54,38% | 53,77% |
| 22 | §4.3.4 RFE (NB05) | k=10 F1 | 0,7502 | 0,7529 |
| 23 | §4.3.4 RFE (NB05) | k=20 F1 | 0,7506 | 0,7502 |
| 24 | §5.3.4 Word2Vec SVM F1 (NB06) | Word2Vec SVM F1 | 0,6965 | 0,6953 |
| 25 | §5.2.2 Pipeline vocab counts (NB06) | Raw vocab và steps 0-6 đều cao hơn 3 | 40.616 | 40.613 |
