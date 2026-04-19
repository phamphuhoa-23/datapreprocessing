# FINALFEEDBACK — Highlights & Notes trích xuất từ PDF

> File: `CSC14004 - Data Mining FINALFEEDBACK.pdf`  
> Tổng trang: 12

---
## Trang 3

**[Highlight 🟢 GREEN]**
> Sinh viên chọn một tập dữliệu ảnh. Tập dữliệu phải có ít nhất 5 lớp và ít nhất 5.000 ảnh.
Một sốtập dữliệu gợi ý như sau:

---
## Trang 4

**[Highlight 🟢 GREEN]**
> Tính và trực quan hóa phân phối giá trịpixel trên toàn tập (histogram, KDE) theo từng
kênh màu.
  💬 **NOTE:** check lại kiểm định, viết nhiều quá hăm hiểu rồi

Kết luận của bạn **về cơ bản là chính xác**, nhưng có vài điểm cần làm rõ hoặc bổ sung để chặt chẽ hơn.

---

**Những điểm đúng:**

- Bác bỏ $H_0$ (các lớp có phân phối pixel mean giống nhau) ở cả 3 kênh — hoàn toàn đúng, cả ANOVA lẫn Kruskal-Wallis đều cho $p \approx 0$.
- Eta² "lớn" theo ngưỡng Cohen ($\eta^2 \geq 0.14$) — đúng, cả 3 kênh đều vượt xa ngưỡng này.
- Kênh R có $\eta^2$ cao nhất (0.500) — đúng.
- Levene reject → dùng Kruskal-Wallis làm kiểm định chính thay vì ANOVA — đúng về mặt logic, vì ANOVA giả định homoscedasticity (phương sai đồng nhất giữa các nhóm), khi Levene reject thì giả định này bị vi phạm.
- freeway vs river ở kênh R là `ns` — đúng theo output.

---

**Một số điểm cần làm rõ để tránh hiểu nhầm:**

**1. "Lớp ảnh giải thích 43–50% variance của pixel mean"** — cần nói rõ hơn.

$\eta^2$ ở đây được tính trên **per-class means** (tức là mỗi sample là mean của một ảnh thuộc lớp đó), **không phải trên raw pixel values**. Nếu viết báo cáo, nên nói: *"lớp ảnh giải thích 43–50% variance của pixel channel mean tính theo từng ảnh."*

**2. ANOVA "cho kết quả tương đồng"** — diễn đạt hơi mơ hồ.

Thực ra ANOVA và KW đều reject $H_0$, nhưng chúng kiểm định những thứ hơi khác nhau:
- ANOVA kiểm định sự khác biệt về **mean**.
- KW kiểm định sự khác biệt về **phân phối rank** (robust hơn với outlier và non-normality).

Khi cả hai cùng reject, điều đó củng cố kết luận, nhưng không nên nói "tương đồng" như thể chúng là hai phiên bản của cùng một test.

**3. freeway vs river ở kênh R là ns — nhưng kênh G và B thì khác.**

Đây là chi tiết thú vị mà kết luận bỏ qua: ở kênh G và B, freeway vs river **có** significant ($p\_bonf = 1.54 \times 10^{-17}$ và $1.37 \times 10^{-8}$). Tức là 2 lớp này chỉ "trùng nhau" về kênh R, còn G và B vẫn phân biệt được. Điều này gợi ý rằng nếu dùng cả 3 kênh kết hợp thì freeway và river vẫn phân biệt được — kết luận nên nêu điểm này thay vì chỉ nói "2 lớp có pixel mean R gần nhau."

---

**Tóm lại:** Kết luận đúng về mặt thống kê, nhưng nếu dùng cho báo cáo chính thức thì nên bổ sung điểm 1 và 3 ở trên để tránh overstatement hoặc bỏ sót thông tin quan trọng.

**[Highlight 🟢 GREEN]**
> Phân tích mất cân bằng lớp (class imbalance): tính tỉlệmỗi lớp, kiểm tra xem có lớp
nào chiếm tỉlệvượt mức 3× so với lớp ít nhất không.

**[Highlight 🟢 GREEN]**
> Phát hiện ảnh trùng lặp hoặc gần trùng bằng hàm băm perceptual hash (pHash). Báo
cáo tỉlệtrùng lặp và xửlý chúng.
  💬 **NOTE:** near dup chưa xóa

**[Highlight 🟢 GREEN]**
> Phân tích độtương phản và độsáng toàn cục: tính mean intensity và standard deviation
theo lớp, thểhiện qua boxplot phân lớp.
  💬 **NOTE:** Kết luận **đúng về cơ bản**, nhưng có một số điểm cần xem xét kỹ hơn.

---

**Những điểm đúng:**

- Bác bỏ $H_0$ ở cả Brightness và Contrast — đúng.
- Contrast có $\eta^2$ cao hơn (0.478 vs 0.449) — đúng, nhưng khoảng cách chỉ là 0.029, khá nhỏ. Nên diễn đạt thận trọng hơn: *"Contrast có effect size nhỉnh hơn một chút, nhưng cả hai đều ở mức large."*
- Levene reject → dùng KW làm kiểm định chính — đúng, lý do tương tự phần R/G/B trước.

---

**Điểm cần cẩn thận:**

**"Quy luật lớp tự nhiên vs lớp cấu trúc phức tạp"** — đây là điểm **bạn cần kiểm tra lại bằng data thực tế**, không nên viết như một kết luận chắc chắn nếu chưa có con số backing.

Cụ thể:
- Kết luận này **không thể rút ra từ ANOVA/KW/η²** — những test này chỉ nói "các lớp khác nhau có ý nghĩa thống kê", không nói lớp nào cao lớp nào thấp.
- Muốn kết luận "forest có contrast thấp hơn harbor", bạn cần **post-hoc test** (như phần R/G/B bạn đã làm với Mann-Whitney) hoặc ít nhất là **so sánh mean/median theo lớp trực tiếp**.
- Nếu bạn chỉ nhìn scatter plot mà kết luận, thì nên viết: *"Scatter plot gợi ý xu hướng..."* thay vì *"Quy luật: ..."*

---

**Điểm còn thiếu (nếu muốn báo cáo đầy đủ):**

Bạn không có post-hoc ở đây (khác với phần pixel R/G/B), nên không biết **cặp lớp nào cụ thể khác nhau**. Nếu mục tiêu là chứng minh 2 feature này hữu ích để phân loại, thì post-hoc hoặc visualization phân phối theo lớp sẽ thuyết phục hơn là chỉ dừng ở KW.

---

**Tóm lại:** Phần thống kê formal là đúng. Phần "quy luật" cần được support bởi data cụ thể hơn, không nên viết như kết luận chắc chắn nếu chỉ dựa trên quan sát scatter plot.

**[Highlight 🟢 GREEN]**
> Thay đổi kích thước và chất lượng ảnh: Resize vềít nhất 3 kích thước khác nhau
(ví dụ: 32 × 32, 64 × 64, 128 × 128). Với mỗi kích thước, tính chỉsốSSIM (Structural
Similarity Index) và PSNR so với ảnh gốc đểđịnh lượng mức độmất mát thông tin.
Vẽđường cong SSIM theo kích thước và biện hộcho kích thước được chọn.

**[Highlight 🟡 YELLOW]**
> Chuyển đổi không gian màu: Chuyển đổi và so sánh ít nhất 3 không gian màu:
RGB, Grayscale, HSV, và LAB. Với mỗi không gian màu, tính phương sai giải thích
(explained variance) theo PCA với k = 50 thành phần. Thảo luận không gian màu nào
bảo toàn thông tin tốt nhất cho bài toán phân loại.
  💬 **NOTE:** chưa tính phương sai giải thích cho 50 thành phần

**[Highlight 🟢 GREEN]**
> Chuẩn hóa: Cài đặt và so sánh 4 phương pháp: (i) Min-Max [0, 1]; (ii) Min-Max
[−1, 1]; (iii) Z-score toàn tập; (iv) Z-score theo từng kênh (per-channel). Dùng kiểm
định Kolmogorov-Smirnov (KS test) đểđánh giá sựkhác biệt phân phối trước và sau
chuẩn hóa. Báo cáo p-value và diễn giải ý nghĩa thống kê.

**[Highlight 🟢 GREEN]**
> Tăng cường dữliệu (Data Augmentation): [Bắt buộc] Cài đặt pipeline augmentation
gồm ít nhất 5 phép biến đổi (lật ngang/dọc, xoay, cắt ngẫu nhiên, thêm nhiễu Gaussian,
điều chỉnh độsáng/tương phản). Sau đó, đánh giá tác động của augmentation đến phân
phối đặc trưng bằng cách so sánh t-SNE visualization của tập gốc và tập đã augment.
  💬 **NOTE:** ghi thêm markdown chỗ augment là chọn 2 trong 6 phép, không cần vẽ incremental lớp, với lại nó đang dính nhau qua thì chỉnh lại perlexity một chút

**[Highlight 🟡 YELLOW]**
> [Nâng cao] Phân tích PCA trên không gian đặc trưng ảnh: Chiếu toàn bộtập dữ
liệu lên không gian PCA. Vẽscree plot và xác định sốthành phần cần thiết đểgiải
thích 90%, 95%, và 99% phương sai. Trực quan hóa 2D/3D bằng PCA và t-SNE, tô
màu theo nhãn lớp. Thảo luận mức độtách biệt giữa các lớp.
  💬 **NOTE:** VẪN CHƯA XÁC ĐỊNH SỐ THÀNH PHẦN CẦN THIẾT 90 95 99 (VÌ NÓ LỚN HƠN 200 RẤT NHIỀU)

**[Highlight 🟡 YELLOW]**
> [Nâng cao] Phát hiện cạnh và phân tích đặc trưng cục bộ: Áp dụng Sobel, Prewitt
và Canny với ít nhất 2 bộsiêu tham sốmỗi loại. Tính Edge Density (tỉlệpixel cạnh /
tổng pixel) theo lớp và kiểm định sựkhác biệt giữa các lớp bằng ANOVA một chiều.
Kết luận liệu thông tin cạnh có phân biệt được các lớp hay không.
  💬 **NOTE:** VẼ CÁI TƯƠNG QUAN SOBEL CANNY LÀM GÌ KHÔNG RÕ, ĐỌC LẠI ĐỀ BÀI

---
## Trang 5

**[Highlight 🟡 YELLOW]**
> Kiểm tra phân phối: Với mỗi thuộc tính số, kiểm định Shapiro-Wilk (nếu n ≤5000)
hoặc D’Agostino-Pearson (nếu n > 5000). Phân loại từng thuộc tính là phân phối
chuẩn hay không chuẩn dựa trên p-value, và sửdụng kết quảnày đểlựa chọn phương
pháp chuẩn hóa phù hợp.
  💬 **NOTE:** KHÔNG CHẠY FULL DATASET

MARKDOWN nhận xét lệch so với output cell

**[Highlight 🟢 GREEN]**
> Phân tích tương quan đa biến: Vẽheatmap tương quan Pearson và Spearman. Xác
định các cặp thuộc tính có khảnăng đa cộng tuyến mạnh (ví dụ: |r| > 0.9) và đềxuất
xửlý.
  💬 **NOTE:** VẪN LẤY SAMPLES

**[Highlight 🟢 GREEN]**
> Phân tích giá trịthiếu: Trực quan hóa ma trận thiếu dữliệu (missing data matrix)
bằng thư viện missingno. Kiểm định giảthuyết MCAR bằng Little’s MCAR test.
Phân loại cơ chếthiếu dữliệu (MCAR/MAR/MNAR) và giải thích.
  💬 **NOTE:** một đống cell phân tích tương quan gì đó, không rõ công dụng

**[Highlight 🟢 GREEN]**
> Xửlý giá trịthiếu có kiểm soát: Cài đặt 5 chiến lược điền khuyết: trung bình, trung
vị, mode, k-NN imputation (ví dụ: k ∈{3, 5, 10}), và MICE (Multiple Imputation
by Chained Equations). Với mỗi chiến lược, tạo nhân tạo 10% giá trịthiếu bổsung
(MCAR) và tính RMSE điền khuyết đểđánh giá độchính xác. Trình bày bảng so
sánh tất cảchiến lược và lựa chọn chiến lược tốt nhất có lý giải.
  💬 **NOTE:** số liệu cho thấy MEAN tốt hơn MEDIAN khi điền nhưng chọn median ???

**[Highlight 🟢 GREEN]**
> Phát hiện và xửlý ngoại lai bằng nhiều kỹthuật: [Bắt buộc] Cài đặt và so sánh 4
phương pháp phát hiện ngoại lai sau:

---
## Trang 6

**[Highlight 🟢 GREEN]**
> Chuẩn hóa dữliệu có kiểm định: Áp dụng Min-Max, Z-score, Robust Scaling và
Quantile Transform (uniform và normal output). Sau mỗi phương pháp, có thểsửdụng
kiểm định Levene’s test đểđánh giá sựđồng nhất phương sai (homoscedasticity). Thử
trực quan hóa phân phối bằng violin plot theo từng thuộc tính và đánh giá.

**[Highlight 🟡 YELLOW]**
> Mã hóa biến phân loại nâng cao: Ngoài One-Hot và Ordinal encoding cơ bản, sinh
viên phải cài đặt và so sánh thêm:
• Target Encoding (mean encoding) với cross-validation đểtránh target leakage.
• Binary Encoding cho thuộc tính có nhiều giá trị(high-cardinality, > 20 giá trị).
• Frequency Encoding.
Với từng phương pháp, đo tỉlệphương sai giải thích (variance inflation factor) đểphát
hiện đa cộng tuyến mới phát sinh.
  💬 **NOTE:** **Đúng đề bài**, đủ các yêu cầu cốt lõi. Nhưng có vài điểm cần lưu ý.

---

**Những gì đã làm đúng:**

- Cài đặt đủ 5 phương pháp: OHE, Ordinal, Target Encoding (CV), Binary, Frequency ✓
- Target Encoding dùng 5-fold CV để tránh target leakage ✓
- Binary Encoding áp dụng cho high-cardinality (>20) ✓
- Đo VIF sau mỗi phương pháp trong phần demo ✓
- Có bảng so sánh và visualization ✓

---

**Điểm cần cẩn thận khi nộp/trình bày:**

**1. VIF của OHE bị astronomically cao ($\approx 10^{12}$)**

Bạn có nhận xét *"category_encoders tự động xử lý dummy variable trap"* — nhưng VIF vẫn ra $10^{12}$, tức là **multicollinearity vẫn xảy ra nghiêm trọng**. Hai trường hợp có thể xảy ra:

- `use_cat_names=True` không drop một cột (không dùng `drop='first'` convention), hoặc
- Dù drop một cột, các cột OHE vẫn near-perfectly correlated do structure của data.

Nếu thầy hỏi tại sao VIF OHE vẫn cao dù đã "xử lý", bạn cần giải thích rõ hơn thay vì chỉ nói "tự động xử lý."

**2. Phần VIF sau encoding thực tế (Bước 2) chỉ tính trên `_te` và `_ord`**

Bạn không tính VIF cho các cột Binary (`_bin`) và OHE sau khi apply lên toàn bộ train/test. Đề bài yêu cầu *"với từng phương pháp, đo VIF"* — nếu hiểu strict thì phần demo đã cover, nhưng nếu thầy expect VIF trên full encoding thì chưa đủ.

**3. Frequency Encoding thêm được 0 cột ở Bước 2**

```
[5/5] Frequency Encoding: thêm 0 cột _freq
```

Lý do là sau Binary Encoding, không còn cột object nào trong `freq_cols_all`. Về mặt kỹ thuật thì logic đúng (đã drop object cols), nhưng **trên thực tế Frequency Encoding không được apply lên train/test thực sự** — chỉ có trong demo. Nếu thầy kiểm tra pipeline cuối cùng, đây là điểm yếu.

---

**Tóm lại:**

| Tiêu chí | Trạng thái |
|---|---|
| 5 phương pháp được cài đặt | ✓ |
| Target Encoding có CV | ✓ |
| Binary cho high-cardinality | ✓ |
| VIF đo cho từng phương pháp (demo) | ✓ |
| Frequency Encoding apply thực tế | ✗ chỉ trong demo |
| Giải thích VIF OHE | Cần làm rõ hơn |

**[Highlight 🟡 YELLOW]**
> Lựa chọn và giảm chiều đặc trưng: [Bắt buộc] Cài đặt và so sánh ba tầng:
• Lọc thống kê: ANOVA F-test (thuộc tính số), Chi-square test (thuộc tính phân
loại), Mutual Information.
• Lọc dựa trên mô hình: Feature importance từRandom Forest và Gradient Boosting;
Recursive Feature Elimination (RFE) với cross-validation.
• Giảm chiều: PCA; t-SNE đểtrực quan hóa; UMAP nếu tập dữliệu lớn.
Với mỗi phương pháp lọc, có thểhuấn luyện mô hình học máy trên tập đặc trưng được
chọn và báo cáo cross-validation F1-score (5-fold). Vẽbiểu đồso sánh hiệu năng theo
sốlượng đặc trưng.
  💬 **NOTE:** **Về cơ bản là đúng hướng**, nhưng có **2 vấn đề nghiêm trọng** và **1 vấn đề trung bình** cần xử lý.

---

## Vấn đề nghiêm trọng

**1. RFE cho F1 = 0.0000 ở tất cả n_features**

Đây là **bug thực sự**, không phải kết quả hợp lệ. Nguyên nhân gần như chắc chắn là:

- Dataset cực kỳ imbalanced (fraud rate = 3.6%), với chỉ 3000 dòng thì sample có thể chỉ có ~108 fraud cases.
- Logistic Regression với `C=0.1` (regularization mạnh) trên data imbalanced như vậy sẽ **predict toàn bộ là class 0** → F1 = 0 vì không có true positive nào.
- Fix: thêm `class_weight='balanced'` vào LogisticRegression, hoặc dùng `scoring='f1_weighted'` thay vì `'f1'`.

Kết quả F1 = 0 mà vẫn kết luận "RFE tốt nhất: n=5" là **không có ý nghĩa** và sẽ bị trừ điểm nặng.

**2. Biểu đồ so sánh hiệu năng theo số lượng đặc trưng — bị thiếu hoàn toàn**

Đề bài yêu cầu rõ: *"vẽ biểu đồ so sánh hiệu năng theo số lượng đặc trưng."* Code hiện tại chỉ có:
- Bar chart feature importance (RF, GB) — đây là importance plot, **không phải** performance-vs-n_features plot.
- Không có biểu đồ nào thể hiện CV F1-score thay đổi theo số đặc trưng được chọn.

Biểu đồ cần thiết là dạng: trục x = số đặc trưng (5, 10, 20, 38...), trục y = 5-fold CV F1, vẽ đường cho từng phương pháp (ANOVA top-k, MI top-k, RF top-k, RFE).

---

## Vấn đề trung bình

**3. UMAP bị skip**

Đề nói *"UMAP nếu tập dữ liệu lớn"* — dataset này 590k dòng, rõ ràng là lớn. Việc không cài `umap-learn` trước khi submit là thiếu sót cần fix. Chạy `pip install umap-learn` và re-run là đủ.

---

## Những gì đã đúng

| Yêu cầu | Trạng thái |
|---|---|
| ANOVA F-test | ✓ |
| Chi-square test | ✓ (dù chỉ 6 cột) |
| Mutual Information | ✓ |
| RF Feature Importance | ✓ |
| GB Feature Importance | ✓ |
| RFE + CV | ✗ F1=0, cần fix |
| PCA (95% variance) | ✓ |
| t-SNE visualization | ✓ |
| UMAP | ✗ chưa chạy |
| Performance chart vs n_features | ✗ thiếu hoàn toàn |

---

**Ưu tiên fix theo thứ tự:** RFE bug → performance chart → UMAP.

**[Highlight 🟢 GREEN]**
> [Nâng cao] Phát hiện và xửlý mất cân bằng lớp: Áp dụng SMOTE, ADASYN và
Random Under-sampling. Với mỗi chiến lược, huấn luyện và đánh giá mô hình trên
tập test chưa tái cân bằng. Báo cáo Precision, Recall, F1-macro và AUC-ROC. Giải
thích tại sao không được áp dụng resampling trước khi chia tập train/test. (Nếu tập dữ
liệu có xuất hiện tình trạng mất cân bằng)

---
## Trang 7

**[Highlight 🟢 GREEN]**
> Tính và vẽphân phối độdài văn bản (sốtừ, sốký tự) theo nhãn lớp. Kiểm định Mann-
Whitney U test đểxác định liệu các lớp có sựkhác biệt đáng kểvềđộdài không.

**[Highlight 🟢 GREEN]**
> Vẽword cloud và bảng top-50 từphổbiến nhất theo từng lớp. Tính type-token ratio
(TTR) đểđánh giá độphong phú từvựng.

**[Highlight 🟢 GREEN]**
> Phân tích phân phối Zipf: vẽlog-log plot của tần suất từvà kiểm tra mức độtuân theo
định luật Zipf.

**[Highlight 🟢 GREEN]**
> Pipeline chuẩn hóa văn bản: Cài đặt pipeline hoàn chỉnh gồm: chuyển vềchữthường,
loại bỏHTML/URL/mention/hashtag, loại bỏký tựđặc biệt và số(có điều kiện), chuẩn
hóa khoảng trắng. Với mỗi bước, báo cáo tỉlệtừvựng thay đổi và tác động đến phân
phối độdài văn bản.

**[Highlight 🟢 GREEN]**
> So sánh chiến lược tokenization: Cài đặt và so sánh 4 chiến lược: word-level (NLTK,
spaCy), sentence-level, character-level, và subword (BPE bằng thư viện tokenizers
của HuggingFace). Với mỗi chiến lược, báo cáo: (i) kích thước từvựng; (ii) tỉlệOOV
(out-of-vocabulary) trên tập test; (iii) độdài chuỗi token trung bình.

**[Highlight 🟢 GREEN]**
> Loại bỏstop words và phân tích thông tin: Loại bỏstop words và so sánh: (i) kích
thước từvựng trước/sau; (ii) MI (Mutual Information) trung bình giữa từvà nhãn
trước/sau; (iii) hiệu năng Naive Bayes trước/sau. Thảo luận liệu việc loại bỏstop
words có luôn cải thiện kết quảkhông.

**[Highlight 🟢 GREEN]**
> Stemming, Lemmatization và so sánh định lượng: Cài đặt Porter Stemmer, Snowball
Stemmer và WordNet Lemmatizer. Tính collision rate (tỉlệcác từkhác nhau bịmap
vềcùng một dạng gốc) cho mỗi phương pháp. Đánh giá tác động đến hiệu năng phân
loại (Logistic Regression, 5-fold CV).

---
## Trang 8

**[Highlight 🟢 GREEN]**
> Vector hóa văn bản và phân tích không gian đặc trưng: Cài đặt BoW, TF-IDF (n-
gram với n ∈{1, 2, 3}), và Word2Vec (huấn luyện trên tập dữliệu). Với mỗi biểu diễn:
(i) báo cáo sốchiều và độthưa (sparsity ratio); (ii) tính cosine similarity giữa các cặp
văn bản cùng lớp và khác lớp; (iii) trực quan hóa t-SNE 2D tô màu theo nhãn. Đánh
giá khảnăng tách lớp bằng silhouette score.

**[Highlight 🟢 GREEN]**
> [Nâng cao] Biểu diễn ngữnghĩa bằng Sentence Transformer: Sửdụng mô hình
pretrained (ví dụ: all-MiniLM-L6-v2 từsentence-transformers). So sánh
chất lượng phân cụm (K-Means, silhouette score) và hiệu năng phân loại (Linear SVM)
giữa TF-IDF và Sentence Transformer embeddings. Giải thích sựkhác biệt vềmặt ngữ
nghĩa.

---
## Trang 10

**[Highlight 🟡 YELLOW]**
> Cấu trúc thư mục bắt:
Sinh viên có thểtham kh
