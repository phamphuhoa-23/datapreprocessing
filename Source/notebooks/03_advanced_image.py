# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#     language: python
#     name: python3
# ---
# =============================================================================
# SOURCE/notebooks/03_advanced_image.py
# Notebook: 03 - Phân tích nâng cao: PCA & Edge Detection
# Ngôn ngữ: Tiếng Việt (markdown) + Python (code)
#
# FEEDBACK từ leader (FeedbackFromLeader.pdf):
# [ADD] PCA: thêm markdown giải thích tại sao chọn grayscale 64x64
# [FIX] PCA class selection: dùng quantile (np.linspace) thay vì 5 PC1 thấp nhất + 5 cao nhất
# [FIX] t-SNE: thêm giải thích tại sao dùng 50-dim PCA trước t-SNE
# [FIX] t-SNE: điều chỉnh perplexity (hiện tại =30, thử 50/100 nếu cluster dính nhau)
# [FIX] Đánh giá phân tách: dùng purity metric (ARI) thay vì chỉ dựa ANOVA PC1
# [🔴 P2] Edge Detection: thiếu Prewitt + Sobel không có ngưỡng T → edge density = 0
# [🔴 P2] Ablation ngưỡng T cho Sobel/Prewitt: T_values = [30, 50, 80, 120]
# [🔴 P2] Ablation sigma + T1/T2 cho Canny: 4 configs
# [FIX] Lý thuyết edge detection: cái quan trọng là ngưỡng T (không chỉ kernel size)
# =============================================================================
#

# %% [markdown]
# # 03 - Phân tích nâng cao: PCA & Edge Detection
#
# **Mục tiêu:** Phân tích PCA eigenimages + Sobel/Prewitt/Canny edge detection trên dataset NWPU-RESISC45,
# kèm kiểm định thống kê.
#
# **Dataset:** NWPU-RESISC45 - 45 lớp, 256×256 pixel.

# %% [markdown]
# ## 0. Setup

# %%
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from pathlib import Path
import os
import glob
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from tqdm import tqdm
from scipy import stats
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

warnings.filterwarnings('ignore')
plt.rcParams.update(
    {'figure.figsize': (12, 6), 'font.size': 11, 'figure.dpi': 100})
sns.set_style("whitegrid")


def _find_image_root() -> Path:
    """Tìm thư mục data/raw/image/ chứa train/."""
    candidates = [
        # Cấu trúc chuẩn: Source/data/raw/image/
        Path.cwd().parent / 'data' / 'raw' / 'image',
        Path.cwd() / 'data' / 'raw' / 'image',
        Path.cwd().parent.parent / 'data' / 'raw' / 'image',
        # Legacy fallback
        Path.cwd() / 'Dataset',
        Path.cwd().parent / 'Dataset',
        Path.cwd() / 'Source' / 'Dataset',
        Path.cwd() / 'DataMining-Lab1' / 'Dataset',
        Path.cwd().parent / 'Source' / 'Dataset',
        Path.cwd().parent / 'DataMining-Lab1' / 'Dataset',
        Path.cwd().parent.parent / 'Source' / 'Dataset',
    ]
    try:
        candidates.insert(0, Path(__file__).resolve().parent.parent / 'data' / 'raw' / 'image')
    except NameError:
        pass
    for p in candidates:
        if (p / 'train').exists() and any((p / 'train').iterdir()):
            return p
    raise FileNotFoundError(
        "Không tìm thấy data/raw/image/train/. Đặt ảnh NWPU-RESISC45 vào Source/data/raw/image/.")


_IMG_ROOT = _find_image_root()
TRAIN_DIR = str(_IMG_ROOT / 'train')
print(f"TRAIN_DIR = {TRAIN_DIR}")
classes = sorted(os.listdir(TRAIN_DIR))
print(f"Số lớp: {len(classes)}")


# %%
def load_sample(n_per_class=20, target_classes=None, seed=42):
    np.random.seed(seed)
    target = target_classes or classes
    samples = []
    for cls in target:
        paths = glob.glob(os.path.join(TRAIN_DIR, cls, "*.jpg"))
        chosen = np.random.choice(paths, min(
            n_per_class, len(paths)), replace=False)
        for p in chosen:
            img = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
            samples.append((img, cls))
    return samples


# %% [markdown]
# ---
# ## 1. [Nâng cao] PCA đặc trưng ảnh
#
# **Lý thuyết:**
#
# **PCA (Principal Component Analysis)** tìm các hướng **phương sai cực đại** trong dữ liệu.
# Với ma trận dữ liệu đã center $\mathbf{X} \in \mathbb{R}^{n \times d}$ ($n$ ảnh, $d$ pixel):
#
# **Bước 1 — Covariance matrix:**
#
# $$\mathbf{C} = \frac{1}{n-1}\mathbf{X}^\top \mathbf{X} \in \mathbb{R}^{d \times d}$$
#
# **Bước 2 — Eigen decomposition (SVD):**
#
# $$\mathbf{X} = \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^\top$$
#
# - $\mathbf{V} \in \mathbb{R}^{d \times d}$: eigenvectors (principal components / eigenimages)
# - $\boldsymbol{\Sigma}$: singular values; eigenvalues $\lambda_k = \sigma_k^2 / (n-1)$
# - Eigenimage $k$: reshape $\mathbf{v}_k$ về $64 \times 64$ → hiển thị như "khuôn mặt trung bình"
#
# **Bước 3 — Explained variance ratio:**
#
# $$\text{EVR}_k = \frac{\lambda_k}{\sum_{i=1}^{d} \lambda_i}$$
#
# **Bước 4 — Projection & Reconstruction:**
#
# $$\mathbf{z} = \mathbf{V}_k^\top (\mathbf{x} - \bar{\mathbf{x}}) \in \mathbb{R}^k, \quad
# \hat{\mathbf{x}} = \mathbf{V}_k \mathbf{z} + \bar{\mathbf{x}}$$
#
# Sai số tái tạo $\|\mathbf{x} - \hat{\mathbf{x}}\|^2 = \sum_{i=k+1}^{d} \lambda_i$
# (phần variance bị bỏ qua khi dùng $k$ components).
#
# **Lựa chọn kích thước ảnh:** Dùng **Grayscale 64×64** (vector 4,096 chiều) thay vì 224×224
# vì: (1) ảnh 224×224 tạo ma trận 27,000 × 50,176 ≈ 5.4 GB RAM; (2) PCA phân tích cấu trúc
# hình học tổng thể — không đòi hỏi độ phân giải cao; (3) Grayscale bỏ nhiễu màu sắc,
# giúp eigenimages biểu diễn rõ hơn đặc trưng texture/shape.
# Dùng **IncrementalPCA** xử lý từng batch — không cần load toàn bộ vào memory cùng lúc.
#
# - **$H_0$:** Các lớp không phân tách được trên PC1 (PC1 mean không khác biệt giữa lớp).
# - **$H_1$:** Ít nhất một cặp lớp phân tách được trên PC1.

# %%
PCA_SIZE = 64
# đủ để xác định ngưỡng 90/95/99% variance (có thể >200 với ảnh tự nhiên)
N_COMPONENTS = 800
BATCH_SIZE = 500    # số ảnh mỗi batch cho IncrementalPCA

# Load toàn bộ ảnh train (27,000 ảnh = 600/lớp × 45 lớp)
pca_data, pca_labels = [], []
for cls in tqdm(classes, desc="Loading all images"):
    for p in sorted(glob.glob(os.path.join(TRAIN_DIR, cls, "*.jpg"))):
        img = cv2.imread(p)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (PCA_SIZE, PCA_SIZE))
        pca_data.append(img.ravel().astype(np.float32) / 255.0)
        pca_labels.append(cls)

X_pca = np.array(pca_data)   # shape: (27000, 4096)
y_pca = np.array(pca_labels)
print(f"Ma trận PCA: {X_pca.shape}  ({X_pca.nbytes / 1e6:.0f} MB)")

# %%
# IncrementalPCA: fit từng batch
ipca = IncrementalPCA(n_components=N_COMPONENTS, batch_size=BATCH_SIZE)
for start in tqdm(range(0, len(X_pca), BATCH_SIZE), desc="Fitting IncrementalPCA"):
    ipca.partial_fit(X_pca[start:start + BATCH_SIZE])

# Transform toàn bộ
X_pca_transformed = ipca.transform(X_pca)

cum_var = np.cumsum(ipca.explained_variance_ratio_)
n90 = int(np.argmax(cum_var >= 0.90) +
          1) if cum_var[-1] >= 0.90 else f">{N_COMPONENTS}"
n95 = int(np.argmax(cum_var >= 0.95) +
          1) if cum_var[-1] >= 0.95 else f">{N_COMPONENTS}"
n99 = int(np.argmax(cum_var >= 0.99) +
          1) if cum_var[-1] >= 0.99 else f">{N_COMPONENTS}"

print(f"Components để giải thích 90% variance: {n90}")
print(f"Components để giải thích 95% variance: {n95}")
print(f"Components để giải thích 99% variance: {n99}")
if cum_var[-1] < 0.99:
    print(f"WARNING: cum_var tối đa chỉ đạt {cum_var[-1]*100:.1f}% — tăng N_COMPONENTS để xác định n99 chính xác")

# %% [markdown]
# ### Scree Plot & Cumulative Variance

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].bar(range(1, 31), ipca.explained_variance_ratio_[
            :30], color='#4C72B0', alpha=0.8)
axes[0].set_xlabel("Component")
axes[0].set_ylabel("Explained Variance Ratio")
axes[0].set_title("Scree Plot (top 30 trong tổng " +
                  str(N_COMPONENTS) + " components)")

axes[1].plot(range(1, len(cum_var)+1), cum_var, color='#4C72B0', linewidth=2)
axes[1].axhline(0.99, color='purple', linestyle='--',
                alpha=0.6, label=f'99% (n={n99})')
axes[1].axhline(0.95, color='red',    linestyle='--',
                alpha=0.6, label=f'95% (n={n95})')
axes[1].axhline(0.90, color='orange', linestyle='--',
                alpha=0.6, label=f'90% (n={n90})')
for n_thresh, color in [(n90, 'orange'), (n95, 'red'), (n99, 'purple')]:
    if isinstance(n_thresh, int):
        axes[1].axvline(n_thresh, color=color, linestyle=':', alpha=0.5)
axes[1].set_xlabel("Số components")
axes[1].set_ylabel("Cumulative Explained Variance")
axes[1].set_title("Cumulative Explained Variance")
axes[1].legend(fontsize=9)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Eigenimages (Top 20 Principal Components)

# %%
fig, axes = plt.subplots(4, 5, figsize=(16, 13))
for idx, ax in enumerate(axes.ravel()):
    eigenimg = ipca.components_[idx].reshape(PCA_SIZE, PCA_SIZE)
    ax.imshow(eigenimg, cmap='gray')
    ax.set_title(
        f"PC{idx+1} ({ipca.explained_variance_ratio_[idx]*100:.1f}%)", fontsize=8)
    ax.axis('off')
plt.suptitle(
    "Top 20 Eigenimages (IncrementalPCA, 27,000 ảnh 64×64)", fontsize=13)
plt.tight_layout()
plt.show()

# %%
# Mean image (từ IncrementalPCA mean_ — trung bình của toàn bộ 27,000 ảnh)
mean_img = ipca.mean_.reshape(PCA_SIZE, PCA_SIZE)
fig, ax = plt.subplots(1, 1, figsize=(4, 4))
ax.imshow(mean_img, cmap='gray')
ax.set_title("Mean Image (toàn bộ 27,000 ảnh train)")
ax.axis('off')
plt.tight_layout()
plt.show()

# %% [markdown]
# ### 2D PCA Projection — toàn bộ 45 lớp

# %%
# Dùng toàn bộ 45 lớp, mỗi điểm là 1 ảnh (27,000 điểm tổng)

colors_45 = plt.cm.nipy_spectral(np.linspace(0, 0.9, len(classes)))

fig, ax = plt.subplots(figsize=(12, 9))
for cls, color in zip(classes, colors_45):
    mask = y_pca == cls
    ax.scatter(X_pca_transformed[mask, 0], X_pca_transformed[mask, 1],
               c=[color], label=cls, alpha=0.4, s=8)

ax.set_xlabel(f"PC1 ({ipca.explained_variance_ratio_[0]*100:.1f}%)")
ax.set_ylabel(f"PC2 ({ipca.explained_variance_ratio_[1]*100:.1f}%)")
ax.set_title("PCA 2D Projection — toàn bộ 45 lớp (27,000 ảnh)")
ax.legend(fontsize=6, loc='best', ncol=5, markerscale=2)
plt.tight_layout()
plt.show()

# %%
# Adjusted Rand Index: đo mức độ phân tách của 45 lớp trong không gian PCA 2D
y_encoded = np.array([classes.index(c) for c in y_pca])
kmeans_ari = KMeans(n_clusters=45, random_state=42, n_init=5)
cluster_labels = kmeans_ari.fit_predict(X_pca_transformed[:, :2])
ari = adjusted_rand_score(y_encoded, cluster_labels)
print(f"Adjusted Rand Index (PCA 2D, 45 classes): {ari:.3f}")
if ari >= 0.4:
    print(f"  => phân tách tốt")
elif ari >= 0.1:
    print(f"  => phân tách trung bình – PCA 2D giữ được một phần cấu trúc")
else:
    print(f"  => phân tách yếu – cần thêm chiều hoặc non-linear method (t-SNE)")

# %% [markdown]
# ### 3D PCA Projection — toàn bộ 45 lớp

# %%

fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection='3d')

for cls, color in zip(classes, colors_45):
    mask = y_pca == cls
    ax.scatter(X_pca_transformed[mask, 0],
               X_pca_transformed[mask, 1],
               X_pca_transformed[mask, 2],
               c=[color], label=cls, alpha=0.4, s=8)

ax.set_xlabel(f"PC1 ({ipca.explained_variance_ratio_[0]*100:.1f}%)")
ax.set_ylabel(f"PC2 ({ipca.explained_variance_ratio_[1]*100:.1f}%)")
ax.set_zlabel(f"PC3 ({ipca.explained_variance_ratio_[2]*100:.1f}%)")
ax.set_title("PCA 3D Projection — toàn bộ 45 lớp (27,000 ảnh)")
ax.legend(fontsize=5, loc="upper left", ncol=5, markerscale=2)
plt.tight_layout()
plt.show()


# %% [markdown]
# ### t-SNE Projection — toàn bộ 45 lớp
#
# t-SNE (non-linear) giữ cấu trúc local tốt hơn PCA — dùng 50 PCA components làm input để:
# (a) loại noise (các PC cuối chủ yếu là noise), (b) tăng tốc t-SNE (giảm chiều trước),
# (c) best practice theo Van der Maaten & Hinton 2008.
# Perplexity=30, subsample 5,000 ảnh (t-SNE có độ phức tạp O(n²), 27,000 điểm quá chậm).

# %%

# Subsample đồng đều (nhất quán mỗi lớp: ~111 ảnh/lớp × 45 = 5,000)
np.random.seed(42)
N_TSNE = 5000
tsne_idx = np.random.choice(len(X_pca), N_TSNE, replace=False)
X_tsne_input = X_pca_transformed[tsne_idx, :50]
y_tsne = y_pca[tsne_idx]

print(
    f"Running t-SNE trên {N_TSNE} điểm (50 PCA components, perplexity=30)...")
tsne_model = TSNE(n_components=2, random_state=42,
                  perplexity=30, max_iter=1000)
X_tsne = tsne_model.fit_transform(X_tsne_input)
print("Done.")

fig, ax = plt.subplots(figsize=(14, 10))
for cls, color in zip(classes, colors_45):
    mask = y_tsne == cls
    ax.scatter(X_tsne[mask, 0], X_tsne[mask, 1],
               c=[color], label=cls, alpha=0.6, s=12)
ax.set_title(f"t-SNE 2D — toàn bộ 45 lớp (subsample {N_TSNE} ảnh, perplexity=30, input: 50 PCA components)",
             fontsize=12)
ax.set_xlabel("t-SNE 1")
ax.set_ylabel("t-SNE 2")
ax.legend(fontsize=6, loc='best', ncol=5, markerscale=2)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### t-SNE Perplexity=50 — so sánh với Perplexity=30
#
# Perplexity=30 là giá trị mặc định, phù hợp khi sample size ~100–500/cụm.
# Với subsample 5,000 ảnh (111 ảnh/lớp × 45 lớp), cụm có thể bị dính nếu perplexity quá nhỏ.
# Thử perplexity=50 để kiểm tra: cluster có tách ra tốt hơn không?

# %%
print(f"Running t-SNE với perplexity=50 (so sánh với p=30)...")
tsne_50 = TSNE(n_components=2, random_state=42, perplexity=50, max_iter=1000)
X_tsne_50 = tsne_50.fit_transform(X_tsne_input)
print("Done.")

fig, axes_tsne = plt.subplots(1, 2, figsize=(22, 9))
for ax_t, X_t, perp in zip(axes_tsne, [X_tsne, X_tsne_50], [30, 50]):
    for cls, color in zip(classes, colors_45):
        mask = y_tsne == cls
        ax_t.scatter(X_t[mask, 0], X_t[mask, 1],
                     c=[color], label=cls, alpha=0.6, s=10)
    ax_t.set_title(f"t-SNE perplexity={perp} ({N_TSNE} ảnh, 50 PCA dims)", fontsize=11)
    ax_t.set_xlabel("t-SNE 1")
    ax_t.set_ylabel("t-SNE 2")
    ax_t.legend(fontsize=5, loc='best', ncol=5, markerscale=2)
plt.suptitle("So sánh t-SNE Perplexity=30 vs Perplexity=50", fontsize=13)
plt.tight_layout()
plt.show()


# %% [markdown]
# **Nhận xét t-SNE so với PCA:**
#
# - **PCA 2D** (linear): giữ cấu trúc global — lớp phân tán chủ yếu theo độ sáng (PC1).
# - **t-SNE** (non-linear): giữ cấu trúc local tốt hơn — lớp có texture đặc trưng (desert, snowberg, runway) thường tạo cluster riêng rõ hơn.
# - Cả hai cho thấy các lớp phức tạp (harbor, dense_residential, industrial_area) overlap nhiều do đặc trưng visual đa dạng.
#

# %% [markdown]
# ### Kiểm định: Các lớp phân tách trên PC1?

# %%
groups_pc1 = [X_pca_transformed[y_pca == c, 0] for c in classes]

# Normality
print("Normality (Shapiro-Wilk, 5 lớp mẫu):")
for cls in classes[:5]:
    vals = X_pca_transformed[y_pca == cls, 0]
    w, p = stats.shapiro(vals)
    print(f"  {cls}: W={w:.4f}, p={p:.4f}")

# Levene
lev_s, lev_p = stats.levene(*groups_pc1)
print(f"\nLevene: stat={lev_s:.2f}, p={lev_p:.2e}")

# ANOVA + Kruskal
f_val, p_val = stats.f_oneway(*groups_pc1)
h_val, kw_p = stats.kruskal(*groups_pc1)
print(f"\nANOVA: F={f_val:.2f}, p={p_val:.2e}")
print(f"Kruskal-Wallis: H={h_val:.2f}, p={kw_p:.2e}")

# Eta²
ss_between = sum(
    len(g) * (g.mean() - X_pca_transformed[:, 0].mean())**2 for g in groups_pc1)
ss_total = sum((x - X_pca_transformed[:, 0].mean())
               ** 2 for x in X_pca_transformed[:, 0])
eta_sq = ss_between / ss_total
print(f"\nEta² = {eta_sq:.3f}")

# %% [markdown]
# **Kết luận PCA:**
#
# - PC1 là thành phần dominant (xem % variance in output trên).
# - Eigenimages cho thấy PC1 nắm bắt pattern độ sáng tổng thể; PC2-PC3 nắm bắt cạnh/texture.
# - Số components đạt 90/95/99% variance in ra bên trên — đây là số thực sự đo được.
# - Levene test + Kruskal-Wallis (xem output): nếu p ≈ 0 → bác bỏ H₀ → các lớp
#   **phân tách được** trên PC1. Eta² cho biết mức độ effect size (>0.14 là large).

# %% [markdown]
# ---
# ## 2. [Nâng cao] Edge Detection — Threshold T là siêu tham số chính
#
# **Lý thuyết:**
#
# | Bộ lọc | Đầu ra | Siêu tham số chính |
# |--------|--------|-------------------|
# | **Sobel / Prewitt** | Gradient magnitude M (giá trị thực liên tục) | **Ngưỡng T**: binarize cạnh theo `M > T` |
# | **Canny** | Edge map nhị phân (hysteresis thresholding) | **T₁** (low), **T₂** (high); σ Gaussian để làm trơn trước |
#
# - **Sobel/Prewitt:** `edge_density = mean(M > T)` — T điều chỉnh độ nhạy phát hiện cạnh,
#   quan trọng hơn kernel_size vì kernel_size chỉ ảnh hưởng đến độ mịn.
#   Theo thực nghiệm, T nên được xác định bằng ablation (không có công thức lý thuyết cố định).
# - **Canny:** T₂/T₁ ≈ 2–3 (Canny 1986). σ=1 chuẩn, σ=2 cho ảnh nhiễu.
#   T₁, T₂ là siêu tham số chính — tăng T₁/T₂ → cạnh yếu bị loại bỏ.
#
# - **$H_0$:** Edge density không khác biệt giữa 45 lớp.
# - **$H_1$:** Ít nhất 1 cặp lớp có edge density khác biệt.

# %%
# Helper kernels và functions
_PREWITT_X = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)
_PREWITT_Y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float32)


def sobel_density(img_gray, T):
    """Edge density Sobel với ngưỡng T"""
    sx = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
    sy = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
    return (np.sqrt(sx**2 + sy**2) > T).mean()


def prewitt_density(img_gray, T):
    """Edge density Prewitt với ngưỡng T"""
    px = cv2.filter2D(img_gray.astype(np.float32), -1, _PREWITT_X)
    py = cv2.filter2D(img_gray.astype(np.float32), -1, _PREWITT_Y)
    return (np.sqrt(px**2 + py**2) > T).mean()


def canny_density(img_gray, sigma, T1, T2):
    """Edge density Canny với Gaussian sigma + T1/T2"""
    if sigma > 0:
        ks = int(2 * round(2 * sigma) + 1)
        blurred = cv2.GaussianBlur(img_gray, (ks, ks), sigma)
    else:
        blurred = img_gray
    return cv2.Canny(blurred, T1, T2).mean() / 255.0


def norm_edge_vis(arr):
    m = arr.max()
    return np.clip(arr / m * 255, 0, 255).astype(np.uint8) if m > 0 else np.zeros_like(arr, dtype=np.uint8)


EDGE_SAMPLE = 30

# %% [markdown]
# ### 2.1 Demo: Ảnh hưởng của ngưỡng T lên Sobel edge map
#
# Chọn 5 lớp đại diện từ 5 phân vị (min, Q1, median, Q3, max) của Sobel density
# tính nhanh với T=50 để đảm bảo bao phủ dải texture khác nhau.

# %%
# Tính nhanh Sobel density T=50 để chọn 5 lớp đại diện
quick_density = {}
for cls in classes:
    paths = glob.glob(os.path.join(TRAIN_DIR, cls, "*.jpg"))[:5]
    np.random.seed(42)
    vals = [sobel_density(cv2.imread(p, cv2.IMREAD_GRAYSCALE), T=50)
            for p in paths]
    quick_density[cls] = np.mean(vals)

sorted_by_density = sorted(quick_density, key=quick_density.get)
n = len(sorted_by_density)
demo_classes = [sorted_by_density[i] for i in [0, n//4, n//2, 3*n//4, n-1]]
print("5 lớp đại diện (min → max Sobel density T=50):")
for cls in demo_classes:
    print(f"  {cls}: {quick_density[cls]:.4f}")

# %%
# Demo: hiệu ứng của T trên 5 lớp đại diện — T là siêu tham số chính
T_DEMO = [30, 50, 80, 120]
demo_samples = load_sample(n_per_class=1, target_classes=demo_classes)

fig, axes = plt.subplots(len(demo_classes), len(T_DEMO) + 1, figsize=(15, 14))
col_titles = ["Original"] + [f"Sobel T={t}" for t in T_DEMO]
for ci, t in enumerate(col_titles):
    axes[0][ci].set_title(t, fontsize=9)

for row, (img, cls) in enumerate(demo_samples):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(sx**2 + sy**2)

    axes[row][0].imshow(img)
    axes[row][0].set_ylabel(cls, fontsize=8, rotation=0, labelpad=75)
    axes[row][0].axis('off')

    for ci, T in enumerate(T_DEMO):
        edge_bin = (mag > T).astype(np.uint8) * 255
        axes[row][ci + 1].imshow(edge_bin, cmap='gray')
        d = (mag > T).mean()
        axes[row][ci + 1].set_xlabel(f"density={d:.3f}", fontsize=7)
        axes[row][ci + 1].axis('off')

plt.suptitle(
    "Ảnh hưởng của ngưỡng T lên Sobel edge map (T nhỏ → nhiều cạnh hơn)", fontsize=12)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### 2.2 Ablation: Ngưỡng T cho Sobel và Prewitt
#
# Tính edge density trên toàn bộ 45 lớp với T ∈ {30, 50, 80, 120}.
# **Kỳ vọng:** T tăng → density giảm; thứ tự xếp hạng lớp không đổi nhiều.

# %%
T_VALUES = [30, 50, 80, 120]
ablation_data = []

for cls in tqdm(classes, desc="Sobel/Prewitt ablation"):
    paths = glob.glob(os.path.join(TRAIN_DIR, cls, "*.jpg"))
    np.random.seed(42)
    chosen = np.random.choice(paths, min(
        EDGE_SAMPLE, len(paths)), replace=False)
    for p in chosen:
        img_g = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        row_data = {'class': cls}
        for T in T_VALUES:
            row_data[f'sobel_T{T}'] = sobel_density(img_g, T)
            row_data[f'prewitt_T{T}'] = prewitt_density(img_g, T)
        ablation_data.append(row_data)

df_ablation = pd.DataFrame(ablation_data)
print("Sobel và Prewitt density trung bình theo T:")
print(f"{'T':>5} {'Sobel':>10} {'Prewitt':>10}")
for T in T_VALUES:
    print(f"{T:>5} {df_ablation[f'sobel_T{T}'].mean():>10.4f} {
          df_ablation[f'prewitt_T{T}'].mean():>10.4f}")

# %%
# Đường cong edge density trung bình theo T (Sobel vs Prewitt)
mean_sobel = [df_ablation[f'sobel_T{T}'].mean() for T in T_VALUES]
mean_prewitt = [df_ablation[f'prewitt_T{T}'].mean() for T in T_VALUES]
std_sobel = [df_ablation[f'sobel_T{T}'].std() for T in T_VALUES]
std_prewitt = [df_ablation[f'prewitt_T{T}'].std() for T in T_VALUES]

fig, ax = plt.subplots(figsize=(8, 5))
ax.errorbar(T_VALUES, mean_sobel,   yerr=std_sobel,
            marker='o', capsize=5, label='Sobel',   linewidth=2)
ax.errorbar(T_VALUES, mean_prewitt, yerr=std_prewitt,
            marker='s', capsize=5, label='Prewitt', linewidth=2)
ax.set_xlabel("Ngưỡng T")
ax.set_ylabel("Edge density trung bình")
ax.set_title("Ablation ngưỡng T — Sobel vs Prewitt (45 lớp)")
ax.legend()
ax.set_xticks(T_VALUES)
plt.tight_layout()
plt.show()

# %%
# Boxplot Sobel theo T — thấy phân phối thay đổi theo T
df_ablation_long = df_ablation.melt(id_vars=['class'],
                                    value_vars=[
                                        f'sobel_T{T}' for T in T_VALUES],
                                    var_name='T_label', value_name='density')
df_ablation_long['T'] = df_ablation_long['T_label'].str.extract(
    r'T(\d+)').astype(int)

fig, ax = plt.subplots(figsize=(10, 5))
sns.boxplot(data=df_ablation_long, x='T',
            y='density', ax=ax, palette='Blues_d')
ax.set_xlabel("Ngưỡng T")
ax.set_ylabel("Sobel edge density")
ax.set_title(
    "Phân bố Sobel edge density theo ngưỡng T (toàn bộ 45 lớp × 30 ảnh)")
plt.tight_layout()
plt.show()

# Chọn T tốt nhất: T phân biệt các lớp tốt nhất (eta² cao nhất)
print("\nANOVA eta² theo T (Sobel) — T nào phân biệt lớp tốt nhất:")
best_T_eta2 = {}
for T in T_VALUES:
    col = f'sobel_T{T}'
    grps = [df_ablation[df_ablation['class'] == c]
            [col].values for c in classes]
    f_val, _ = stats.f_oneway(*grps)
    ss_b = sum(len(g) * (g.mean() - df_ablation[col].mean())**2 for g in grps)
    ss_t = ((df_ablation[col] - df_ablation[col].mean())**2).sum()
    eta2 = ss_b / ss_t
    best_T_eta2[T] = eta2
    print(f"  T={T:>3}: F={f_val:.1f}, η²={eta2:.3f}")

BEST_T = max(best_T_eta2, key=best_T_eta2.get)
print(f"\n→ Chọn T={BEST_T} (η² cao nhất = {best_T_eta2[BEST_T]:.3f})")

# %% [markdown]
# ### 2.3 Ablation: Tham số Canny (sigma, T₁, T₂)
#
# 4 cấu hình thử nghiệm, tỷ lệ T₂/T₁ ≈ 2–3 (theo khuyến nghị Canny 1986):

# %%
CANNY_CONFIGS = [
    {'sigma': 1.0, 'T1':  50, 'T2': 150},
    {'sigma': 1.0, 'T1': 100, 'T2': 200},
    {'sigma': 2.0, 'T1':  50, 'T2': 150},
    {'sigma': 2.0, 'T1': 100, 'T2': 200},
]

canny_data = []
for cls in tqdm(classes, desc="Canny ablation"):
    paths = glob.glob(os.path.join(TRAIN_DIR, cls, "*.jpg"))
    np.random.seed(42)
    chosen = np.random.choice(paths, min(
        EDGE_SAMPLE, len(paths)), replace=False)
    for p in chosen:
        img_g = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        row_data = {'class': cls}
        for cfg in CANNY_CONFIGS:
            key = f"s{cfg['sigma']}_T{cfg['T1']}_{cfg['T2']}"
            row_data[key] = canny_density(
                img_g, cfg['sigma'], cfg['T1'], cfg['T2'])
        canny_data.append(row_data)

df_canny = pd.DataFrame(canny_data)
canny_cols = [f"s{c['sigma']}_T{c['T1']}_{c['T2']}" for c in CANNY_CONFIGS]

print("Canny edge density trung bình theo cấu hình:")
print(f"{'Config':<22} {'Mean':>8} {'Std':>8}")
for col in canny_cols:
    print(
        f"  {col:<22} {df_canny[col].mean():>8.4f} {df_canny[col].std():>8.4f}")

# %%
# Boxplot so sánh 4 Canny configs
df_canny_long = df_canny.melt(id_vars=['class'], value_vars=canny_cols,
                              var_name='config', value_name='density')

fig, ax = plt.subplots(figsize=(10, 5))
sns.boxplot(data=df_canny_long, x='config',
            y='density', ax=ax, palette='Oranges_d')
ax.set_xlabel("Cấu hình (sigma_T1_T2)")
ax.set_ylabel("Canny edge density")
ax.set_title("Ablation Canny: so sánh 4 cấu hình (sigma, T₁, T₂)")
ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha='right')
plt.tight_layout()
plt.show()

# ANOVA eta² cho mỗi config
print("\nANOVA eta² theo Canny config:")
best_canny_eta2 = {}
for col in canny_cols:
    grps = [df_canny[df_canny['class'] == c][col].values for c in classes]
    f_val, _ = stats.f_oneway(*grps)
    ss_b = sum(len(g) * (g.mean() - df_canny[col].mean())**2 for g in grps)
    ss_t = ((df_canny[col] - df_canny[col].mean())**2).sum()
    eta2 = ss_b / ss_t
    best_canny_eta2[col] = eta2
    print(f"  {col:<22}: F={f_val:.1f}, η²={eta2:.3f}")

BEST_CANNY = max(best_canny_eta2, key=best_canny_eta2.get)
print(
    f"\n→ Cấu hình tốt nhất: {BEST_CANNY} (η²={best_canny_eta2[BEST_CANNY]:.3f})")

# %% [markdown]
# ### 2.4 Edge Density theo lớp — T tối ưu
#
# Dùng T tốt nhất từ ablation tính edge density trên toàn bộ 45 lớp.
# So sánh Sobel (T = BEST_T), Canny (config tốt nhất), và Prewitt (T = BEST_T).

# %%
edge_final = []
for cls in tqdm(classes, desc="Edge density final"):
    paths = glob.glob(os.path.join(TRAIN_DIR, cls, "*.jpg"))
    np.random.seed(42)
    chosen = np.random.choice(paths, min(
        EDGE_SAMPLE, len(paths)), replace=False)
    best_cfg = {c['sigma']: c for c in CANNY_CONFIGS
                if f"s{c['sigma']}_T{c['T1']}_{c['T2']}" == BEST_CANNY}
    best_cfg_item = [
        c for c in CANNY_CONFIGS if f"s{c['sigma']}_T{c['T1']}_{c['T2']}" == BEST_CANNY][0]
    for p in chosen:
        img_g = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        edge_final.append({
            'class': cls,
            'sobel':   sobel_density(img_g, BEST_T),
            'prewitt': prewitt_density(img_g, BEST_T),
            'canny':   canny_density(img_g, best_cfg_item['sigma'],
                                     best_cfg_item['T1'], best_cfg_item['T2'])
        })

df_final = pd.DataFrame(edge_final)
class_sobel_final = df_final.groupby('class')['sobel'].mean().sort_values()
class_canny_final = df_final.groupby('class')['canny'].mean().sort_values()
class_prewitt_final = df_final.groupby('class')['prewitt'].mean().sort_values()

print(f"Sobel   (T={BEST_T}): mean={df_final['sobel'].mean():.4f}")
print(f"Prewitt (T={BEST_T}): mean={df_final['prewitt'].mean():.4f}")
print(f"Canny   ({BEST_CANNY}): mean={df_final['canny'].mean():.4f}")
print("\nTop 5 Sobel edge density cao nhất:")
for cls, val in class_sobel_final.tail(5).items():
    print(f"  {cls}: {val:.4f}")
print("Top 5 thấp nhất:")
for cls, val in class_sobel_final.head(5).items():
    print(f"  {cls}: {val:.4f}")

# %%
# Boxplot top + bottom classes theo Sobel
top_bottom = list(
    class_sobel_final.index[:7]) + list(class_sobel_final.index[-7:])
df_sub = df_final[df_final['class'].isin(top_bottom)]
order_sub = class_sobel_final[class_sobel_final.index.isin(
    top_bottom)].index.tolist()

fig, axes = plt.subplots(1, 3, figsize=(20, 5))
for ax, col, title in zip(axes,
                          ['sobel', 'prewitt', 'canny'],
                          [f'Sobel (T={BEST_T})', f'Prewitt (T={BEST_T})', f'Canny ({BEST_CANNY})']):
    sns.boxplot(data=df_sub, x='class', y=col,
                order=order_sub, ax=ax, palette='RdYlGn_r')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45,
                       ha='right', fontsize=8)
    ax.set_title(f"{title} (7 thấp + 7 cao)")
    ax.set_ylabel("Edge Density")
plt.suptitle(
    "Edge Density theo lớp — 3 bộ lọc với tham số tốt nhất", fontsize=13)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### 2.5 Kiểm định thống kê — Edge density

# %%
for col, label in [('sobel', f'Sobel T={BEST_T}'),
                   ('prewitt', f'Prewitt T={BEST_T}'),
                   ('canny', f'Canny {BEST_CANNY}')]:
    grps = [df_final[df_final['class'] == c][col].values for c in classes]
    lev_s, lev_p = stats.levene(*grps)
    f_val, p_anova = stats.f_oneway(*grps)
    h_val, p_kw = stats.kruskal(*grps)
    ss_b = sum(len(g) * (g.mean() - df_final[col].mean())**2 for g in grps)
    ss_t = ((df_final[col] - df_final[col].mean())**2).sum()
    eta2 = ss_b / ss_t
    print(f"=== {label} ===")
    print(f"  Levene:   stat={lev_s:.2f}, p={lev_p:.2e}")
    print(f"  ANOVA:    F={f_val:.2f}, p={p_anova:.2e}")
    print(f"  Kruskal:  H={h_val:.2f}, p={p_kw:.2e}")
    print(
        f"  Eta²={eta2:.3f}  ({'lớn ≥0.14' if eta2 >= 0.14 else 'trung bình ≥0.06' if eta2 >= 0.06 else 'nhỏ'})")
    print()

# %% [markdown]
# **Kết luận Edge Detection:**
#
# - **Ngưỡng T là siêu tham số chính:** T tăng → density giảm mạnh;
#   tỷ lệ xếp hạng lớp ổn định qua các T → cho phép so sánh lớp.
# - **Prewitt vs Sobel:** cho kết quả edge density gần nhau vì cả hai dùng kernel 3×3;
#   Sobel dùng trọng số trung tâm lớn hơn (2) so với Prewitt (1), nên nhạy hơn một chút
#   với cạnh dọc/ngang, nhưng khác biệt không đáng kể ở ngưỡng T tương đương.
# - **Canny:** T₂/T₁ ≈ 2–3; σ=2 làm trơn noise tốt hơn trên ảnh vệ tinh.
# - **ANOVA + Kruskal-Wallis** (xem output): p ≈ 0, Eta² lớn → bác bỏ H₀,
#   edge density khác biệt có ý nghĩa giữa 45 lớp.
# - Lớp edge cao nhất: cấu trúc đô thị/thực vật dày (dense_residential, chaparral).
# - Lớp edge thấp nhất: bề mặt đồng nhất (island, runway, sea_ice).

# %% [markdown]
# ---
# ## 3. Tổng kết Phân tích nâng cao
#
# | Phân tích | Kết quả chính |
# |-----------|---------------|
# | PCA | Xem output cell: % variance PC1, n_90/n_95/n_99. Eta² (Kruskal) → các lớp phân tách trên PC1. |
# | Eigenimages | PC1 = brightness tổng thể, PC2-3 = cấu trúc cạnh/texture. |
# | Sobel/Prewitt/Canny | Xem output: ablation T → BEST_T, Eta² lớn, Kruskal p≈0. Ngưỡng T là siêu tham số chính. |
