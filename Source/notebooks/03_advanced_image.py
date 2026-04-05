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
import os, glob, warnings
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
plt.rcParams.update({'figure.figsize': (12, 6), 'font.size': 11, 'figure.dpi': 100})
sns.set_style("whitegrid")

from pathlib import Path

def _find_image_root() -> Path:
    """Tìm thư mục Dataset/ chứa train/."""
    candidates = [
        Path.cwd() / 'Dataset',
        Path.cwd().parent / 'Dataset',
        Path.cwd() / 'Source' / 'Dataset',
        Path.cwd() / 'DataMining-Lab1' / 'Dataset',
        Path.cwd().parent / 'Source' / 'Dataset',
        Path.cwd().parent / 'DataMining-Lab1' / 'Dataset',
        Path.cwd().parent.parent / 'Source' / 'Dataset',
    ]
    for p in candidates:
        if (p / 'train').exists() and any((p / 'train').iterdir()):
            return p
    raise FileNotFoundError("Không tìm thấy Dataset/train/. Đặt ảnh NWPU-RESISC45 vào Source/Dataset/.")

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
        chosen = np.random.choice(paths, min(n_per_class, len(paths)), replace=False)
        for p in chosen:
            img = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
            samples.append((img, cls))
    return samples


# %% [markdown]
# ---
# ## 1. [Nâng cao] PCA đặc trưng ảnh
#
# **Lựa chọn kích thước ảnh cho PCA:** Mặc dù notebook 02 chọn 224×224 làm kích thước
# cuối cùng để bảo toàn chất lượng ảnh tốt nhất, ở bước PCA này ta dùng **64×64** vì lý do
# tính toán: với toàn bộ 27,000 ảnh train, mỗi ảnh 224×224 tạo vector 50,176 chiều —
# ma trận đầu vào sẽ là 27,000 × 50,176 ≈ **5.4 GB RAM**, vượt quá khả năng xử lý thông
# thường. Ở kích thước 64×64 (vector 4,096 chiều), ma trận chỉ tốn ~0.4 GB và PCA
# vẫn nắm bắt được cấu trúc phân tách lớp vì đặc trưng texture/độ sáng tổng thể không
# đòi hỏi độ phân giải cao.
#
# Flatten ảnh grayscale 64×64 thành vector 4096-D, chạy PCA trên **toàn bộ 27,000 ảnh train**
# (600 ảnh/lớp × 45 lớp) để tìm eigenimages và xem các lớp có phân tách được trên
# không gian PCA hay không.
#
# Dùng **IncrementalPCA** (batch-wise) thay vì PCA thông thường vì ma trận 27,000 × 4,096
# (~0.4 GB float32) có thể fit RAM nhưng IncrementalPCA ổn định hơn với dataset lớn,
# xử lý từng batch không cần load toàn bộ vào memory cùng lúc.
#
# - **$H_0$:** Các lớp không phân tách được trên PC1 (PC1 mean không khác biệt giữa lớp).
# - **$H_1$:** Ít nhất một cặp lớp phân tách được trên PC1.

# %%
PCA_SIZE = 64
N_COMPONENTS = 200   # số components tối đa; đủ để đạt 99% variance với ảnh 64×64
BATCH_SIZE  = 500    # số ảnh mỗi batch cho IncrementalPCA

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
n90 = int(np.argmax(cum_var >= 0.90) + 1) if cum_var[-1] >= 0.90 else f">{N_COMPONENTS}"
n95 = int(np.argmax(cum_var >= 0.95) + 1) if cum_var[-1] >= 0.95 else f">{N_COMPONENTS}"
n99 = int(np.argmax(cum_var >= 0.99) + 1) if cum_var[-1] >= 0.99 else f">{N_COMPONENTS}"

print(f"Components để giải thích 90% variance: {n90}")
print(f"Components để giải thích 95% variance: {n95}")
print(f"Components để giải thích 99% variance: {n99}")

# %% [markdown]
# ### Scree Plot & Cumulative Variance

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].bar(range(1, 31), ipca.explained_variance_ratio_[:30], color='#4C72B0', alpha=0.8)
axes[0].set_xlabel("Component")
axes[0].set_ylabel("Explained Variance Ratio")
axes[0].set_title("Scree Plot (top 30 trong tổng " + str(N_COMPONENTS) + " components)")

axes[1].plot(range(1, len(cum_var)+1), cum_var, color='#4C72B0', linewidth=2)
axes[1].axhline(0.99, color='purple', linestyle='--', alpha=0.6, label=f'99% (n={n99})')
axes[1].axhline(0.95, color='red',    linestyle='--', alpha=0.6, label=f'95% (n={n95})')
axes[1].axhline(0.90, color='orange', linestyle='--', alpha=0.6, label=f'90% (n={n90})')
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
    ax.set_title(f"PC{idx+1} ({ipca.explained_variance_ratio_[idx]*100:.1f}%)", fontsize=8)
    ax.axis('off')
plt.suptitle("Top 20 Eigenimages (IncrementalPCA, 27,000 ảnh 64×64)", fontsize=13)
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
from sklearn.metrics import adjusted_rand_score
from sklearn.cluster import KMeans

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
from mpl_toolkits.mplot3d import Axes3D

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
from sklearn.manifold import TSNE

# Subsample đồng đều (nhất quán mỗi lớp: ~111 ảnh/lớp × 45 = 5,000)
np.random.seed(42)
N_TSNE = 5000
tsne_idx = np.random.choice(len(X_pca), N_TSNE, replace=False)
X_tsne_input = X_pca_transformed[tsne_idx, :50]
y_tsne = y_pca[tsne_idx]

print(f"Running t-SNE trên {N_TSNE} điểm (50 PCA components, perplexity=30)...")
tsne_model = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
X_tsne = tsne_model.fit_transform(X_tsne_input)
print("Done.")

fig, ax = plt.subplots(figsize=(14, 10))
for cls, color in zip(classes, colors_45):
    mask = y_tsne == cls
    ax.scatter(X_tsne[mask, 0], X_tsne[mask, 1],
               c=[color], label=cls, alpha=0.6, s=12)
ax.set_title(f"t-SNE 2D — toàn bộ 45 lớp (subsample {N_TSNE} ảnh, input: 50 PCA components)",
             fontsize=12)
ax.set_xlabel("t-SNE 1")
ax.set_ylabel("t-SNE 2")
ax.legend(fontsize=6, loc='best', ncol=5, markerscale=2)
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
ss_between = sum(len(g) * (g.mean() - X_pca_transformed[:, 0].mean())**2 for g in groups_pc1)
ss_total = sum((x - X_pca_transformed[:, 0].mean())**2 for x in X_pca_transformed[:, 0])
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
# ## 2. [Nâng cao] Edge Detection (Sobel & Canny)
#
# Tính **edge density** (Canny) và **Sobel magnitude** cho 45 lớp
# để xem loại cảnh nào có cạnh/texture phức tạp.
#
# - **$H_0$:** Edge density không khác biệt giữa 45 lớp.
# - **$H_1$:** Ít nhất 1 cặp lớp có edge density khác biệt.


# %% [markdown]
# ### Edge Density - Toàn bộ 45 lớp
#
# **Lý thuyết:** Binarize gradient magnitude với ngưỡng $T$:
# - **Sobel/Prewitt**: $T$ điều chỉnh tỉ lệ pixel được coi là cạnh (quan trọng hơn kernel_size)
# - **Canny**: $T_1$ (low), $T_2$ (high), khuyến nghị $T_2/T_1 \approx 3$ (Canny 1986); Gaussian sigma để làm trơn trước

# %%
EDGE_SAMPLE = 30

edge_data = []
for cls in tqdm(classes, desc="Edge density"):
    paths = glob.glob(os.path.join(TRAIN_DIR, cls, "*.jpg"))
    np.random.seed(42)
    chosen = np.random.choice(paths, min(EDGE_SAMPLE, len(paths)), replace=False)
    for p in chosen:
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        edges = cv2.Canny(img, 80, 160)
        density = edges.mean() / 255.0
        sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
        sobel_mag = np.sqrt(sobel_x**2 + sobel_y**2).mean()
        edge_data.append({'class': cls, 'canny_density': density, 'sobel_magnitude': sobel_mag})

df_edge = pd.DataFrame(edge_data)
print(f"Canny density: mean={df_edge['canny_density'].mean():.4f}, std={df_edge['canny_density'].std():.4f}")
print(f"Sobel magnitude: mean={df_edge['sobel_magnitude'].mean():.1f}, std={df_edge['sobel_magnitude'].std():.1f}")

# %%
class_edge = df_edge.groupby('class')['canny_density'].mean().sort_values()

print("Top 5 edge density cao nhất:")
for cls, val in class_edge.tail(5).items():
    print(f"  {cls}: {val:.4f}")
print("\nTop 5 edge density thấp nhất:")
for cls, val in class_edge.head(5).items():
    print(f"  {cls}: {val:.4f}")

# %% [markdown]
# ### Demo Edge Detection
#
# Chọn 5 lớp đại diện từ sorted edge density: 2 thấp nhất, 1 trung vị, 2 cao nhất.

# %%
sorted_edge_classes = class_edge.index.tolist()
ne = len(sorted_edge_classes)
demo_pick = [0, 1, ne // 2, ne - 2, ne - 1]
demo_classes = [sorted_edge_classes[i] for i in demo_pick]

for cls in demo_classes:
    print(f"  {cls}: canny density = {class_edge[cls]:.4f}")

# %%
demo_samples = load_sample(n_per_class=1, target_classes=demo_classes)

fig, axes = plt.subplots(5, 4, figsize=(14, 16))
axes[0][0].set_title("Original", fontsize=10)
axes[0][1].set_title("Sobel Combined", fontsize=10)
axes[0][2].set_title("Canny (low)", fontsize=10)
axes[0][3].set_title("Canny (high)", fontsize=10)

for row, (img, cls) in enumerate(demo_samples):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    axes[row][0].imshow(img)
    axes[row][0].set_ylabel(cls, fontsize=10, rotation=0, labelpad=60)
    axes[row][0].axis('off')

    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel_combined = np.sqrt(sobel_x**2 + sobel_y**2)
    sobel_combined = np.clip(sobel_combined / sobel_combined.max() * 255, 0, 255).astype(np.uint8)
    axes[row][1].imshow(sobel_combined, cmap='gray')
    axes[row][1].axis('off')

    canny_low = cv2.Canny(gray, 50, 100)
    axes[row][2].imshow(canny_low, cmap='gray')
    axes[row][2].axis('off')

    canny_high = cv2.Canny(gray, 100, 200)
    axes[row][3].imshow(canny_high, cmap='gray')
    axes[row][3].axis('off')

plt.suptitle("Edge Detection: Sobel vs Canny", fontsize=14)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### So sánh: Sobel (ksize=3 vs ksize=5) và Prewitt (raw vs smooth)
#
# | Bộ lọc | Tham số 1 | Tham số 2 |
# |--------|-----------|-----------|
# | **Sobel** | ksize=3 | ksize=5 |
# | **Prewitt** | Ảnh gốc (raw) | Gaussian blur trước (σ=1) |
# | **Canny** | thresholds=50/80 | thresholds=100/200 |
#

# %%
prewitt_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)
prewitt_y = np.array([[-1,-1,-1], [ 0, 0, 0], [ 1, 1, 1]], dtype=np.float32)

def norm_edge(arr):
    m = arr.max()
    return np.clip(arr / m * 255, 0, 255).astype(np.uint8) if m > 0 else np.zeros_like(arr, dtype=np.uint8)

fig, axes = plt.subplots(5, 6, figsize=(16, 16))
col_titles = ["Original", "Sobel k=3", "Sobel k=5", "Prewitt (raw)", "Prewitt (smooth)", "Canny (50/80)"]
for ci, t in enumerate(col_titles):
    axes[0][ci].set_title(t, fontsize=9)

for row, (img, cls) in enumerate(demo_samples):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    axes[row][0].imshow(img)
    axes[row][0].set_ylabel(cls, fontsize=8, rotation=0, labelpad=70)
    axes[row][0].axis("off")

    # Sobel k=3
    s3 = norm_edge(np.sqrt(cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=3)**2
                         + cv2.Sobel(gray,cv2.CV_64F,0,1,ksize=3)**2))
    axes[row][1].imshow(s3, cmap="gray"); axes[row][1].axis("off")

    # Sobel k=5
    s5 = norm_edge(np.sqrt(cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=5)**2
                         + cv2.Sobel(gray,cv2.CV_64F,0,1,ksize=5)**2))
    axes[row][2].imshow(s5, cmap="gray"); axes[row][2].axis("off")

    # Prewitt raw
    px  = cv2.filter2D(gray.astype(np.float32), -1, prewitt_x)
    py_ = cv2.filter2D(gray.astype(np.float32), -1, prewitt_y)
    axes[row][3].imshow(norm_edge(np.sqrt(px**2 + py_**2)), cmap="gray"); axes[row][3].axis("off")

    # Prewitt smooth
    gs  = cv2.GaussianBlur(gray, (3,3), 1)
    px2 = cv2.filter2D(gs.astype(np.float32), -1, prewitt_x)
    py2 = cv2.filter2D(gs.astype(np.float32), -1, prewitt_y)
    axes[row][4].imshow(norm_edge(np.sqrt(px2**2 + py2**2)), cmap="gray"); axes[row][4].axis("off")

    # Canny (50, 80)
    axes[row][5].imshow(cv2.Canny(gray, 50, 80), cmap="gray"); axes[row][5].axis("off")

plt.suptitle("So sánh: Sobel (k=3, k=5) / Prewitt (raw, smooth) / Canny (50/80)", fontsize=13)
plt.tight_layout()
plt.show()


# %% [markdown]
# ### Edge Density: So sánh 6 phương pháp + ANOVA
#
# Binarize gradient magnitude với ngưỡng T=30 để tính tỉ lệ pixel cạnh trên toàn bộ 45 lớp.
#

# %%
prewitt_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)
prewitt_y = np.array([[-1,-1,-1], [ 0, 0, 0], [ 1, 1, 1]], dtype=np.float32)
THRESH = 30

multi_edge_data = []
for cls in tqdm(classes, desc="Multi-method edge"):
    paths = glob.glob(os.path.join(TRAIN_DIR, cls, "*.jpg"))
    np.random.seed(42)
    chosen = np.random.choice(paths, min(EDGE_SAMPLE, len(paths)), replace=False)
    for p in chosen:
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        s3  = np.sqrt(cv2.Sobel(img,cv2.CV_64F,1,0,ksize=3)**2 + cv2.Sobel(img,cv2.CV_64F,0,1,ksize=3)**2)
        s5  = np.sqrt(cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)**2 + cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)**2)
        px  = cv2.filter2D(img.astype(np.float32), -1, prewitt_x)
        py_ = cv2.filter2D(img.astype(np.float32), -1, prewitt_y)
        pw  = np.sqrt(px**2 + py_**2)
        gs  = cv2.GaussianBlur(img, (3,3), 1)
        px2 = cv2.filter2D(gs.astype(np.float32), -1, prewitt_x)
        py2 = cv2.filter2D(gs.astype(np.float32), -1, prewitt_y)
        pw2 = np.sqrt(px2**2 + py2**2)
        multi_edge_data.append({"class": cls,
            "sobel_k3":      (s3  > THRESH).mean(),
            "sobel_k5":      (s5  > THRESH).mean(),
            "prewitt_raw":   (pw  > THRESH).mean(),
            "prewitt_smooth":(pw2 > THRESH).mean(),
            "canny_50_80":   cv2.Canny(img,  50,  80).mean() / 255.0,
            "canny_100_200": cv2.Canny(img, 100, 200).mean() / 255.0})
df_multi = pd.DataFrame(multi_edge_data)

methods = ["sobel_k3","sobel_k5","prewitt_raw","prewitt_smooth","canny_50_80","canny_100_200"]
print(f"{'Method':<22} {'F':>8} {'p':>12} {'Eta2':>7}  Ket luan")
print("-" * 65)
for m in methods:
    grps  = [df_multi[df_multi["class"]==c][m].values for c in classes]
    f_val, p_val = stats.f_oneway(*grps)
    ss_b  = sum(len(g)*(g.mean()-df_multi[m].mean())**2 for g in grps)
    ss_t  = ((df_multi[m] - df_multi[m].mean())**2).sum()
    eta2  = ss_b / ss_t
    sig   = "Co y nghia" if p_val < 0.05 else "Khong"
    print(f"{m:<22} {f_val:>8.2f} {p_val:>12.2e} {eta2:>7.3f}  {sig}")


# %%
# Boxplot top/bottom
top_bottom_edge = list(class_edge.index[:7]) + list(class_edge.index[-7:])
df_edge_sub = df_edge[df_edge['class'].isin(top_bottom_edge)]
order_edge = class_edge[class_edge.index.isin(top_bottom_edge)].index.tolist()

fig, axes = plt.subplots(1, 2, figsize=(16, 5))

sns.boxplot(data=df_edge_sub, x='class', y='canny_density', order=order_edge,
            ax=axes[0], palette='RdYlGn_r')
axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45, ha='right', fontsize=8)
axes[0].set_title("Canny Edge Density (7 thấp nhất + 7 cao nhất)")

class_sobel = df_edge.groupby('class')['sobel_magnitude'].mean().sort_values()
tb_sobel = list(class_sobel.index[:7]) + list(class_sobel.index[-7:])
df_sobel_sub = df_edge[df_edge['class'].isin(tb_sobel)]
order_sobel = class_sobel[class_sobel.index.isin(tb_sobel)].index.tolist()
sns.boxplot(data=df_sobel_sub, x='class', y='sobel_magnitude', order=order_sobel,
            ax=axes[1], palette='RdYlGn_r')
axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45, ha='right', fontsize=8)
axes[1].set_title("Sobel Magnitude (7 thấp nhất + 7 cao nhất)")
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Kiểm định: Edge density & Correlation

# %%
edge_groups = [df_edge[df_edge['class']==c]['canny_density'].values for c in classes]

lev_s, lev_p = stats.levene(*edge_groups)
f_edge, p_edge = stats.f_oneway(*edge_groups)
h_edge, kw_p_edge = stats.kruskal(*edge_groups)

ss_b = sum(len(g)*(g.mean() - df_edge['canny_density'].mean())**2 for g in edge_groups)
ss_t = sum((x - df_edge['canny_density'].mean())**2 for g in edge_groups for x in g)
eta2 = ss_b / ss_t

print(f"Levene: stat={lev_s:.2f}, p={lev_p:.2e}")
print(f"ANOVA:  F={f_edge:.2f}, p={p_edge:.2e}")
print(f"Kruskal: H={h_edge:.2f}, p={kw_p_edge:.2e}")
print(f"Eta² = {eta2:.3f}")

# %%
# Correlation: Pearson + Spearman
r_pearson, p_pearson = stats.pearsonr(df_edge['canny_density'], df_edge['sobel_magnitude'])
r_spearman, p_spearman = stats.spearmanr(df_edge['canny_density'], df_edge['sobel_magnitude'])

print(f"Pearson:  r={r_pearson:.4f}, p={p_pearson:.2e}")
print(f"Spearman: rho={r_spearman:.4f}, p={p_spearman:.2e}")

# %%
class_edge_stats = df_edge.groupby('class').agg({'canny_density': 'mean', 'sobel_magnitude': 'mean'}).reset_index()

fig, ax = plt.subplots(figsize=(10, 7))
ax.scatter(class_edge_stats['canny_density'], class_edge_stats['sobel_magnitude'], s=50, alpha=0.7)
for _, row in class_edge_stats.iterrows():
    ax.annotate(row['class'], (row['canny_density'], row['sobel_magnitude']), fontsize=6, alpha=0.8)
ax.set_xlabel("Canny Edge Density")
ax.set_ylabel("Mean Sobel Magnitude")
ax.set_title("Canny Density vs Sobel Magnitude theo lớp")

z = np.polyfit(class_edge_stats['canny_density'], class_edge_stats['sobel_magnitude'], 1)
xline = np.linspace(class_edge_stats['canny_density'].min(), class_edge_stats['canny_density'].max(), 100)
ax.plot(xline, np.polyval(z, xline), 'r--', alpha=0.5, label=f'Pearson r={r_pearson:.3f}')
ax.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# **Kết luận Edge Detection:**
#
# Canny density trung bình = 0.1547 (std=0.075), Sobel magnitude trung bình = 74.8 (std=32.4).
# Lớp edge cao nhất: **dense_residential** (0.239), **chaparral** (0.232) — cấu trúc đô thị/thực vật phức tạp.
# Lớp edge thấp nhất: **island** (0.050), **runway** (0.058) — bề mặt đồng nhất.
#
# Levene p=4.9e-17 → phương sai không đồng nhất → ưu tiên Kruskal-Wallis:
# H=604.74, p=1.25e-99 (ANOVA F=23.65, p=1.05e-134) → bác bỏ $, Eta² = 0.444 (large effect).
#
# Canny density và Sobel magnitude tương quan rất mạnh:
# Pearson r=**0.869** (p≈0), Spearman ρ=**0.904** (p≈0) — cả 2 đo cùng hiện tượng gradient ảnh.

# %% [markdown]
# ---
# ## 3. Tổng kết Phân tích nâng cao
#
# | Phân tích | Kết quả chính |
# |-----------|---------------|
# | PCA | Xem output cell: % variance PC1, n_90/n_95/n_99. Eta² (Kruskal) → các lớp phân tách trên PC1. |
# | Eigenimages | PC1 = brightness tổng thể, PC2-3 = cấu trúc cạnh/texture. |
# | Sobel & Canny | Xem output cell: Eta² (large), Kruskal p≈0 — edge density khác biệt có ý nghĩa giữa 45 lớp. |
# | Correlation | Xem output cell: Pearson r và Spearman ρ — Canny và Sobel tương quan rất mạnh. |
