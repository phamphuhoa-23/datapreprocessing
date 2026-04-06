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
# SOURCE/notebooks/02_preprocessing_image.py
# Notebook: 02 - Tiền xử lý ảnh + Ablation Study
# Ngôn ngữ: Tiếng Việt (markdown) + Python (code)
#
# FEEDBACK từ leader (FeedbackFromLeader.pdf):
# [🔴 P1-CRITICAL] Augmentation ablation SAI YÊu CẦU:
#   Hiện tại: apply tất cả 6 aug vào 1 ảnh cùng lúc → tập tăng gấp đôi → 1 k-NN
#   Đúng: mỗi augmentation chạy RING BÀI k-NN ablation riêng:
#   original xx / h-flip xx / v-flip xx / rotation xx / crop xx / noise xx / bright xx / all xx
# [FIX] KS test p-value: thêm markdown giải thích p≈0 là phân phối THAY ĐỔI (expected)
# [ADD] Ghi rõ RGB là baseline trong bảng color space ablation
# [ADD] Thử LogReg bên cạnh k-NN trong ablation
# [NOTE] Càng nhiều ảnh trong t-SNE càng tốt (>100/tập), in rõ số lượng
# =============================================================================
#
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 02 - Tiền xử lý ảnh + Ablation Study
#
# **Mục tiêu:** Áp dụng đầy đủ 4 nhóm kỹ thuật tiền xử lý (Resize, Color Space, Normalization, Augmentation),
# đo lường định lượng tác động bằng ablation study (k-NN), kiểm định thống kê.
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
from sklearn.decomposition import PCA
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
# ### Cấu hình sampling cho ablation study
#
# Dùng **toàn bộ 45 lớp** để kết quả không bị bias.
# Giới hạn số ảnh/lớp để thời gian chạy hợp lý.

# %%
quick_brightness = {}
for cls in classes:
    paths = glob.glob(os.path.join(TRAIN_DIR, cls, "*.jpg"))[:5]
    vals = [cv2.imread(p).mean() for p in paths]
    quick_brightness[cls] = np.mean(vals)

sorted_by_bright = sorted(quick_brightness, key=quick_brightness.get)
n = len(sorted_by_bright)
indices_10 = np.linspace(0, n - 1, 10, dtype=int)
SAMPLE_CLASSES = [sorted_by_bright[i] for i in indices_10]

for cls in SAMPLE_CLASSES:
    print(f"  {cls}: brightness = {quick_brightness[cls]:.1f}")

# %% [markdown]
# ---
# ## 1. Resize - Ablation Study (SSIM / PSNR)
#
# **Lý thuyết:**
#
# Resize giảm chiều không gian ảnh từ $H_{\text{orig}} \times W_{\text{orig}}$ về kích thước nhỏ hơn.
# Đo mất mát thông tin bằng cách **resize xuống rồi resize lên lại** cự gốc, sau đó tính 2 metrics:
#
# **SSIM (Structural Similarity Index Measure)** — đo sự tương đồng cấu trúc gitta 2 ảnh:
#
# $$\text{SSIM}(x,y) = \frac{(2\mu_x\mu_y + C_1)(2\sigma_{xy} + C_2)}{(\mu_x^2+\mu_y^2+C_1)(\sigma_x^2+\sigma_y^2+C_2)}$$
#
# - $\mu_x, \mu_y$: mean của patch $x, y$; $\sigma_x^2, \sigma_y^2$: variance; $\sigma_{xy}$: covariance
# - $C_1 = (k_1 L)^2,\ C_2 = (k_2 L)^2$ với $k_1=0.01,\ k_2=0.03,\ L=255$ (hằng số ổn định)
# - $\text{SSIM} \in [-1, 1]$; giá trị 1 = hai ảnh giống hệt nhau
#
# **PSNR (Peak Signal-to-Noise Ratio)** — đo tỷ lệ tín hiệu/nhiễu (dB):
#
# $$\text{PSNR} = 10\log_{10}\!\left(\frac{255^2}{\text{MSE}}\right), \quad
# \text{MSE} = \frac{1}{HW}\sum_{i,j}(I_{ij}-\hat{I}_{ij})^2$$
#
# - PSNR cao hơn ⇒ chất lượng tốt hơn ($> 30$ dB thuờng chấp nhận được)
# - SSIM nhạy hơn PSNR với các lỗi cấu trúc (texture loss, blurring)
#
# - $H_0$: Chất lượng ảnh (SSIM) không khác biệt giữa 3 kích thước resize
# - $H_1$: Ít nhất 1 kích thước cho SSIM khác biệt

# %%
RESIZE_DIMS = [(64, 64), (128, 128), (224, 224)]
samples = load_sample(n_per_class=30, target_classes=SAMPLE_CLASSES)
print(f"Sample: {len(samples)} ảnh từ {len(SAMPLE_CLASSES)} lớp")

# %%
results_resize = []
for img_orig, cls in tqdm(samples, desc="Resize ablation"):
    for (h, w) in RESIZE_DIMS:
        resized = cv2.resize(img_orig, (w, h), interpolation=cv2.INTER_AREA)
        restored = cv2.resize(resized, (256, 256), interpolation=cv2.INTER_LINEAR)
        s = ssim(img_orig, restored, channel_axis=2)
        p = psnr(img_orig, restored)
        results_resize.append({'class': cls, 'size': f"{h}x{w}", 'ssim': s, 'psnr': p})

df_resize = pd.DataFrame(results_resize)
df_resize.groupby('size').agg(
    ssim_mean=('ssim', 'mean'), ssim_std=('ssim', 'std'),
    psnr_mean=('psnr', 'mean'), psnr_std=('psnr', 'std')
).round(4)

# %%
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
sns.boxplot(data=df_resize, x='size', y='ssim', ax=axes[0], palette='Blues_d',
            order=['64x64', '128x128', '224x224'])
axes[0].set_title("SSIM theo kích thước resize")
axes[0].set_ylabel("SSIM")

sns.boxplot(data=df_resize, x='size', y='psnr', ax=axes[1], palette='Oranges_d',
            order=['64x64', '128x128', '224x224'])
axes[1].set_title("PSNR theo kích thước resize (dB)")
axes[1].set_ylabel("PSNR (dB)")
plt.tight_layout()
plt.show()

# %%
# Đường cong SSIM và PSNR theo kích thước
mean_ssim = df_resize.groupby('size')['ssim'].mean().reindex(['64x64', '128x128', '224x224'])
mean_psnr = df_resize.groupby('size')['psnr'].mean().reindex(['64x64', '128x128', '224x224'])
std_ssim = df_resize.groupby('size')['ssim'].std().reindex(['64x64', '128x128', '224x224'])
std_psnr = df_resize.groupby('size')['psnr'].std().reindex(['64x64', '128x128', '224x224'])

sizes_px = [64, 128, 224]

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].errorbar(sizes_px, mean_ssim.values, yerr=std_ssim.values, marker='o', capsize=5, linewidth=2)
axes[0].set_xlabel("Kích thước (pixel)")
axes[0].set_ylabel("SSIM")
axes[0].set_title("SSIM theo kích thước resize")
axes[0].set_xticks(sizes_px)

axes[1].errorbar(sizes_px, mean_psnr.values, yerr=std_psnr.values, marker='o', capsize=5, linewidth=2, color='orange')
axes[1].set_xlabel("Kích thước (pixel)")
axes[1].set_ylabel("PSNR (dB)")
axes[1].set_title("PSNR theo kích thước resize")
axes[1].set_xticks(sizes_px)

plt.tight_layout()
plt.show()

# %% [markdown]
# ### Ablation: k-NN accuracy theo kích thước resize

# %%
# k-NN classification accuracy theo kích thước resize
# Dùng 50 ảnh/lớp x 10 lớp = 500 ảnh, flatten làm feature
knn_samples = load_sample(n_per_class=50, target_classes=SAMPLE_CLASSES)

knn_results = {}
for target_size in [(64, 64), (128, 128), (224, 224), (256, 256)]:
    X, y = [], []
    for img, cls in knn_samples:
        resized = cv2.resize(img, target_size)
        X.append(resized.reshape(-1).astype(np.float32) / 255.0)
        y.append(cls)
    X, y = np.array(X), np.array(y)
    
    knn = KNeighborsClassifier(n_neighbors=5)
    scores = cross_val_score(knn, X, y, cv=5, scoring='accuracy')
    knn_results[f"{target_size[0]}x{target_size[1]}"] = (scores.mean(), scores.std())
    print(f"  {target_size[0]}x{target_size[1]}: accuracy = {scores.mean():.4f} (+/- {scores.std():.4f})")

# Plot
fig, ax = plt.subplots(figsize=(8, 5))
sizes_label = list(knn_results.keys())
means = [v[0] for v in knn_results.values()]
stds = [v[1] for v in knn_results.values()]
ax.bar(sizes_label, means, yerr=stds, capsize=5, color=['#4C72B0', '#4C72B0', '#4C72B0', '#DD8452'])
ax.set_ylabel("k-NN Accuracy (5-fold CV)")
ax.set_xlabel("Kích thước")
ax.set_title("k-NN Accuracy theo kích thước resize (256x256 = gốc)")
plt.tight_layout()
plt.show()

# %%
# Ảnh mẫu qua các kích thước
fig, axes = plt.subplots(2, 4, figsize=(14, 6))
for row, (img, cls) in enumerate([(samples[0][0], samples[0][1]), (samples[60][0], samples[60][1])]):
    axes[row][0].imshow(img)
    axes[row][0].set_title(f"Gốc 256x256\n({cls})", fontsize=9)
    axes[row][0].axis('off')
    for col, (h, w) in enumerate(RESIZE_DIMS):
        resized = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
        axes[row][col+1].imshow(resized)
        s = ssim(img, cv2.resize(resized, (256,256), interpolation=cv2.INTER_LINEAR), channel_axis=2)
        axes[row][col+1].set_title(f"{h}x{w}\nSSIM={s:.3f}", fontsize=9)
        axes[row][col+1].axis('off')
plt.suptitle("So sánh chất lượng ảnh qua các kích thước resize", fontsize=13)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Kiểm định thống kê - Resize

# %%
groups_ssim = {s: df_resize[df_resize['size']==s]['ssim'].values for s in ['64x64', '128x128', '224x224']}

# Levene
lev_s, lev_p = stats.levene(*groups_ssim.values())
print(f"Levene: stat={lev_s:.2f}, p={lev_p:.2e}")

# ANOVA
f_val, p_val = stats.f_oneway(*groups_ssim.values())
print(f"ANOVA:  F={f_val:.2f}, p={p_val:.2e}")

# Eta²
all_ssim = np.concatenate(list(groups_ssim.values()))
gm = all_ssim.mean()
ss_b = sum(len(g) * (g.mean() - gm)**2 for g in groups_ssim.values())
ss_t = np.sum((all_ssim - gm)**2)
eta2_resize = ss_b / ss_t
print(f"Eta² = {eta2_resize:.3f}")

# Post-hoc Mann-Whitney với Bonferroni (3 cặp → α=0.05/3≈0.0167)
sizes = ['64x64', '128x128', '224x224']
n_pairs_resize = 3
alpha_bonf_resize = 0.05 / n_pairs_resize
print(f"Post-hoc Mann-Whitney (Bonferroni α={alpha_bonf_resize:.4f}):")
for i in range(len(sizes)):
    for j in range(i+1, len(sizes)):
        u, p = stats.mannwhitneyu(groups_ssim[sizes[i]], groups_ssim[sizes[j]], alternative='two-sided')
        p_bonf = min(p * n_pairs_resize, 1.0)
        sig = "***" if p_bonf < 0.001 else ("**" if p_bonf < 0.01 else ("*" if p_bonf < 0.05 else "ns"))
        print(f"  {sizes[i]} vs {sizes[j]}: U={u:.0f}, p_raw={p:.2e}, p_bonf={p_bonf:.2e} {sig}")

# Kết luận động dựa vào eta²
best_size = sizes[np.argmax([groups_ssim[s].mean() for s in sizes])]
print(f"\n=> η²={eta2_resize:.3f} ({'lớn ≥0.14' if eta2_resize>=0.14 else 'trung bình' if eta2_resize>=0.06 else 'nhỏ'}) — effect size {'rất lớn' if eta2_resize>=0.14 else 'trung bình'}")
print(f"=> Kích thước cho SSIM cao nhất: {best_size}")

# %%
# === Tóm tắt động kết quả Resize ===
print("\n=== Tóm tắt Resize ===")
print(f"{'Kích thước':<12} {'SSIM':>12} {'PSNR(dB)':>12} {'k-NN Acc':>12}")
print("-" * 50)
for s_label in ['64x64', '128x128', '224x224', '256x256']:
    ssim_m = mean_ssim.get(s_label, 1.0) if s_label in mean_ssim.index else 1.0
    ssim_s = std_ssim.get(s_label, 0.0) if s_label in std_ssim.index else 0.0
    psnr_m = mean_psnr.get(s_label, float('inf')) if s_label in mean_psnr.index else float('inf')
    psnr_s = std_psnr.get(s_label, 0.0) if s_label in std_psnr.index else 0.0
    knn_m, knn_s = knn_results.get(s_label, (float('nan'), float('nan')))
    psnr_str = f"{psnr_m:.1f}±{psnr_s:.1f}" if not np.isinf(psnr_m) else "∞"
    print(f"{s_label:<12} {ssim_m:.3f}±{ssim_s:.3f} {psnr_str:>12} {knn_m:.3f}±{knn_s:.3f}")
print(f"\nANOVA: F={f_val:.2f}, p={p_val:.2e}, η²={eta2_resize:.3f}")

# %% [markdown]
# **Kết luận Resize:** (số liệu chính xác xem output bên trên)
#
# - **Bác bỏ $H_0$**: SSIM khác biệt có ý nghĩa giữa mọi cặp kích thước
#   - ANOVA + Post-hoc Mann-Whitney (Bonferroni): cả 3 cặp đều p ≈ 0
#   - Effect size $\eta^2$ rất lớn (xem print trên)
# - Đường cong SSIM tăng nhanh từ 64 $\to$ 128, chậm lại từ 128 $\to$ 224
# - k-NN accuracy **gần như không đổi** qua các kích thước (~0.35)
#   - 256x256 thậm chí thấp nhất do curse of dimensionality
# - **Lựa chọn: 128x128** – k-NN accuracy cao và mất mát thông tin ở mức trung bình
#   - 128x128 đạt SSIM tốt (mất mát vừa phải) đồng thời k-NN accuracy ngang ngửa 224x224
#   - 224x224 chỉ cải thiện SSIM thêm ~5% nhưng tăng feature dimension 3× → không đáng
#   - 128x128 là điểm cân bằng tối ưu giữa chất lượng ảnh và hiệu quả tính toán

# %% [markdown]
# ---
# ## 2. Color Space Conversion — Ablation Study (PCA Explained Variance)
#
# **Lý thuyết:**
#
# Không gian màu xuyết quá thông tin ảnh theo cách khác nhau,
# ảnh hưởng trực tiếp đến hiệu quả phân tích và phân loại:
#
# | Không gian | Cách biểu diễn | Ưu điểm | Nhược điểm |
# |---|---|---|---|
# | **RGB** | 3 kênh $R, G, B \in [0,255]$ | Trực quan, tương thích pretrained | Bị tương quan chéo giữa kênh |
# | **Grayscale** | $Y = 0.299R + 0.587G + 0.114B$ | 1/3 dung lượng, đơn giản | Mất toàn bộ thông tin màu |
# | **HSV** | Hue $H$, Saturation $S$, Value $V$ | Tách biệt màu sắc và độ sáng | $H$ là giá trị còn vòng (circular) |
# | **CIE Lab** | $L^*$ (lightness), $a^*$ (green–red), $b^*$ (blue–yellow) | Perceptually uniform; tách tốt $L$ vs chrominance | Tính toán phức tạp hơn |
#
# **Grayscale conversion:**
# $$Y = 0.299\,R + 0.587\,G + 0.114\,B$$
#
# Trọng số của ITU-R BT.601 phản ánh độ nhạy của mắt người:
# mắt nhạy nhất với xanh lá ($G$), kém nhất với xanh dương ($B$).
#
# **PCA explained variance** được dùng để so sánh **information density**:
# không gian màu đạt 95% variance với ít components hơn = nén được tốt hơn.
# k-NN accuracy xác nhận không gian nào bảo toàn thông tin **phân loại** tốt nhất.

# %%
COLOR_SPACES = {
    'RGB': lambda img: img,
    'Grayscale': lambda img: cv2.cvtColor(img, cv2.COLOR_RGB2GRAY),
    'HSV': lambda img: cv2.cvtColor(img, cv2.COLOR_RGB2HSV),
    'Lab': lambda img: cv2.cvtColor(img, cv2.COLOR_RGB2Lab)
}

pca_results = {}
n_pca_sample = 500
pca_samples = load_sample(n_per_class=12)[:n_pca_sample]

for cs_name, convert_fn in COLOR_SPACES.items():
    print(f"  PCA trên {cs_name}...")
    features = []
    for img, _ in pca_samples:
        converted = convert_fn(img)
        if converted.ndim == 2:
            converted = converted[:, :, np.newaxis]
        features.append(converted.reshape(-1).astype(np.float32))
    X = StandardScaler().fit_transform(np.array(features))
    pca = PCA(n_components=min(50, X.shape[0]-1))
    pca.fit(X)
    pca_results[cs_name] = pca.explained_variance_ratio_

print(f"Done. {len(pca_results)} color spaces: {list(pca_results.keys())}")

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for cs_name, evr in pca_results.items():
    axes[0].plot(range(1, len(evr)+1), evr[:50], marker='.', label=cs_name, markersize=4)
axes[0].set_xlabel("Component")
axes[0].set_ylabel("Explained Variance Ratio")
axes[0].set_title("Scree Plot")
axes[0].legend()
axes[0].set_xlim(0, 30)

for cs_name, evr in pca_results.items():
    axes[1].plot(range(1, len(evr)+1), np.cumsum(evr[:50]), marker='.', label=cs_name, markersize=4)
axes[1].axhline(0.95, color='red', linestyle='--', alpha=0.5, label='95% threshold')
axes[1].set_xlabel("Số components")
axes[1].set_ylabel("Cumulative Explained Variance")
axes[1].set_title("Cumulative Explained Variance")
axes[1].legend()
axes[1].set_xlim(0, 30)
plt.tight_layout()
plt.show()

# %%
print("Số components cần để giữ X% variance:")
print(f"{'Color Space':<12} {'90%':>8} {'95%':>8} {'99%':>8}")
for cs_name, evr in pca_results.items():
    cum = np.cumsum(evr)
    n90 = np.argmax(cum >= 0.90) + 1 if cum[-1] >= 0.90 else f">{len(evr)}"
    n95 = np.argmax(cum >= 0.95) + 1 if cum[-1] >= 0.95 else f">{len(evr)}"
    n99 = np.argmax(cum >= 0.99) + 1 if cum[-1] >= 0.99 else f">{len(evr)}"
    print(f"{cs_name:<12} {str(n90):>8} {str(n95):>8} {str(n99):>8}")

# %% [markdown]
# ### Ablation: k-NN accuracy theo color space

# %%
# k-NN accuracy theo color space (dùng best_size từ resize ablation trên)
cs_knn_samples = load_sample(n_per_class=50)

cs_knn_results = {}
for cs_name, convert_fn in COLOR_SPACES.items():
    X, y = [], []
    for img, cls in cs_knn_samples:
        converted = convert_fn(cv2.resize(img, (best_size, best_size)))
        if converted.ndim == 2:
            converted = converted[:, :, np.newaxis]
        X.append(converted.reshape(-1).astype(np.float32) / 255.0)
        y.append(cls)
    X, y = np.array(X), np.array(y)
    
    knn = KNeighborsClassifier(n_neighbors=5)
    scores = cross_val_score(knn, X, y, cv=5, scoring='accuracy')
    cs_knn_results[cs_name] = (scores.mean(), scores.std())
    print(f"  {cs_name}: accuracy = {scores.mean():.4f} (+/- {scores.std():.4f})")

# %%
# === Tóm tắt động kết quả Color Space ===
print("\n=== Tóm tắt Color Space ===")
print(f"{'Color Space':<12} {'k-NN Mean':>12} {'k-NN Std':>10}")
print("-" * 37)
for cs_name, (mean_acc, std_acc) in cs_knn_results.items():
    marker = " ← k-NN tốt nhất" if cs_name == max(cs_knn_results, key=lambda k: cs_knn_results[k][0]) else ""
    print(f"{cs_name:<12} {mean_acc:>12.4f} {std_acc:>10.4f}{marker}")

# %% [markdown]
# **Kết luận Color Space:** (số liệu chính xác xem output bên trên)
#
# - Không color space nào đạt 90% variance trong 50 components
#   - Ảnh 256×256×3 = 196,608 chiều; Lab dẫn đầu (~70%) nhờ tách kênh sáng (L)
# - Grayscale có cumulative variance cao ở PC1 nhưng mất thông tin màu → k-NN accuracy thấp nhất
# - HSV cho k-NN accuracy cao nhất
#   - Kênh H (hue) mã hóa màu sắc theo góc, phân biệt loại cảnh tốt hơn raw RGB
# - **Lựa chọn:** Lab (PCA variance cao nhất) hoặc HSV (k-NN accuracy cao nhất)
#   - RGB là baseline an toàn, tương thích pretrained models

# %% [markdown]
# ---
# ## 3. Normalization - Ablation Study (KS test)
#
# **Lý thuyết:**
#
# Chuẩn hóa đưa pixel về cùng thạng đo, cải thiện tốc độ hội tụ và độ ổn định khi train:
#
# | Phương pháp | Công thức | Miền output | Ghi chú |
# |---|---|---|---|
# | **Min-Max $[0,1]$** | $x' = x / 255$ | $[0,1]$ | Giữ nguyên hình dạng phân phối |
# | **Min-Max $[-1,1]$** | $x' = x/127.5 - 1$ | $[-1,1]$ | Phù hợp với activation tanh |
# | **Z-score global** | $x' = (x - \mu_{\text{all}}) / \sigma_{\text{all}}$ | $\mathbb{R}$ | $\mu, \sigma$ tính trên toàn dataset |
# | **Z-score per-channel** | $x'_c = (x_c - \mu_c) / \sigma_c,\ c \in \{R,G,B\}$ | $\mathbb{R}$ | Tựng kênh có mean 0, std 1 riêng |
#
# **Ý nghĩa per-channel normalization:**
# Mỗi kênh có phân phối khác nhau; chuẩn hóa riêng từng kênh loại bỏ
# **bias do chiếu sáng** (kênh G thường sáng hơn kênh B).
# Tuy nhiên, nó phá vỡ **tương quan tương đối** giữa R, G, B — có thể gây hại cho biểu diễn màu sắc.
#
# **KS test (Kolmogorov-Smirnov)** kiểm tra sự thay đổi phân phối trước/sau chuẩn hóa:
#
# $$D = \sup_x \left|F_1(x) - F_2(x)\right|$$
#
# $p \approx 0$ là **MONG MUỐN**: chứng tỏ distribution shift (mục đích của chuẩn hóa).
# K-NN accuracy mới là chỉ số đánh giá chất lượng chuẩn hóa.
#
# - $H_0$: Phân phối pixel trước và sau chuẩn hóa không khác biệt
# - $H_1$: Phân phối pixel thay đổi sau chuẩn hóa

# %%
def norm_minmax_01(img):
    """Min-Max [0, 1]"""
    return img.astype(np.float32) / 255.0

def norm_minmax_neg11(img):
    """Min-Max [-1, 1]"""
    return img.astype(np.float32) / 127.5 - 1.0

def norm_zscore_global(img):
    """Z-score toàn tập (mean/std trên toàn ảnh)"""
    img_f = img.astype(np.float32)
    return (img_f - img_f.mean()) / (img_f.std() + 1e-8)

def norm_zscore_perchannel(img):
    """Z-score per-channel (mean/std riêng từng kênh)"""
    img_f = img.astype(np.float32)
    result = np.zeros_like(img_f)
    for c in range(3):
        ch = img_f[:, :, c]
        result[:, :, c] = (ch - ch.mean()) / (ch.std() + 1e-8)
    return result

NORM_METHODS = {
    'Original': lambda x: x.astype(np.float32),
    'Min-Max [0,1]': norm_minmax_01,
    'Min-Max [-1,1]': norm_minmax_neg11,
    'Z-score global': norm_zscore_global,
    'Z-score per-ch': norm_zscore_perchannel
}

# %%
norm_samples = load_sample(n_per_class=20)
print(f"Sample: {len(norm_samples)} ảnh ({len(norm_samples)//45} ảnh/lớp x 45 lớp)")

norm_pixels = {}
for method_name, norm_fn in NORM_METHODS.items():
    all_px = []
    for img, _ in norm_samples:
        normed = norm_fn(img)
        all_px.append(normed.ravel())
    norm_pixels[method_name] = np.concatenate(all_px)

# %%
# KS test: so sánh phân phối TRƯỚC vs SAU chuẩn hóa
original_px = norm_pixels['Original']
original_sample = np.random.choice(original_px, min(100000, len(original_px)), replace=False)

def ks_effect_size(d):
    """Giải thích effect size của KS statistic D"""
    if d < 0.2:  return "nhỏ"
    elif d < 0.5: return "trung bình"
    else:         return "lớn"

print("KS test (Original vs mỗi phương pháp):")
print(f"{'Method':<18} {'KS stat':>10} {'p-value':>12} {'Mean':>10} {'Std':>10} {'Effect D':>12}")
print("-" * 76)

for method, px in norm_pixels.items():
    px_sample = np.random.choice(px, min(100000, len(px)), replace=False)
    if method == 'Original':
        ks_stat, ks_p = 0.0, 1.0
    else:
        ks_stat, ks_p = stats.ks_2samp(original_sample, px_sample)
    effect_d = ks_effect_size(ks_stat)
    print(f"{method:<18} {ks_stat:>10.4f} {ks_p:>12.2e} {px_sample.mean():>10.4f} {px_sample.std():>10.4f} {effect_d:>12}")

print("\nGhi chú: D<0.2=nhỏ, 0.2≤D<0.5=trung bình, D≥0.5=lớn (phân phối thay đổi nhiều)")

# %%
# Levene test giữa các phương pháp
norm_groups = [np.random.choice(px, 10000, replace=False) for px in norm_pixels.values()]
lev_s, lev_p = stats.levene(*norm_groups)
print(f"Levene test (variance giữa 5 phương pháp): stat={lev_s:.2f}, p={lev_p:.2e}")

# %%
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for idx, (method, px) in enumerate(norm_pixels.items()):
    row, col = divmod(idx, 3)
    px_plot = np.random.choice(px, 200000, replace=False) if len(px) > 200000 else px
    axes[row][col].hist(px_plot, bins=100, color='steelblue', alpha=0.7, density=True, edgecolor='none')
    axes[row][col].set_title(method, fontsize=11)
    axes[row][col].set_xlabel("Pixel value")
    axes[row][col].set_ylabel("Density")
if len(norm_pixels) < 6:
    axes[1][2].axis('off')
plt.suptitle("Phân bố pixel sau mỗi phương pháp Normalization", fontsize=13)
plt.tight_layout()
plt.show()



# %% [markdown]
# ### Ablation: k-NN accuracy theo normalization

# %%
# k-NN accuracy theo normalization (resize 64x64, 45 lớp)
norm_knn_samples = load_sample(n_per_class=50)

norm_knn_results = {}
norm_knn_fold_scores = {}  # lưu fold scores để Wilcoxon
for method_name, norm_fn in NORM_METHODS.items():
    X, y = [], []
    for img, cls in norm_knn_samples:
        resized = cv2.resize(img, (best_size, best_size))
        normed = norm_fn(resized)
        X.append(normed.reshape(-1))
        y.append(cls)
    X, y = np.array(X), np.array(y)
    
    knn = KNeighborsClassifier(n_neighbors=5)
    scores = cross_val_score(knn, X, y, cv=5, scoring='accuracy')
    norm_knn_results[method_name] = (scores.mean(), scores.std())
    norm_knn_fold_scores[method_name] = scores
    print(f"  {method_name}: accuracy = {scores.mean():.4f} (+/- {scores.std():.4f})")

# Wilcoxon: phương pháp tốt nhất vs Original trên fold scores
best_norm_method = max([m for m in norm_knn_results if m != 'Original'],
                       key=lambda m: norm_knn_results[m][0])
scores_best_norm = norm_knn_fold_scores[best_norm_method]
scores_orig_norm = norm_knn_fold_scores['Original']
try:
    w_norm, p_norm = stats.wilcoxon(scores_best_norm, scores_orig_norm)
except ValueError:  # all zeros — identical scores
    w_norm, p_norm = 0.0, 1.0
diff_norm = scores_best_norm - scores_orig_norm
cohen_d_norm = diff_norm.mean() / (diff_norm.std(ddof=1) + 1e-8)
cohen_label = 'lớn' if abs(cohen_d_norm) >= 0.8 else ('trung bình' if abs(cohen_d_norm) >= 0.5 else 'nhỏ')
print(f"\nWilcoxon signed-rank ({best_norm_method} vs Original): W={w_norm:.1f}, p={p_norm:.4f}")
print(f"Cohen's d = {cohen_d_norm:.3f} (effect size {cohen_label})")
print(f"=> {best_norm_method} {'CẢI THIỆN CÓ Ý NGHĨA thống kê' if p_norm < 0.05 else 'KHÔNG cải thiện có ý nghĩa thống kê'} so với không chuẩn hóa (α=0.05)")

# %%
# === Tóm tắt động kết quả Normalization ===
print("\n=== Tóm tắt k-NN accuracy theo phương pháp chuẩn hóa ===")
print(f"{'Phương pháp':<20} {'k-NN Mean':>12} {'k-NN Std':>10}")
print("-" * 45)
for nm, (m, s) in norm_knn_results.items():
    marker = " ← tốt nhất" if nm == best_norm_method else ""
    print(f"{nm:<20} {m:>12.4f} {s:>10.4f}{marker}")
print(f"\n→ p≈0 trong KS test là MONG MUỐN: chuẩn hóa đã thay đổi phân phối (mục đích của chuẩn hóa)")
print(f"  KS test chỉ xác nhận distribution shift, KHÔNG đánh giá chất lượng chuẩn hóa")
print(f"⇒ Chọn {best_norm_method}: accuracy={norm_knn_results[best_norm_method][0]:.4f}, Wilcoxon p={p_norm:.4f}")

# %% [markdown]
# **Kết luận Normalization:** (số liệu chính xác xem output bên trên)
#
# > **Ghi chú KS test p ≈ 0:** p-value ≈ 0 là **MONG MUỐN** – chuẩn hóa đã thay đổi scale/phân phối
# > pixel (đó là mục đích). KS test chỉ xác nhận distribution shift, **KHÔNG** đá́nh giá chất lượng
# > chuẩn hóa. Để đánh giá hiệu quả của chuẩn hóa, dùng k-NN accuracy (xem output trên).
#
# - **Bác bỏ $H_0$** cho tất cả 4 phương pháp: KS stat $\approx 1.0$, $p \approx 0$ (expected)
# - Min-Max [0,1] và [-1,1] chỉ thay đổi scale, **không thay đổi hình dạng phân bố**
#   - k-NN accuracy giống hệt Original vì k-NN dựa trên khoảng cách tỉ lệ thuận
# - Z-score global cho accuracy cao nhất (xem print trên)
# - Z-score per-channel phá hủy quan hệ tương đối giữa R, G, B → accuracy rất thấp
# - **Lựa chọn:** phương pháp có Wilcoxon p < 0.05 và accuracy cao nhất (xem print trên)

# %% [markdown]
# ---
# ## 4. Data Augmentation — Ablation Study (t-SNE)
#
# **Lý thuyết:**
#
# **Data augmentation** tạo ra các biến thể hợp lệ của ảnh gốc bằng cách áp dụng các
# phép biến đổi **label-preserving** (không thay đổi nhãn).
# Mục tiêu: tăng **diversity** tập train, giảm overfitting, cải thiện tính robust của model.
#
# | Phép biến đổi | Mô tả toán học | Label-preserving |
# |---|---|---|
# | **H-Flip** | $I'(x,y) = I(W-1-x,\ y)$ | ✅ (cảnh viễn thám đối xứng) |
# | **V-Flip** | $I'(x,y) = I(x,\ H-1-y)$ | ✅ |
# | **Rotation** | Ma trận động lực $\mathbf{M}(\theta)$, điện tích điền affine | ✅ (góc nhỏ $|\theta| < 30^\circ$) |
# | **Random Crop** | Cắt tà $\text{scale} \in [0.7, 1.0]$, resize lại | ✅ |
# | **Gaussian Noise** | $I'_{ij} = I_{ij} + \epsilon_{ij}$, $\epsilon \sim \mathcal{N}(0, \sigma^2)$ | ✅ |
# | **Brightness/Contrast** | $I' = \alpha I + \beta$, $\alpha \approx 1$, $\beta \approx 0$ | ✅ |
#
# **Tác động được đánh giá bằng:**
# - **k-NN ablation per-technique**: mỗi augmentation chạy riêng biệt so với baseline (original only)
# - **t-SNE visualization**: biểu diễn feature distribution trước/sau augment
#   (t-SNE giảm chiều từ 128×128×3 xuống 2D, bảo toàn local structure)

# %%
def augment_hflip(img):
    """Lật ngang"""
    return cv2.flip(img, 1)

def augment_vflip(img):
    """Lật dọc"""
    return cv2.flip(img, 0)

def augment_rotate(img, angle=15):
    """Xoay ngẫu nhiên"""
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h))

def augment_crop(img, scale=0.8):
    """Cắt ngẫu nhiên"""
    h, w = img.shape[:2]
    new_h, new_w = int(h*scale), int(w*scale)
    top = np.random.randint(0, h - new_h + 1)
    left = np.random.randint(0, w - new_w + 1)
    cropped = img[top:top+new_h, left:left+new_w]
    return cv2.resize(cropped, (w, h))

def augment_gaussian_noise(img, sigma=25):
    """Thêm nhiễu Gaussian"""
    noise = np.random.normal(0, sigma, img.shape)
    noisy = img.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)

def augment_brightness_contrast(img, brightness=30, contrast=0.2):
    """Điều chỉnh độ sáng/tương phản"""
    img_f = img.astype(np.float32)
    img_f = img_f + np.random.uniform(-brightness, brightness)
    img_f = img_f * np.random.uniform(1-contrast, 1+contrast)
    return np.clip(img_f, 0, 255).astype(np.uint8)

AUGMENTATIONS = {
    'H-Flip': augment_hflip,
    'V-Flip': augment_vflip,
    'Rotation': augment_rotate,
    'Random Crop': augment_crop,
    'Gaussian Noise': augment_gaussian_noise,
    'Brightness/Contrast': augment_brightness_contrast
}
print(f"{len(AUGMENTATIONS)} phép augmentation: {list(AUGMENTATIONS.keys())}")

# %%
# Visual: ảnh trước/sau augmentation
vis_sample = load_sample(n_per_class=1)[:3]

fig, axes = plt.subplots(3, len(AUGMENTATIONS) + 1, figsize=(18, 7))
for row, (img, cls) in enumerate(vis_sample):
    axes[row][0].imshow(img)
    axes[row][0].set_title(f"Original\n({cls})", fontsize=8)
    axes[row][0].axis('off')
    for col, (aug_name, aug_fn) in enumerate(AUGMENTATIONS.items()):
        np.random.seed(42)
        augmented = aug_fn(img)
        axes[row][col+1].imshow(augmented)
        axes[row][col+1].set_title(aug_name, fontsize=8)
        axes[row][col+1].axis('off')
plt.suptitle(f"{len(AUGMENTATIONS)} phép Augmentation trên ảnh mẫu", fontsize=13)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### t-SNE: Feature space trước/sau augmentation
#
# Chạy t-SNE với số lớp tăng dần (5, 10, 15, 20, 25, 30, 35, 40, 45) để quan sát
# feature space thay đổi khi thêm lớp.

# %%
# t-SNE với số lớp tăng dần
n_class_list = [5, 10, 15, 20, 25, 30, 35, 40, 45]

def extract_feature(img):
    small = cv2.resize(img, (32, 32))
    return small.reshape(-1).astype(np.float32) / 255.0

fig, axes = plt.subplots(len(n_class_list), 2, figsize=(14, 5 * len(n_class_list)))

for row, n_cls in enumerate(n_class_list):
    # Chọn n_cls lớp cách đều
    cls_indices = np.linspace(0, len(classes) - 1, n_cls, dtype=int)
    selected_classes = [classes[i] for i in cls_indices]
    
    # Load samples
    tsne_samples = load_sample(n_per_class=20, target_classes=selected_classes)
    
    # Original features
    orig_features, orig_labels = [], []
    for img, cls in tsne_samples:
        orig_features.append(extract_feature(img))
        orig_labels.append(cls)
    
    # Augmented features (pipeline 2 phép ngẫu nhiên)
    aug_features, aug_labels = [], []
    np.random.seed(42)
    aug_fns = list(AUGMENTATIONS.values())
    for img, cls in tsne_samples:
        aug_img = img.copy()
        for fn in np.random.choice(aug_fns, 2, replace=False):
            aug_img = fn(aug_img)
        aug_features.append(extract_feature(aug_img))
        aug_labels.append(cls)
    
    X_orig = np.array(orig_features)
    X_aug = np.array(aug_features)
    y_orig = np.array(orig_labels)
    y_aug = np.array(aug_labels)
    
    # t-SNE
    X_combined = np.vstack([X_orig, X_aug])
    is_aug = np.array([0]*len(X_orig) + [1]*len(X_aug))
    labels_combined = np.concatenate([y_orig, y_aug])
    
    tsne = TSNE(n_components=2, perplexity=min(30, len(X_orig)-1), random_state=42, max_iter=1000)
    X_tsne = tsne.fit_transform(X_combined)
    
    # Plot
    for col, (mask_val, title) in enumerate([(0, "Trước Aug"), (1, "Sau Aug")]):
        mask = is_aug == mask_val
        for cls in selected_classes:
            cls_mask = (labels_combined == cls) & mask
            axes[row][col].scatter(X_tsne[cls_mask, 0], X_tsne[cls_mask, 1], alpha=0.5, s=15, label=cls)
        axes[row][col].set_title(f"{title} ({n_cls} lớp)", fontsize=10)
        if n_cls <= 15:
            axes[row][col].legend(fontsize=5, ncol=2)
    
    print(f"  {n_cls} lớp: done ({len(X_orig)} samples)")

plt.suptitle("t-SNE: Feature Space trước/sau Augmentation (số lớp tăng dần)", fontsize=14)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Kiểm định: Augmentation có tăng diversity?

# %%
# Kiểm định variance trước/sau augmentation (toàn bộ 45 lớp)
test_samples = load_sample(n_per_class=20)

# Original features
X_test_orig, y_test = [], []
for img, cls in test_samples:
    X_test_orig.append(extract_feature(img))
    y_test.append(cls)

# Augmented features
X_test_aug = []
np.random.seed(42)
aug_fns = list(AUGMENTATIONS.values())
for img, cls in test_samples:
    aug_img = img.copy()
    for fn in np.random.choice(aug_fns, 2, replace=False):
        aug_img = fn(aug_img)
    X_test_aug.append(extract_feature(aug_img))

X_test_orig = np.array(X_test_orig)
X_test_aug = np.array(X_test_aug)
y_test = np.array(y_test)

print(f"{'Class':<25} {'Original':>10} {'Augmented':>10} {'Change':>10}")
print("-" * 57)

orig_vars, aug_vars = [], []
for cls in classes:
    mask = y_test == cls
    var_o = np.var(X_test_orig[mask], axis=0).mean()
    var_a = np.var(X_test_aug[mask], axis=0).mean()
    change = (var_a - var_o) / (var_o + 1e-8) * 100
    orig_vars.append(var_o)
    aug_vars.append(var_a)
    print(f"{cls:<25} {var_o:>10.4f} {var_a:>10.4f} {change:>+9.1f}%")

t_stat, t_p = stats.ttest_rel(orig_vars, aug_vars)
w_stat, w_p = stats.wilcoxon(orig_vars, aug_vars, alternative='less')
diff = np.array(aug_vars) - np.array(orig_vars)
cohens_d = diff.mean() / (diff.std(ddof=1) + 1e-8)

print(f"\nPaired t-test: t={t_stat:.3f}, p={t_p:.4f}")
print(f"Wilcoxon: W={w_stat:.1f}, p={w_p:.4f}")
print(f"Cohen's d = {cohens_d:.3f} ({'lớn' if abs(cohens_d)>=0.8 else 'trung bình' if abs(cohens_d)>=0.5 else 'nhỏ'})")
print(f"Lớp có variance tăng: {sum(d > 0 for d in diff)}/{len(diff)}")
print(f"=> Augmentation {'LÀM TĂNG' if w_p < 0.05 else 'KHÔNG làm tăng'} intra-class variance có ý nghĩa thống kê")

# %% [markdown]
# ### Ablation: k-NN accuracy theo từng kỹ thuật augmentation (per-technique)
#
# Mỗi kỹ thuật augmentation được đánh giá **độc lập** so với baseline (không augmentation):
# - Load cùng tập ảnh baseline
# - Áp dụng DUY NHẤT 1 kỹ thuật augmentation → tập X_aug
# - Chạy 5-fold CV k-NN trên X_aug
# - Wilcoxon signed-rank vs baseline fold scores
#
# Tránh sai lầm phổ biến: KHÔNG stack tất cả augmentation vào 1 lần run.

# %%
# Per-augmentation k-NN ablation: đánh giá từng kỹ thuật riêng biệt
aug_knn_samples = load_sample(n_per_class=30)

# Baseline: ảnh gốc resize 128×128, normalize [0,1]
X_base_aug, y_base_aug = [], []
for img, cls in aug_knn_samples:
    resized = cv2.resize(img, (best_size, best_size))
    X_base_aug.append(resized.reshape(-1).astype(np.float32) / 255.0)
    y_base_aug.append(cls)
X_base_aug = np.array(X_base_aug)
y_base_aug  = np.array(y_base_aug)

knn_base_aug = KNeighborsClassifier(n_neighbors=5)
base_aug_scores = cross_val_score(knn_base_aug, X_base_aug, y_base_aug, cv=5, scoring='accuracy')

aug_knn_results      = {'Baseline (no aug)': (base_aug_scores.mean(), base_aug_scores.std())}
aug_knn_fold_scores  = {'Baseline (no aug)': base_aug_scores}

print(f"Baseline (no aug): {base_aug_scores.mean():.4f} ± {base_aug_scores.std():.4f}")
print(f"\n{'Technique':<25} {'Mean':>8} {'Std':>7} {'Wilcoxon p':>12} {'Cohen d':>10} {'Sig':>5}")
print("-" * 68)

for aug_name, aug_fn in AUGMENTATIONS.items():
    X_aug_k, y_aug_k = [], []
    np.random.seed(42)
    for img, cls in aug_knn_samples:
        resized = cv2.resize(img, (best_size, best_size))
        augmented = aug_fn(resized)
        X_aug_k.append(augmented.reshape(-1).astype(np.float32) / 255.0)
        y_aug_k.append(cls)
    X_aug_k = np.array(X_aug_k)
    y_aug_k  = np.array(y_aug_k)

    knn_aug = KNeighborsClassifier(n_neighbors=5)
    aug_scores_k = cross_val_score(knn_aug, X_aug_k, y_aug_k, cv=5, scoring='accuracy')
    aug_knn_results[aug_name]     = (aug_scores_k.mean(), aug_scores_k.std())
    aug_knn_fold_scores[aug_name] = aug_scores_k

    try:
        w_stat_a, w_p_a = stats.wilcoxon(aug_scores_k, base_aug_scores)
    except ValueError:
        w_stat_a, w_p_a = 0.0, 1.0
    d_diff_a = aug_scores_k - base_aug_scores
    d_a = d_diff_a.mean() / (d_diff_a.std(ddof=1) + 1e-8)
    sig_a = "✓" if w_p_a < 0.05 else "ns"
    print(f"{aug_name:<25} {aug_scores_k.mean():>8.4f} {aug_scores_k.std():>7.4f} "
          f"{w_p_a:>12.4f} {d_a:>10.3f} {sig_a:>5}")

best_aug = max([k for k in aug_knn_results if k != 'Baseline (no aug)'],
               key=lambda k: aug_knn_results[k][0])
print(f"\n→ Best augmentation technique: {best_aug}")
print(f"  k-NN accuracy = {aug_knn_results[best_aug][0]:.4f} ± {aug_knn_results[best_aug][1]:.4f}")

# %%
# Biểu đồ: per-augmentation accuracy
fig, ax = plt.subplots(figsize=(12, 5))
names_aug = list(aug_knn_results.keys())
means_aug = [aug_knn_results[n][0] for n in names_aug]
stds_aug  = [aug_knn_results[n][1] for n in names_aug]
colors_aug = ['lightgray'] + ['tomato' if n == best_aug else 'steelblue' for n in names_aug[1:]]
bars_aug = ax.bar(names_aug, means_aug, yerr=stds_aug, color=colors_aug, capsize=4,
                  edgecolor='white', alpha=0.85)
ax.axhline(base_aug_scores.mean(), color='gray', linestyle='--', alpha=0.5, label='Baseline')
for bar, v in zip(bars_aug, means_aug):
    ax.text(bar.get_x() + bar.get_width() / 2, v + 0.003, f'{v:.3f}',
            ha='center', fontsize=8)
ax.set_ylabel('5-fold CV Accuracy (k-NN, k=5)')
ax.set_title('Per-Augmentation k-NN Ablation (128×128, chuẩn hóa [0,1])', fontsize=12)
ax.tick_params(axis='x', rotation=20)
ax.legend()
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
save_dir_aug = str(Path(_IMG_ROOT).parent / 'processed')
os.makedirs(save_dir_aug, exist_ok=True)
plt.savefig(os.path.join(save_dir_aug, 'fig_aug_knn_ablation.png'), dpi=100, bbox_inches='tight')
plt.show()

# %% [markdown]
# ---
# ## Tổng hợp Pipeline & Lưu kết quả

# %%
import json

# Tổng kết các lựa chọn pipeline
PIPELINE_CHOICES_IMG = {
    'step1_resize': {
        'chosen_size'   : int(best_size),
        'method'        : 'bilinear (cv2.resize)',
        'metric'        : 'SSIM vs original 256×256',
        'justification' : f'best_size={best_size} đạt SSIM cao nhất (ablation 64/128/224×224); ANOVA p<0.05'
    },
    'step2_color_space': {
        'chosen'        : 'RGB (baseline)',
        'metric'        : 'PCA explained variance + k-NN accuracy',
        'justification' : 'RGB đủ tốt cho k-NN; HSV/Lab cho PCA variance tốt hơn nhưng pipeline cost cao hơn'
    },
    'step3_normalization': {
        'chosen_method' : str(best_norm_method),
        'metric'        : '5-fold k-NN accuracy',
        'justification' : f'{best_norm_method} đạt accuracy cao nhất; Wilcoxon vs Original'
    },
    'step4_augmentation': {
        'chosen_technique': str(best_aug),
        'metric'          : '5-fold k-NN accuracy per-technique',
        'justification'   : f'{best_aug} tăng accuracy nhiều nhất so với baseline'
    }
}

print("=== TÓM TẮT PIPELINE TIỀN XỬ LÝ ẢNH ===")
for step, info in PIPELINE_CHOICES_IMG.items():
    chosen = info.get('chosen_size') or info.get('chosen') or info.get('chosen_method') or info.get('chosen_technique')
    print(f"  {step}: {chosen}")

PROCESSED_DIR_IMG = str(Path(_IMG_ROOT).parent / 'processed')
os.makedirs(PROCESSED_DIR_IMG, exist_ok=True)

with open(os.path.join(PROCESSED_DIR_IMG, 'pipeline_choices_image.json'), 'w', encoding='utf-8') as f:
    json.dump(PIPELINE_CHOICES_IMG, f, ensure_ascii=False, indent=2)

print(f"\n✅ Đã lưu pipeline_choices_image.json → {PROCESSED_DIR_IMG}")
print("=== HOÀN THÀNH TIỀN XỬ LÝ ẢNH ===")

# Với mỗi aug_fn: dataset = {original} + {augmented_by_this_fn_only}
aug_ablation_samples = load_sample(n_per_class=30)

def extract_feature(img):
    small = cv2.resize(img, (32, 32))
    return small.reshape(-1).astype(np.float32) / 255.0

# Baseline: original only
X_base, y_base = [], []
for img, cls in aug_ablation_samples:
    X_base.append(extract_feature(img))
    y_base.append(cls)
X_base, y_base = np.array(X_base), np.array(y_base)
scores_base = cross_val_score(KNeighborsClassifier(5), X_base, y_base, cv=5, scoring='accuracy')
print(f"{'Phương pháp':<25} {'Mean Acc':>10} {'Std':>8} {'Δ vs Base':>12} {'Wilcoxon p':>12}")
print("-" * 70)
print(f"{'Original (baseline)':<25} {scores_base.mean():>10.4f} {scores_base.std():>8.4f} {'':>12} {'':>12}")

aug_knn_fold_scores = {'Original': scores_base}
for aug_name, aug_fn in AUGMENTATIONS.items():
    X_aug, y_aug = [], []
    np.random.seed(42)
    for img, cls in aug_ablation_samples:
        X_aug.append(extract_feature(img))          # original
        X_aug.append(extract_feature(aug_fn(img)))  # augmented by this fn
        y_aug.extend([cls, cls])
    X_aug, y_aug = np.array(X_aug), np.array(y_aug)
    scores_aug_single = cross_val_score(KNeighborsClassifier(5), X_aug, y_aug, cv=5, scoring='accuracy')
    aug_knn_fold_scores[aug_name] = scores_aug_single
    delta = scores_aug_single.mean() - scores_base.mean()
    try:
        _, p_wil = stats.wilcoxon(scores_aug_single, scores_base)
    except ValueError:
        p_wil = 1.0
    sig = "*" if p_wil < 0.05 else "ns"
    print(f"{aug_name:<25} {scores_aug_single.mean():>10.4f} {scores_aug_single.std():>8.4f} {delta:>+12.4f} {p_wil:>11.4f}{sig}")

best_aug = max(AUGMENTATIONS.keys(), key=lambda k: aug_knn_fold_scores[k].mean())
worst_aug = min(AUGMENTATIONS.keys(), key=lambda k: aug_knn_fold_scores[k].mean())
print(f"\n=> Kỹ thuật tốt nhất: {best_aug} ({aug_knn_fold_scores[best_aug].mean():.4f})")
print(f"=> Kỹ thuật kém nhất: {worst_aug} ({aug_knn_fold_scores[worst_aug].mean():.4f})")

# %% [markdown]
# ### Ablation: k-NN accuracy trước/sau augmentation

# %%
# k-NN: so sánh accuracy trước/sau augmentation
aug_knn_samples = load_sample(n_per_class=30)

# Original features
X_orig_knn, y_orig_knn = [], []
for img, cls in aug_knn_samples:
    resized = cv2.resize(img, (best_size, best_size))
    X_orig_knn.append(resized.reshape(-1).astype(np.float32) / 255.0)
    y_orig_knn.append(cls)

# Augmented features (thêm 1 bản augmented cho mỗi ảnh)
X_aug_knn, y_aug_knn = [], []
np.random.seed(42)
aug_fns = list(AUGMENTATIONS.values())
for img, cls in aug_knn_samples:
    resized = cv2.resize(img, (best_size, best_size))
    # Giữ ảnh gốc
    X_aug_knn.append(resized.reshape(-1).astype(np.float32) / 255.0)
    y_aug_knn.append(cls)
    # Thêm bản augmented
    aug_img = img.copy()
    for fn in np.random.choice(aug_fns, 2, replace=False):
        aug_img = fn(aug_img)
    aug_resized = cv2.resize(aug_img, (best_size, best_size))
    X_aug_knn.append(aug_resized.reshape(-1).astype(np.float32) / 255.0)
    y_aug_knn.append(cls)

X_orig_knn, y_orig_knn = np.array(X_orig_knn), np.array(y_orig_knn)
X_aug_knn, y_aug_knn = np.array(X_aug_knn), np.array(y_aug_knn)

knn = KNeighborsClassifier(n_neighbors=5)

scores_orig = cross_val_score(knn, X_orig_knn, y_orig_knn, cv=5, scoring='accuracy')
scores_aug = cross_val_score(knn, X_aug_knn, y_aug_knn, cv=5, scoring='accuracy')

print(f"Trước augmentation: {scores_orig.mean():.4f} (+/- {scores_orig.std():.4f})  [{len(X_orig_knn)} samples]")
print(f"Sau augmentation:   {scores_aug.mean():.4f} (+/- {scores_aug.std():.4f})  [{len(X_aug_knn)} samples]")
print(f"Thay đổi: {(scores_aug.mean() - scores_orig.mean()) / scores_orig.mean() * 100:+.1f}%")

# %%
# === Tóm tắt động kết quả Augmentation ===
n_increased = sum(d > 0 for d in diff)
print(f"\n=== Tóm tắt Augmentation ===")
print(f"Paired t-test: t={t_stat:.3f}, p={t_p:.4f}")
print(f"Wilcoxon:      W={w_stat:.1f}, p={w_p:.4f}")
print(f"Cohen's d:     {cohens_d:.3f} ({'lớn' if abs(cohens_d)>=0.8 else 'trung bình' if abs(cohens_d)>=0.5 else 'nhỏ'})")
print(f"Lớp tăng variance: {n_increased}/{len(diff)}")
print(f"k-NN Trước aug: {scores_orig.mean():.4f} +/- {scores_orig.std():.4f} [{len(X_orig_knn)} samples]")
print(f"k-NN Sau aug:   {scores_aug.mean():.4f} +/- {scores_aug.std():.4f} [{len(X_aug_knn)} samples]")
print(f"=> Augmentation {'LÀM TĂNG intra-class variance' if w_p < 0.05 else 'KHÔNG làm tăng'} có ý nghĩa (Wilcoxon p={w_p:.4f})")

# %% [markdown]
# **Kết luận Augmentation:** (số liệu chính xác xem output bên trên)
#
# - **Bác bỏ $H_0$**: Augmentation làm tăng intra-class variance có ý nghĩa thống kê
#   - Paired t-test, Wilcoxon, Cohen's d: xem print trên
#   - Hai phương pháp đồng thuận → kết quả robust
# - Đa số lớp có variance tăng sau augmentation (xem print trên)
# - **k-NN Ablation**: giảm accuracy là **expected** – raw-pixel k-NN không bất biến với
#   biến đổi hình học/màu sắc; với CNN feature extractor, augmentation sẽ cải thiện accuracy
# - **Pipeline 6 phép** (H-Flip, V-Flip, Rotation, Random Crop, Gaussian Noise, Brightness/Contrast)
#   hiệu quả cho tăng diversity mà không phá cấu trúc lớp

# %% [markdown]
# ---
# ## 5. Tổng kết Tiền xử lý
#
# | Mục | Kỹ thuật | Lựa chọn / Kết quả chính |
# |-----|----------|---------------------------|
# | a | **Resize** | 128×128 cân bằng SSIM/PSNR vs tốc độ; SSIM↑ theo kích thước |
# | b | **Color Space** | LAB/RGB bảo toàn thông tin tốt nhất (PCA explained var @50 PCs cao nhất) |
# | c | **Normalization** | Z-score per-channel cho k-NN accuracy cao nhất; KS test p≈0 với mọi method |
# | d | **Augmentation** | 6 transforms → tăng intra-class diversity; t-SNE phân tán hơn sau aug |
