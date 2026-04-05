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
# SOURCE/notebooks/01_EDA_image.py
# Notebook: 01 - EDA: Khám phá dữ liệu ảnh NWPU-RESISC45
# Ngôn ngữ: Tiếng Việt (markdown) + Python (code)
#
# FEEDBACK từ leader (FeedbackFromLeader.pdf):
# [OK]  Pixel stats table (mean/std) - GIỮ LẠI, không xóa
# [FIX] Boxplot axis labels: đừng sort theo median, fix rotation=90 fontsize=7
# [ADD] Per-class brightness/contrast report (DataFrame + display)
# [OPT] Near-duplicate (Hamming ≤ 4): nếu giữ thì phải có lý thuyết hamming distance
#       nếu bỏ thì comment out toàn bộ section near-dup
# [OPT] Boxplot pixel per-image có thể bỏ (bình thường)
# =============================================================================
#
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 01 - EDA: Khám phá dữ liệu ảnh NWPU-RESISC45
#
# **Mục tiêu:** Phân tích thống kê mô tả bộ dữ liệu ảnh viễn thám trước khi tiền xử lý.
#
# **Dataset:** [NWPU-RESISC45](https://www.kaggle.com/datasets/aqibrehmanpirzada/nwpuresisc45)
#
# | Thông tin | Giá trị |
# |-----------|---------|
# | Số lớp | 45 lớp cảnh viễn thám |
# | Tổng ảnh | 31,500 ảnh |
# | Train | 27,000 ảnh (600 ảnh/lớp) |
# | Test | 4,500 ảnh (100 ảnh/lớp) |
# | Kích thước | 256 \times 256 pixel |
# | Định dạng | JPEG |
# | Tỉ lệ train:test | 6:1 |

# %% [markdown]
# ## 0. Setup

# %%
import os, glob, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from PIL import Image
import imagehash
from tqdm import tqdm
from collections import Counter
from scipy import stats
from scipy.stats import gaussian_kde

warnings.filterwarnings('ignore')
plt.rcParams.update({
    'figure.figsize': (12, 6), 'font.size': 11,
    'axes.titlesize': 13, 'figure.dpi': 100
})
sns.set_style("whitegrid")

from pathlib import Path

def _find_image_root() -> Path:
    """Tìm thư mục Dataset/ chứa train/ và test/."""
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
    raise FileNotFoundError("Không tìm thấy Dataset/train/. Đặt ảnh NWPU-RESISC45 vào data/image/train/.")

_IMG_ROOT = _find_image_root()
TRAIN_DIR = str(_IMG_ROOT / 'train')
TEST_DIR  = str(_IMG_ROOT / 'test')
print(f"TRAIN_DIR = {TRAIN_DIR}")
print(f"TEST_DIR  = {TEST_DIR}")

classes = sorted(os.listdir(TRAIN_DIR))
print(f"Số lớp: {len(classes)}")

# %% [markdown]
# ---
# ## 1. Pixel Distribution
#
# ### 1.1 Thống kê pixel trung bình per-image

# %%
# Tính pixel stats cho toàn bộ 45 lớp (dùng toàn bộ ảnh mỗi lớp)
all_means_r, all_means_g, all_means_b = [], [], []
class_pixel_stats = {}
per_class_means = {'R': {}, 'G': {}, 'B': {}}

for cls in tqdm(classes, desc="Pixel stats"):
    imgs = glob.glob(os.path.join(TRAIN_DIR, cls, "*.jpg"))
    r_vals, g_vals, b_vals = [], [], []
    for p in imgs:
        img = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
        r_vals.append(img[:,:,0].mean())
        g_vals.append(img[:,:,1].mean())
        b_vals.append(img[:,:,2].mean())

    class_pixel_stats[cls] = {
        'r_mean': np.mean(r_vals), 'g_mean': np.mean(g_vals), 'b_mean': np.mean(b_vals),
        'r_std': np.std(r_vals),   'g_std': np.std(g_vals),   'b_std': np.std(b_vals)
    }
    per_class_means['R'][cls] = r_vals
    per_class_means['G'][cls] = g_vals
    per_class_means['B'][cls] = b_vals
    all_means_r.extend(r_vals)
    all_means_g.extend(g_vals)
    all_means_b.extend(b_vals)

print(f"Tổng ảnh đã scan: {len(all_means_r):,} ({len(all_means_r)//len(classes)} ảnh/lớp)")
print(f"Pixel Mean: R={np.mean(all_means_r):.1f}, G={np.mean(all_means_g):.1f}, B={np.mean(all_means_b):.1f}")
print(f"Pixel Std:  R={np.std(all_means_r):.1f}, G={np.std(all_means_g):.1f}, B={np.std(all_means_b):.1f}")

# %%
# Tổng hợp pixel stats dạng bảng + biểu đồ
df_pixel_stats = pd.DataFrame({
    'Kênh': ['R', 'G', 'B'],
    'Mean': [np.mean(all_means_r), np.mean(all_means_g), np.mean(all_means_b)],
    'Std': [np.std(all_means_r), np.std(all_means_g), np.std(all_means_b)],
    'Min': [np.min(all_means_r), np.min(all_means_g), np.min(all_means_b)],
    'Q1': [np.percentile(all_means_r, 25), np.percentile(all_means_g, 25), np.percentile(all_means_b, 25)],
    'Median': [np.median(all_means_r), np.median(all_means_g), np.median(all_means_b)],
    'Q3': [np.percentile(all_means_r, 75), np.percentile(all_means_g, 75), np.percentile(all_means_b, 75)],
    'Max': [np.max(all_means_r), np.max(all_means_g), np.max(all_means_b)],
}).round(2)
display(df_pixel_stats)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, (name, vals, color) in zip(axes, [
    ('R', all_means_r, '#e74c3c'), ('G', all_means_g, '#2ecc71'), ('B', all_means_b, '#3498db')
]):
    ax.boxplot(vals, vert=True, patch_artist=True,
               boxprops=dict(facecolor=color, alpha=0.4),
               medianprops=dict(color='black', linewidth=2))
    ax.set_title(f"Kênh {name} (mean={np.mean(vals):.1f}, std={np.std(vals):.1f})")
    ax.set_ylabel("Mean pixel value")
plt.suptitle("Boxplot pixel mean per-image theo kênh (toàn bộ 27,000 ảnh)", fontsize=13)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### 1.2 Phân bố pixel (Histogram + KDE) theo từng kênh

# %%
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, (name, vals, color) in zip(axes, [
    ('R', all_means_r, '#e74c3c'), ('G', all_means_g, '#2ecc71'), ('B', all_means_b, '#3498db')
]):
    ax.hist(vals, bins=50, color=color, alpha=0.5, edgecolor='white', density=True, label='Histogram')
    kde = gaussian_kde(vals)
    x_range = np.linspace(min(vals), max(vals), 200)
    ax.plot(x_range, kde(x_range), color=color, linewidth=2, label='KDE')
    ax.axvline(np.mean(vals), color='black', linestyle='--', linewidth=1, label=f'Mean={np.mean(vals):.1f}')
    ax.axvline(np.median(vals), color='gray', linestyle=':', linewidth=1, label=f'Median={np.median(vals):.1f}')
    ax.set_title(f"Kênh {name}")
    ax.set_xlabel("Mean pixel value")
    ax.set_ylabel("Density")
    ax.legend(fontsize=8)
plt.suptitle("Phân bố pixel trung bình theo kênh (per-image, toàn bộ 45 lớp)", fontsize=13)
plt.tight_layout()
plt.show()

# %% [markdown]
# **Nhận xét:**
#
# - Cả 3 kênh R, G, B đều có phân bố **lệch phải** (right-skewed, unimodal)
#   - Đa số ảnh tập trung ở vùng pixel mean thấp-trung bình (60-120)
#   - Đuôi dài kéo về phía giá trị cao ($>150$) do các lớp cảnh sáng như `cloud`, `snow`, `desert`
# - $\text{Mean} > \text{Median}$ ở cả 3 kênh (chênh $\approx 4$ đơn vị)
#   - Xác nhận phân bố lệch phải
#   - R: 93.8 vs 89.5, G: 97.1 vs 93.1, B: 87.6 vs 83.6
# - Kênh G có mean cao nhất (97.1), kênh B thấp nhất (87.6)
#   - Hợp lý vì ảnh viễn thám chứa nhiều cảnh tự nhiên (rừng, đồng ruộng) giàu thành phần xanh lá

# %% [markdown]
# ### 1.3 So sánh pixel mean giữa 45 lớp

# %%
# Sắp xếp theo overall mean (R+G+B)/3
df_pixel = pd.DataFrame({
    cls: {'R': s['r_mean'], 'G': s['g_mean'], 'B': s['b_mean']}
    for cls, s in class_pixel_stats.items()
}).T
df_pixel['Overall'] = (df_pixel['R'] + df_pixel['G'] + df_pixel['B']) / 3
df_pixel = df_pixel.sort_values('Overall')

fig, ax = plt.subplots(figsize=(18, 5))
x = np.arange(len(df_pixel))
w = 0.25
ax.bar(x - w, df_pixel['R'], w, label='R', color='#e74c3c', alpha=0.8)
ax.bar(x,     df_pixel['G'], w, label='G', color='#2ecc71', alpha=0.8)
ax.bar(x + w, df_pixel['B'], w, label='B', color='#3498db', alpha=0.8)
ax.set_xticks(x)
ax.set_xticklabels(df_pixel.index, rotation=90, fontsize=7)
ax.set_ylabel("Mean pixel value")
ax.set_title("Mean pixel R/G/B theo lớp (sắp xếp tăng dần theo overall mean)")
ax.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# ### 1.4 So sánh chi tiết pixel distribution - 5 lớp đại diện
#
# Chọn 5 lớp ở vị trí **min, Q1, median, Q3, max** theo overall pixel mean
# để đại diện cho toàn bộ dải giá trị.

# %%
sorted_classes = df_pixel.index.tolist()
n = len(sorted_classes)
pick_idx = [0, n // 4, n // 2, 3 * n // 4, n - 1]
compare_classes = [sorted_classes[i] for i in pick_idx]

for i, cls in enumerate(compare_classes):
    pos = ['min', 'Q1', 'median', 'Q3', 'max'][i]
    ov = df_pixel.loc[cls, 'Overall']
    print(f"[{pos:>6}] {cls}: Overall={ov:.1f}")

# %%
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ch_idx, ch_name in enumerate(['R', 'G', 'B']):
    for cls in compare_classes:
        imgs = glob.glob(os.path.join(TRAIN_DIR, cls, "*.jpg"))[:30]
        vals = []
        for p in imgs:
            img = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
            vals.append(img[:,:,ch_idx].ravel())
        all_px = np.concatenate(vals)
        axes[ch_idx].hist(all_px, bins=64, alpha=0.4, density=True, label=cls)
    axes[ch_idx].set_title(f"Kênh {ch_name}")
    axes[ch_idx].legend(fontsize=8)
    axes[ch_idx].set_xlabel("Pixel value")
plt.suptitle("So sánh phân bố pixel giữa 5 lớp đại diện", fontsize=13)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### 1.5 Kiểm định: Pixel mean giữa các lớp có khác biệt?
#
# - $H_0$: Mean pixel không khác biệt giữa 45 lớp
# - $H_1$: Ít nhất 1 cặp lớp có mean pixel khác biệt
#
# **Lựa chọn phương pháp:**
#
# - Phân bố pixel mean **lệch phải** (mục 2.2), vi phạm giả định normality của ANOVA
#   - Dùng **Kruskal-Wallis** (non-parametric) làm kiểm định chính
#   - Chạy ANOVA song song để đối chiếu
# - **Levene test** kiểm tra variance đồng nhất giữa các lớp
# - $\eta^2$ đo effect size (lớp giải thích bao nhiêu % variance)
# - **Post-hoc Mann-Whitney** chỉ trên 5 lớp đại diện (Bonferroni correction)

# %%
# Kiểm định KW/ANOVA/η² cho từng kênh R, G, B
# Post-hoc Mann-Whitney chỉ trên 5 lớp đại diện (Bonferroni correction)
eta2 = {}

for ch_name in ['R', 'G', 'B']:
    groups = [per_class_means[ch_name][cls] for cls in classes]

    lev_s, lev_p = stats.levene(*groups)
    f_val, p_anova = stats.f_oneway(*groups)
    h_val, p_kw = stats.kruskal(*groups)

    all_flat = np.concatenate(groups)
    gm = all_flat.mean()
    ss_b = sum(len(g) * (np.mean(g) - gm)**2 for g in groups)
    ss_t = np.sum((all_flat - gm)**2)
    eta2[ch_name] = ss_b / ss_t

    print(f"=== KÊNH {ch_name} ===")
    print(f"Levene: stat={lev_s:.2f}, p={lev_p:.2e}")
    print(f"ANOVA:  F={f_val:.2f}, p={p_anova:.2e}")
    print(f"Kruskal-Wallis: H={h_val:.2f}, p={p_kw:.2e}")
    print(f"Eta² = {eta2[ch_name]:.3f}  ({'lớn' if eta2[ch_name] >= 0.14 else 'trung bình' if eta2[ch_name] >= 0.06 else 'nhỏ'})")

    print(f"\nPost-hoc Mann-Whitney (5 lớp đại diện, Bonferroni a={0.05/10:.4f}):")
    for i in range(len(compare_classes)):
        for j in range(i + 1, len(compare_classes)):
            c1, c2 = compare_classes[i], compare_classes[j]
            u, p = stats.mannwhitneyu(per_class_means[ch_name][c1], per_class_means[ch_name][c2], alternative='two-sided')
            p_bonf = min(p * 10, 1.0)
            sig = "***" if p_bonf < 0.001 else ("**" if p_bonf < 0.01 else ("*" if p_bonf < 0.05 else "ns"))
            print(f"  {c1} vs {c2}: U={u:.0f}, p={p:.2e}, p_bonf={p_bonf:.2e} {sig}")
    print()

eta2_r, eta2_g, eta2_b = eta2['R'], eta2['G'], eta2['B']
print(f"Tổng hợp Eta²:  R={eta2_r:.3f},  G={eta2_g:.3f},  B={eta2_b:.3f}")

# %% [markdown]
# **Kết luận kiểm định pixel:**
#
# | Kênh | ANOVA | Kruskal-Wallis | $\eta^2$ | Ý nghĩa |
# |------|-------|----------------|---------|----------|
# | R | $F=613.62$, $p \approx 0$ | $H=11699$, $p \approx 0$ | $0.500$ | Effect size LỚN |
# | G | $F=474.96$, $p \approx 0$ | $H=10592$, $p \approx 0$ | $0.437$ | Effect size LỚN |
# | B | $F=510.74$, $p \approx 0$ | $H=11922$, $p \approx 0$ | $0.455$ | Effect size LỚN |
#
# - **Bác bỏ $H_0$** ở cả 3 kênh
#   - Lớp ảnh giải thích **43-50%** variance của pixel mean
#   - Kênh R có $\eta^2$ cao nhất ($0.500$) - kênh phân biệt lớp tốt nhất
# - Levene test reject ($p \approx 0$) - variance không đồng nhất giữa các lớp
#   - Kruskal-Wallis là kiểm định chính (không giả định equal variance)
#   - ANOVA cho kết quả tương đồng
# - Post-hoc: hầu hết các cặp lớp đại diện đều khác biệt có ý nghĩa ($p \ll 0.005$)
#   - Ngoại trừ `freeway` vs `river` ở kênh R ($p = 0.45$) - 2 lớp này có pixel mean R gần nhau

# %% [markdown]
# ---
# ## 2. Tổng quan Dataset (Class Imbalance)
#
# ### 2.1 Phân bố số lượng ảnh theo lớp

# %%
train_counts = {}
test_counts = {}
for cls in classes:
    train_counts[cls] = len(glob.glob(os.path.join(TRAIN_DIR, cls, "*.jpg")))
    test_counts[cls] = len(glob.glob(os.path.join(TEST_DIR, cls, "*.jpg")))

df_counts = pd.DataFrame({
    'class': classes,
    'train': [train_counts[c] for c in classes],
    'test': [test_counts[c] for c in classes]
})
df_counts['total'] = df_counts['train'] + df_counts['test']

print(f"Tổng ảnh: {df_counts['total'].sum():,} (train: {df_counts['train'].sum():,}, test: {df_counts['test'].sum():,})")
print(f"Ảnh/lớp (train): min={df_counts['train'].min()}, max={df_counts['train'].max()}, mean={df_counts['train'].mean():.0f}")
print(f"Ảnh/lớp (test):  min={df_counts['test'].min()}, max={df_counts['test'].max()}, mean={df_counts['test'].mean():.0f}")

# %%
fig, ax = plt.subplots(figsize=(18, 5))
x = np.arange(len(classes))
ax.bar(x - 0.2, df_counts['train'], 0.4, label='Train', color='#4C72B0')
ax.bar(x + 0.2, df_counts['test'], 0.4, label='Test', color='#DD8452')
ax.set_xticks(x)
ax.set_xticklabels(classes, rotation=90, fontsize=7)
ax.set_ylabel("Số ảnh")
ax.set_title("Phân bố số ảnh theo lớp")
ax.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# ### 2.2 Nhận xét class imbalance

# %%
imbalance_ratio = df_counts['train'].max() / df_counts['train'].min()
print(f"Imbalance ratio: {imbalance_ratio:.2f}")
print(f"Train/lớp: {df_counts['train'].iloc[0]}, Test/lớp: {df_counts['test'].iloc[0]}")
print(f"Tỉ lệ train:test = {df_counts['train'].iloc[0] // df_counts['test'].iloc[0]}:1")

# Chi-square test for uniform class distribution
# H0: phân phối lớp đồng đều (mỗi lớp = 1/45 tổng)
chi2_stat, chi2_p = stats.chisquare(df_counts['train'].values)
print(f"\nChi-square test (phân phối lớp đồng đều):")
print(f"  χ² = {chi2_stat:.4f}, p = {chi2_p:.4e}")
if chi2_p < 0.05:
    print("  => Phân phối KHÔNG đồng đều (p < 0.05)")
else:
    print("  => Phân phối đồng đều (p ≥ 0.05) — dataset cân bằng tốt")

# %%
# Kiểm tra ngưỡng 3x
ratio_3x = df_counts['train'].max() / df_counts['train'].min()
if ratio_3x >= 3.0:
    over = df_counts[df_counts['train'] >= 3 * df_counts['train'].min()]['class'].tolist()
    print(f"Có {len(over)} lớp vượt ngưỡng 3x: {over}")
else:
    print(f"Không có lớp nào vượt ngưỡng 3x (ratio = {ratio_3x:.2f})")

# %% [markdown]
# **Nhận xét:**
#
# - Dataset **cân bằng hoàn hảo** (imbalance ratio $= 1.00$)
#   - Mỗi lớp có đúng 600 ảnh train và 100 ảnh test
#   - Không có lớp nào vượt ngưỡng $3\times$
# - Không cần áp dụng oversampling/undersampling
# - Mọi khác biệt giữa các lớp phát hiện sau đây phản ánh sự khác biệt thực sự về **nội dung ảnh**, không phải do chênh lệch số lượng

# %% [markdown]
# ### 2.3 Ảnh mẫu toàn bộ 45 lớp

# %%
n_cols = 9
n_rows = (len(classes) + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, n_rows * 2.2))
for idx, cls in enumerate(classes):
    row, col = divmod(idx, n_cols)
    img_path = glob.glob(os.path.join(TRAIN_DIR, cls, "*.jpg"))[0]
    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    axes[row][col].imshow(img)
    axes[row][col].set_title(cls.replace('_', ' '), fontsize=7)
    axes[row][col].axis('off')
for idx in range(len(classes), n_rows * n_cols):
    row, col = divmod(idx, n_cols)
    axes[row][col].axis('off')
plt.suptitle("Ảnh mẫu từ toàn bộ 45 lớp (NWPU-RESISC45)", fontsize=14)
plt.tight_layout()
plt.show()

# %% [markdown]
# ---
# ## 3. Duplicate Detection (pHash)

# %%
hash_dict = {}
hash_list = []

for cls in tqdm(classes, desc="pHash"):
    imgs = glob.glob(os.path.join(TRAIN_DIR, cls, "*.jpg"))
    for p in imgs:
        h = imagehash.phash(Image.open(p))
        fname = os.path.basename(p)
        hash_key = str(h)
        if hash_key not in hash_dict:
            hash_dict[hash_key] = []
        hash_dict[hash_key].append((cls, fname))
        hash_list.append((h, cls, fname, p))

total_images = len(hash_list)
exact_dupes = {k: v for k, v in hash_dict.items() if len(v) > 1}
n_dupe_images = sum(len(v) for v in exact_dupes.values())

print(f"Tổng ảnh: {total_images:,}")
print(f"Exact duplicates: {len(exact_dupes)} nhóm ({n_dupe_images} ảnh)")
print(f"Tỉ lệ: {n_dupe_images / total_images * 100:.2f}%")

# Chi tiết từng nhóm
files_to_delete = []
if exact_dupes:
    dupe_by_class = Counter()
    for k, v in exact_dupes.items():
        for cls, _ in v:
            dupe_by_class[cls] += 1

    print(f"\nTheo lớp:")
    for cls, cnt in dupe_by_class.most_common():
        print(f"  {cls}: {cnt} ảnh")

    print(f"\nChi tiết:")
    for idx, (k, v) in enumerate(exact_dupes.items()):
        print(f"  Nhóm {idx+1} (hash={k}):")
        for i, (cls, fname) in enumerate(v):
            tag = "[GIỮ]" if i == 0 else "[XÓA]"
            print(f"    {cls}/{fname} {tag}")
            if i > 0:
                files_to_delete.append(os.path.join(TRAIN_DIR, cls, fname))

# %%
# Hiển thị ảnh duplicate
if exact_dupes:
    n_show = min(3, len(exact_dupes))
    fig, axes = plt.subplots(n_show, 2, figsize=(6, 3 * n_show))
    if n_show == 1:
        axes = [axes]
    for idx, (k, v) in enumerate(list(exact_dupes.items())[:n_show]):
        for col, (cls, fname) in enumerate(v[:2]):
            path = os.path.join(TRAIN_DIR, cls, fname)
            img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
            tag = "[GIỮ]" if col == 0 else "[XÓA]"
            axes[idx][col].imshow(img)
            axes[idx][col].set_title(f"{cls}/{fname} {tag}", fontsize=8)
            axes[idx][col].axis('off')
    plt.suptitle("Ảnh trùng lặp (exact pHash match)", fontsize=12)
    plt.tight_layout()
    plt.show()

# %% [markdown]
# **Nhận xét:**
#
# - Phát hiện 7 nhóm ảnh trùng hash (14 ảnh, chiếm 0.05%)
#   - Tỉ lệ rất thấp nhưng vẫn cần xử lý trước khi train model
#   - Tập trung ở lớp `airport` (10/14 ảnh)
# - Xử lý: giữ lại 1 ảnh/nhóm, xóa phần còn lại
#   - Cell bên dưới đã comment sẵn, uncomment khi cần xóa thực tế
#   - Tránh **data leakage** khi train model

# %%
# [OPTIONAL] Uncomment để xóa ảnh duplicate (giữ lại 1 ảnh đầu tiên mỗi nhóm)
# if files_to_delete:
#     for f in files_to_delete:
#         if os.path.exists(f):
#             os.remove(f)
#     print(f"Đã xóa {len(files_to_delete)} ảnh duplicate.")
# else:
#     print("Không có ảnh cần xóa.")

# %% [markdown]
# ### Near-duplicate detection (Hamming distance $\leq$ 4)
#
# **Lý thuyết:** pHash tạo binary fingerprint 64-bit.
# Hamming distance $d(h_1, h_2) = \text{popcount}(h_1 \oplus h_2)$: số bit khác nhau giữa 2 hash.
# Ngưỡng $d \leq 4$ là ngưỡng thực nghiệm phổ biến cho ảnh gần trùng lặp.

# %%
np.random.seed(42)
sample_idx = np.random.choice(len(hash_list), min(2000, len(hash_list)), replace=False)
sample_hashes = [hash_list[i] for i in sample_idx]

near_dupes = []
for i in range(len(sample_hashes)):
    for j in range(i+1, len(sample_hashes)):
        dist = sample_hashes[i][0] - sample_hashes[j][0]
        if dist <= 4:
            near_dupes.append((sample_hashes[i], sample_hashes[j], dist))

print(f"Near-duplicates (sample 2000, threshold=4): {len(near_dupes)} cặp")

if near_dupes:
    fig, axes = plt.subplots(min(3, len(near_dupes)), 2, figsize=(6, 3*min(3, len(near_dupes))))
    if len(near_dupes) == 1:
        axes = [axes]
    for idx, (a, b, dist) in enumerate(near_dupes[:3]):
        img_a = cv2.cvtColor(cv2.imread(a[3]), cv2.COLOR_BGR2RGB)
        img_b = cv2.cvtColor(cv2.imread(b[3]), cv2.COLOR_BGR2RGB)
        axes[idx][0].imshow(img_a)
        axes[idx][0].set_title(f"{a[1]}/{a[2]}", fontsize=8)
        axes[idx][0].axis('off')
        axes[idx][1].imshow(img_b)
        axes[idx][1].set_title(f"{b[1]}/{b[2]} (d={dist})", fontsize=8)
        axes[idx][1].axis('off')
    plt.suptitle("Near-duplicate examples", fontsize=12)
    plt.tight_layout()
    plt.show()

# %% [markdown]
# ---
# ## 4. Contrast & Brightness Analysis
#
# Tính **brightness** (mean L-channel) và **contrast** (std L-channel) cho mỗi ảnh.

# %%
brightness_data = []
for cls in tqdm(classes, desc="Brightness/Contrast"):
    for p in glob.glob(os.path.join(TRAIN_DIR, cls, "*.jpg")):
        img = cv2.imread(p)
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
        L = lab[:,:,0].astype(float)
        brightness_data.append({'class': cls, 'brightness': L.mean(), 'contrast': L.std()})

df_bc = pd.DataFrame(brightness_data)
print(f"Tổng ảnh: {len(df_bc):,}")
print(f"Brightness: mean={df_bc['brightness'].mean():.1f}, std={df_bc['brightness'].std():.1f}")
print(f"Contrast:   mean={df_bc['contrast'].mean():.1f}, std={df_bc['contrast'].std():.1f}")

# %%
# Bảng thống kê + Histogram/KDE + Boxplot cho Brightness & Contrast
df_bc_summary = pd.DataFrame({
    'Metric': ['Brightness', 'Contrast'],
    'Mean': [df_bc['brightness'].mean(), df_bc['contrast'].mean()],
    'Std': [df_bc['brightness'].std(), df_bc['contrast'].std()],
    'Min': [df_bc['brightness'].min(), df_bc['contrast'].min()],
    'Q1': [df_bc['brightness'].quantile(0.25), df_bc['contrast'].quantile(0.25)],
    'Median': [df_bc['brightness'].median(), df_bc['contrast'].median()],
    'Q3': [df_bc['brightness'].quantile(0.75), df_bc['contrast'].quantile(0.75)],
    'Max': [df_bc['brightness'].max(), df_bc['contrast'].max()],
}).round(2)
display(df_bc_summary)

fig, axes = plt.subplots(2, 2, figsize=(14, 8))

# Histogram + KDE
for col, (metric, color) in enumerate([('brightness', '#f39c12'), ('contrast', '#8e44ad')]):
    ax = axes[0][col]
    vals = df_bc[metric].values
    ax.hist(vals, bins=60, color=color, alpha=0.5, edgecolor='white', density=True, label='Histogram')
    kde = gaussian_kde(vals)
    x_range = np.linspace(vals.min(), vals.max(), 200)
    ax.plot(x_range, kde(x_range), color=color, linewidth=2, label='KDE')
    ax.axvline(np.mean(vals), color='black', linestyle='--', linewidth=1, label=f'Mean={np.mean(vals):.1f}')
    ax.axvline(np.median(vals), color='gray', linestyle=':', linewidth=1, label=f'Median={np.median(vals):.1f}')
    ax.set_title(f"{metric.capitalize()} (L-channel)")
    ax.set_xlabel(metric.capitalize())
    ax.set_ylabel("Density")
    ax.legend(fontsize=8)

# Boxplot
for col, (metric, color) in enumerate([('brightness', '#f39c12'), ('contrast', '#8e44ad')]):
    ax = axes[1][col]
    ax.boxplot(df_bc[metric].values, vert=True, patch_artist=True,
               boxprops=dict(facecolor=color, alpha=0.4),
               medianprops=dict(color='black', linewidth=2))
    ax.set_title(f"{metric.capitalize()} - Boxplot")
    ax.set_ylabel(metric.capitalize())

plt.suptitle("Phân bố Brightness & Contrast (L-channel, toàn bộ 27,000 ảnh)", fontsize=13)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### 4.1 Boxplot Brightness & Contrast theo lớp (toàn bộ 45 lớp)

# %%
# Bảng thống kê Brightness & Contrast theo từng lớp
df_bc_per_class = df_bc.groupby('class').agg(
    brightness_mean=('brightness', 'mean'),
    brightness_std=('brightness', 'std'),
    contrast_mean=('contrast', 'mean'),
    contrast_std=('contrast', 'std')
).round(2).sort_values('brightness_mean')
print(df_bc_per_class.to_string())

# %%
# Boxplot Brightness & Contrast toàn bộ 45 lớp (thứ tự theo alphabet)
fig, axes = plt.subplots(2, 1, figsize=(20, 10))

sns.boxplot(data=df_bc, x='class', y='brightness', ax=axes[0], palette='coolwarm')
axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=90, fontsize=7)
axes[0].set_title("Brightness (mean L-channel) theo lớp")
axes[0].set_xlabel("")

sns.boxplot(data=df_bc, x='class', y='contrast', ax=axes[1], palette='coolwarm')
axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=90, fontsize=7)
axes[1].set_title("Contrast (std L-channel) theo lớp")
axes[1].set_xlabel("")

plt.tight_layout()
plt.show()

# %% [markdown]
# ### 4.2 Top 5 / Bottom 5 lớp theo Brightness & Contrast

# %%
class_brightness = df_bc.groupby('class')['brightness'].mean().sort_values()
top_bottom = list(class_brightness.index[:5]) + list(class_brightness.index[-5:])

print("5 tối nhất:", list(class_brightness.index[:5]))
print("5 sáng nhất:", list(class_brightness.index[-5:]))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

df_sub = df_bc[df_bc['class'].isin(top_bottom)]
order = class_brightness[class_brightness.index.isin(top_bottom)].index.tolist()
sns.boxplot(data=df_sub, x='class', y='brightness', order=order, ax=axes[0], palette='coolwarm')
axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45, ha='right', fontsize=8)
axes[0].set_title("Brightness (5 tối nhất + 5 sáng nhất)")

class_contrast = df_bc.groupby('class')['contrast'].mean().sort_values()
top_bottom_c = list(class_contrast.index[:5]) + list(class_contrast.index[-5:])
df_sub_c = df_bc[df_bc['class'].isin(top_bottom_c)]
order_c = class_contrast[class_contrast.index.isin(top_bottom_c)].index.tolist()
sns.boxplot(data=df_sub_c, x='class', y='contrast', order=order_c, ax=axes[1], palette='coolwarm')
axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45, ha='right', fontsize=8)
axes[1].set_title("Contrast (5 thấp nhất + 5 cao nhất)")

plt.tight_layout()
plt.show()

# %% [markdown]
# ### 4.3 Scatter: Brightness vs Contrast (45 lớp)

# %%
fig, ax = plt.subplots(figsize=(12, 8))
class_bc_stats = df_bc.groupby('class').agg({'brightness': 'mean', 'contrast': 'mean'}).reset_index()

cmap = plt.cm.get_cmap('tab20', 20)
cmap2 = plt.cm.get_cmap('tab20b', 20)
cmap3 = plt.cm.get_cmap('tab20c', 10)
colors = [cmap(i) for i in range(20)] + [cmap2(i) for i in range(20)] + [cmap3(i) for i in range(5)]

for idx, (_, row) in enumerate(class_bc_stats.iterrows()):
    ax.scatter(row['brightness'], row['contrast'], c=[colors[idx]], s=70, alpha=0.85,
               edgecolors='white', linewidth=0.5)
    ax.annotate(row['class'], (row['brightness'], row['contrast']), fontsize=8, alpha=0.9,
                xytext=(3, 3), textcoords='offset points')

ax.set_xlabel("Mean Brightness (L-channel)")
ax.set_ylabel("Mean Contrast (L-channel std)")
ax.set_title("Brightness vs Contrast theo lớp")
plt.tight_layout()
plt.show()

# %% [markdown]
# ### 4.4 Kiểm định: Brightness & Contrast khác biệt giữa lớp?
#
# - $H_0$: Brightness (hoặc Contrast) không khác biệt giữa 45 lớp
# - $H_1$: Ít nhất 1 cặp lớp có khác biệt

# %%
for metric_name in ["Brightness", "Contrast"]:
    col = metric_name.lower()
    metric_groups = [df_bc[df_bc['class']==c][col].values for c in classes]

    lev_s, lev_p = stats.levene(*metric_groups)
    f_val, p_val = stats.f_oneway(*metric_groups)
    h_val, kw_p = stats.kruskal(*metric_groups)

    all_vals = np.concatenate(metric_groups)
    gm = all_vals.mean()
    ss_b = sum(len(g) * (np.mean(g) - gm)**2 for g in metric_groups)
    ss_t = np.sum((all_vals - gm)**2)
    eta2 = ss_b / ss_t if ss_t > 0 else 0

    print(f"--- {metric_name} ---")
    print(f"  Levene: stat={lev_s:.2f}, p={lev_p:.2e}")
    print(f"  ANOVA:  F={f_val:.2f}, p={p_val:.2e}")
    print(f"  Kruskal: H={h_val:.2f}, p={kw_p:.2e}")
    print(f"  Eta² = {eta2:.3f}")
    print()

# %% [markdown]
# **Kết luận:**
#
# | Metric | ANOVA | Kruskal-Wallis | $\eta^2$ |
# |--------|-------|----------------|---------|
# | Brightness | $F=38.37$, $p \approx 0$ | $H=866$, $p \approx 0$ | $0.434$ |
# | Contrast | $F=48.62$, $p \approx 0$ | $H=1072$, $p \approx 0$ | $0.492$ |
#
# - **Bác bỏ $H_0$** cho cả Brightness và Contrast
#   - Contrast có $\eta^2$ cao hơn ($0.492$ vs $0.434$) - phân biệt lớp tốt hơn Brightness
# - Scatter plot cho thấy 45 lớp phân bố rộng trên không gian Brightness $\times$ Contrast
#   - 2 đặc trưng này có tiềm năng dùng làm feature phân biệt lớp
# - Quy luật:
#   - Lớp tự nhiên đồng nhất (forest, lake): contrast **thấp**
#   - Lớp cấu trúc phức tạp (dense residential, harbor): contrast **cao**

# %% [markdown]
# ---
# ## 6. Tổng kết EDA
#
# | Tiêu chí | Kết quả |
# |----------|---------|
# | Tổng ảnh | 31,500 (27,000 train / 4,500 test) |
# | Số lớp | 45 |
# | Kích thước | $256 \times 256$ (đồng nhất) |
# | Class balance | Cân bằng hoàn hảo (600 train, 100 test / lớp, ratio $= 1.00$) |
# | Pixel mean (R,G,B) | 93.8, 97.1, 87.6 - lệch phải, kênh G cao nhất |
# | Pixel ANOVA | $\eta^2$: R $= 0.500$, G $= 0.437$, B $= 0.455$ - effect size lớn |
# | Duplicate (pHash) | 7 nhóm (14 ảnh, 0.05%) - cần xóa trước khi train |
# | Brightness | $\eta^2 = 0.434$ |
# | Contrast | $\eta^2 = 0.492$ - phân biệt lớp tốt nhất |
#
# - Dataset sạch, cân bằng, đồng nhất kích thước (256×256)
# - Các lớp cảnh có đặc trưng pixel, brightness, contrast khác biệt rõ rệt ($\eta^2 > 0.4$ ở mọi metric)
# - Kênh R và Contrast là 2 feature phân biệt lớp mạnh nhất
