# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # PHẦN 3: TIỀN XỬ LÝ DỮ LIỆU VĂN BẢN
# ## Dataset: RAGTruth (wandb/RAGTruth-processed)
#
# **Mô tả:** RAGTruth là tập dữ liệu đánh giá ảo giác (hallucination) trong hệ thống RAG (Retrieval-Augmented Generation). Tập dữ liệu chứa ~18.000 mẫu văn bản tiếng Anh với 2 nhãn:
# - **Supported**: Văn bản được hỗ trợ bởi ngữ cảnh (không chứa ảo giác)
# - **Hallucinated**: Văn bản chứa thông tin bịa đặt hoặc mâu thuẫn với ngữ cảnh
#
# **Nguồn:** https://huggingface.co/datasets/wandb/RAGTruth-processed
#
# ---

# %% [markdown]
# ## 0. Cài đặt thư viện và Import

# %%
# Cài đặt thư viện cần thiết (%pip + đủ gói Requirement §3.1)
# %pip install nltk spacy wordcloud gensim scikit-learn matplotlib seaborn tokenizers sentence-transformers statsmodels imbalanced-learn missingno pyarrow -q
# !python -m spacy download en_core_web_sm -q

# %%
import json as _json
import scipy.sparse as sp_io
from sklearn.model_selection import train_test_split
import importlib.util
from sentence_transformers import SentenceTransformer
from itertools import combinations
from scipy.sparse import csr_matrix
from scipy.stats import friedmanchisquare
from scipy.stats import wilcoxon
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer
from tokenizers.models import BPE
from tokenizers import Tokenizer
from gensim.models import Word2Vec
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.metrics import (
    classification_report, accuracy_score, f1_score,
    silhouette_score, confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.cluster import KMeans
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import statsmodels.api as sm
from scipy.stats import mannwhitneyu
from scipy import stats
import missingno as msno  # dùng để đối chiếu đủ thư viện theo Requirement §3.1
from wordcloud import WordCloud
import spacy
from nltk.stem import PorterStemmer, SnowballStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import nltk
from pathlib import Path
import os
import string
import re
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')


# NLP

# Visualization

# Stats

# ML / Vectorization

# Word2Vec

# Subword tokenizer

# KHÔNG import datasets ở đây — dataset đã tải sẵn qua download_text_dataset.py

# NLTK downloads
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('averaged_perceptron_tagger_eng', quiet=True)
nltk.download('omw-1.4', quiet=True)

# spaCy
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

# Style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 12

# Random seed
SEED = 42
np.random.seed(SEED)

# Thư mục output: Kaggle / local


def _resolve_output_dir() -> Path:
    if os.environ.get('KAGGLE_KERNEL_RUN_TYPE') is not None:
        p = Path('/kaggle/working') / 'data' / 'processed'
    else:
        try:
            p = Path(__file__).resolve().parent.parent / 'data' / 'processed'
        except NameError:
            cwd = Path.cwd()
            if (cwd / 'Source' / 'data').is_dir():
                p = cwd / 'Source' / 'data' / 'processed'
            elif (cwd.parent / 'data').is_dir():
                p = cwd.parent / 'data' / 'processed'
            elif (cwd / 'data').is_dir():
                p = cwd / 'data' / 'processed'
            else:
                p = cwd.parent / 'data' / 'processed'
    p.mkdir(parents=True, exist_ok=True)
    return p


OUTPUT_DIR = _resolve_output_dir()
print(f'OUTPUT_DIR = {OUTPUT_DIR}')
print('All libraries imported successfully!')

# %% [markdown]
# ## 1. Tải và Chuẩn bị Dữ liệu
#
# **Lưu ý:** Dataset đã được tải về local bằng `download_text_dataset.py`.
# Nếu chưa tải, chạy: `python DataMining-Lab1/download_text_dataset.py`

# %%
# Load dataset từ local parquet (không cần kết nối internet)


def _find_data_root() -> Path:
    """Tìm thư mục data/raw/text/ chứa ragtruth_full.parquet."""
    candidates = [
        # Cấu trúc chuẩn: Source/data/raw/text/
        Path.cwd().parent / 'data' / 'raw' / 'text',
        Path.cwd() / 'data' / 'raw' / 'text',
        Path.cwd().parent.parent / 'data' / 'raw' / 'text',
        # Khi cwd = workspace root (papermill chạy từ project root)
        Path.cwd() / 'Source' / 'data' / 'raw' / 'text',
        # Legacy fallback
        Path.cwd() / 'data' / 'text' / 'raw',
        Path.cwd().parent / 'data' / 'text' / 'raw',
        Path.cwd().parent.parent / 'data' / 'text' / 'raw',
        Path.cwd() / 'data' / 'raw',
        Path.cwd().parent / 'data' / 'raw',
    ]
    try:
        candidates.insert(0, Path(__file__).resolve(
        ).parent.parent / 'data' / 'raw' / 'text')
    except NameError:
        pass
    for p in candidates:
        if (p / 'ragtruth_full.parquet').exists():
            return p
    raise FileNotFoundError(
        "Không tìm thấy ragtruth_full.parquet!\n"
        "Hãy chạy: python DataMining-Lab1/download_text_dataset.py\n"
        "Sau đó đặt file vào Source/data/raw/text/"
    )


DATA_RAW = _find_data_root()
df = pd.read_parquet(DATA_RAW / 'ragtruth_full.parquet')

print(f"Tải thành công: {DATA_RAW / 'ragtruth_full.parquet'}")
print(f"Dataset shape: {df.shape}")
print(f"\nLabel distribution:")
print(df['label_name'].value_counts())
print(f"\nLabel ratio:")
print(df['label_name'].value_counts(normalize=True).round(4))
df.head(3)

# %% [markdown]
# #### Đối chiếu §2.3.1 — Quy mô dữ liệu
#
# Đề: **≥ 10.000 mẫu**, **≥ 2 nhãn**. Cell dưới lưu biến `N_SAMPLES`, `N_LABELS` và `assert`.

# %%
N_SAMPLES = len(df)
N_LABELS = df['label'].nunique()
LABEL_NAMES = df['label_name'].unique().tolist()
print("=" * 60)
print("KIỂM TRA §2.3.1")
print(f"Mẫu: {N_SAMPLES:,} (≥10.000) | Nhãn: {N_LABELS} (≥2) | {LABEL_NAMES}")
assert N_SAMPLES >= 10_000
assert N_LABELS >= 2
print("=> OK")


# %%
# Trực quan hóa phân phối nhãn
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Bar chart
label_counts = df['label_name'].value_counts()
colors = ['#2ecc71', '#e74c3c']
axes[0].bar(label_counts.index, label_counts.values,
            color=colors, edgecolor='black')
axes[0].set_title('Phân phối nhãn (Label Distribution)', fontsize=14)
axes[0].set_ylabel('Số lượng mẫu')
for i, v in enumerate(label_counts.values):
    axes[0].text(i, v + 100, str(v), ha='center',
                 fontweight='bold', fontsize=12)

# Pie chart
axes[1].pie(label_counts.values, labels=label_counts.index, autopct='%1.1f%%',
            colors=colors, startangle=90, textprops={'fontsize': 12})
axes[1].set_title('Tỉ lệ nhãn', fontsize=14)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'label_distribution.png',
            dpi=150, bbox_inches='tight')
plt.show()

# %%
# Thống kê mô tả cơ bản
print("=" * 60)
print("THỐNG KÊ MÔ TẢ CƠ BẢN")
print("=" * 60)
print(f"Tổng số mẫu: {len(df)}")
print(f"Số mẫu Supported: {(df['label'] == 0).sum()}")
print(f"Số mẫu Hallucinated: {(df['label'] == 1).sum()}")
print(
    f"Số task_type: {df['task_type'].nunique()} - {df['task_type'].unique().tolist()}")
print(f"Số model: {df['model'].nunique()} - {df['model'].unique().tolist()}")
print(f"\nThống kê độ dài text (ký tự):")
df['text_len_char'] = df['text'].str.len()
print(df['text_len_char'].describe().round(2))

# %% [markdown]
# ---
# ## 2. PHÂN TÍCH THỐNG KÊ VĂN BẢN (Text EDA) [Bắt buộc]
#
# ### 2.1. Phân phối độ dài và Mann-Whitney U test
#
# **Lý thuyết:** Phân tích phân phối độ dài văn bản giúp hiểu đặc tính dữ liệu. Mann-Whitney U test là kiểm định phi tham số so sánh hai phân phối độc lập, phù hợp khi dữ liệu không tuân theo phân phối chuẩn. Giả thuyết:
# - H0: Phân phối độ dài của hai nhóm là giống nhau
# - H1: Phân phối độ dài của hai nhóm là khác nhau

# %%
# Tính độ dài theo số từ
df['text_len_words'] = df['text'].apply(lambda x: len(x.split()))
# Tính độ dài theo số câu
df['text_len_sents'] = df['text'].apply(lambda x: len(sent_tokenize(x)))

# Tách theo nhãn
supported_lens = df[df['label'] == 0]['text_len_words']
hallucinated_lens = df[df['label'] == 1]['text_len_words']

print("Thống kê độ dài (số từ) theo nhãn:")
print(
    f"\nSupported:    mean={supported_lens.mean():.1f}, median={supported_lens.median():.1f}, std={supported_lens.std():.1f}")
print(
    f"Hallucinated: mean={hallucinated_lens.mean():.1f}, median={hallucinated_lens.median():.1f}, std={hallucinated_lens.std():.1f}")

# %%
# Trực quan hóa phân phối độ dài (số từ, số câu, số ký tự)
supported_chars_plot = df[df['label'] == 0]['text_len_char']
hallucinated_chars_plot = df[df['label'] == 1]['text_len_char']

fig, axes = plt.subplots(2, 3, figsize=(20, 12))

# [0,0] Histogram - Số từ
axes[0, 0].hist(supported_lens, bins=50, alpha=0.6,
                label='Supported', color='#2ecc71', edgecolor='black')
axes[0, 0].hist(hallucinated_lens, bins=50, alpha=0.6,
                label='Hallucinated', color='#e74c3c', edgecolor='black')
axes[0, 0].set_title('Phân phối độ dài (số từ)', fontsize=13)
axes[0, 0].set_xlabel('Số từ')
axes[0, 0].set_ylabel('Tần suất')
axes[0, 0].legend()

# [0,1] Histogram - Số ký tự
axes[0, 1].hist(supported_chars_plot, bins=50, alpha=0.6,
                label='Supported', color='#2ecc71', edgecolor='black')
axes[0, 1].hist(hallucinated_chars_plot, bins=50, alpha=0.6,
                label='Hallucinated', color='#e74c3c', edgecolor='black')
axes[0, 1].set_title('Phân phối độ dài (số ký tự)', fontsize=13)
axes[0, 1].set_xlabel('Số ký tự')
axes[0, 1].set_ylabel('Tần suất')
axes[0, 1].legend()

# [0,2] KDE - Số từ
sns.kdeplot(supported_lens,
            ax=axes[0, 2], label='Supported', color='#2ecc71', fill=True, alpha=0.3)
sns.kdeplot(hallucinated_lens,
            ax=axes[0, 2], label='Hallucinated', color='#e74c3c', fill=True, alpha=0.3)
axes[0, 2].set_title('KDE - Phân phối độ dài (số từ)', fontsize=13)
axes[0, 2].set_xlabel('Số từ')
axes[0, 2].legend()

# [1,0] Boxplot - Số từ
sns.boxplot(data=df, x='label_name', y='text_len_words',
            palette=colors, ax=axes[1, 0])
axes[1, 0].set_title('Boxplot độ dài (số từ) theo nhãn', fontsize=13)
axes[1, 0].set_xlabel('Nhãn')
axes[1, 0].set_ylabel('Số từ')

# [1,1] Violin - Số câu
sns.violinplot(data=df, x='label_name', y='text_len_sents',
               palette=colors, ax=axes[1, 1])
axes[1, 1].set_title('Violin plot độ dài (số câu) theo nhãn', fontsize=13)
axes[1, 1].set_xlabel('Nhãn')
axes[1, 1].set_ylabel('Số câu')

# [1,2] KDE - Số ký tự
sns.kdeplot(supported_chars_plot,
            ax=axes[1, 2], label='Supported', color='#2ecc71', fill=True, alpha=0.3)
sns.kdeplot(hallucinated_chars_plot,
            ax=axes[1, 2], label='Hallucinated', color='#e74c3c', fill=True, alpha=0.3)
axes[1, 2].set_title('KDE - Phân phối độ dài (số ký tự)', fontsize=13)
axes[1, 2].set_xlabel('Số ký tự')
axes[1, 2].legend()

plt.suptitle("Phân phối độ dài văn bản: số từ, số câu, số ký tự", fontsize=15)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'text_length_distribution.png',
            dpi=150, bbox_inches='tight')
plt.show()

# %%
# Mann-Whitney U Test
stat_words, p_words = mannwhitneyu(
    supported_lens, hallucinated_lens, alternative='two-sided')

supported_chars = df[df['label'] == 0]['text_len_char']
hallucinated_chars = df[df['label'] == 1]['text_len_char']
stat_chars, p_chars = mannwhitneyu(
    supported_chars, hallucinated_chars, alternative='two-sided')

supported_sents = df[df['label'] == 0]['text_len_sents']
hallucinated_sents = df[df['label'] == 1]['text_len_sents']
stat_sents, p_sents = mannwhitneyu(
    supported_sents, hallucinated_sents, alternative='two-sided')

alpha = 0.05

# Tính effect size r = Z / sqrt(N) cho Mann-Whitney
# Z xấp xỉ từ U: Z = (U - mu_U) / sigma_U  (normal approx)
n1_words = len(supported_lens)
n2_words = len(hallucinated_lens)


def mw_effect_size_r(U, n1, n2):
    """Effect size r cho Mann-Whitney U: r = Z / sqrt(N)."""
    mu_U = n1 * n2 / 2
    sig_U = np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
    Z = (U - mu_U) / sig_U
    r = abs(Z) / np.sqrt(n1 + n2)
    return Z, r


def interpret_r(r):
    if r < 0.1:
        return "rất nhỏ"
    if r < 0.3:
        return "nhỏ"
    if r < 0.5:
        return "trung bình"
    return "lớn"


z_words, r_words = mw_effect_size_r(stat_words, n1_words, n2_words)
z_chars, r_chars = mw_effect_size_r(
    stat_chars, len(supported_chars), len(hallucinated_chars))
z_sents, r_sents = mw_effect_size_r(
    stat_sents, len(supported_sents), len(hallucinated_sents))

print("=" * 80)
print("MANN-WHITNEY U TEST + EFFECT SIZE r: So sánh phân phối độ dài giữa 2 nhãn")
print("=" * 80)
print(f"\n{'Biến':<15} {'U-stat':>12} {'Z':>8} {'p-value':>12} {'r (effect)':>12} {'Diễn giải'}")
print("-" * 80)
for label, U, Z, p, r in [
    ('Số từ',   stat_words, z_words, p_words, r_words),
    ('Số ký tự', stat_chars, z_chars, p_chars, r_chars),
    ('Số câu',  stat_sents, z_sents, p_sents, r_sents),
]:
    sig = "sig" if p < alpha else "ns"
    print(f"  {label:<13} {U:>12.0f} {Z:>8.3f} {p:>12.2e} {r:>12.4f}  {interpret_r(r)} - {sig}")
print("-" * 80)
print("Effect size r: <0.1 rất nhỏ | 0.1-0.3 nhỏ | 0.3-0.5 trung bình | >0.5 lớn")

# %% [markdown]
# **Phân tích:**
# - Kết quả thực tế: r = 0.2571 (số từ), 0.2775 (số ký tự), 0.2225 (số câu) — tất cả đều **nhỏ** (0.1–0.3), dù p < 10⁻¹⁹⁵ do n = 17,790 rất lớn.
# - **Lưu ý quan trọng**: cỡ mẫu lớn khiến Mann-Whitney "luôn có ý nghĩa" dù khác biệt thực tế rất khiêm tốn. Effect size r ~ 0.25 cho thấy độ dài **không phải đặc trưng phân biệt mạnh** trong RAGTruth.
# - Kết quả nhất quán trên cả 3 chỉ số (từ, ký tự, câu): Hallucinated có độ dài ngắn hơn một chút, có thể do LLM generate câu trả lời ngắn gọn hơn khi hallucinate.
# - **Hạn chế**: r < 0.3 gợi ý độ dài chỉ nên là đặc trưng *phụ trợ*; đặc trưng ngữ nghĩa (TF-IDF, embedding) sẽ mạnh hơn đáng kể trong bài toán này.
#
# ---
#
# ### 2.2. Word Cloud, Top-50 từ phổ biến, TTR (Type-Token Ratio)
#
# **Lý thuyết:**
# - **Word Cloud**: Trực quan hóa tần suất từ, từ xuất hiện nhiều có kích thước lớn hơn.
# - **Top-50**: Phân tích từ vựng phổ biến nhất trong mỗi nhóm.
# - **TTR (Type-Token Ratio)**: Tỉ lệ giữa số từ duy nhất (types) và tổng số từ (tokens). TTR cao → từ vựng phong phú, đa dạng.

# %%
# Tokenize tất cả văn bản
all_tokens = []
supported_tokens = []
hallucinated_tokens = []

for _, row in df.iterrows():
    tokens = word_tokenize(row['text'].lower())
    tokens = [t for t in tokens if t.isalpha()]  # Chỉ giữ từ alphabet
    all_tokens.extend(tokens)
    if row['label'] == 0:
        supported_tokens.extend(tokens)
    else:
        hallucinated_tokens.extend(tokens)

print(f"Tổng số tokens (all):          {len(all_tokens):,}")
print(f"Tổng số tokens (Supported):    {len(supported_tokens):,}")
print(f"Tổng số tokens (Hallucinated): {len(hallucinated_tokens):,}")
print(f"\nSố types (unique words) (all):          {len(set(all_tokens)):,}")
print(
    f"Số types (unique words) (Supported):    {len(set(supported_tokens)):,}")
print(
    f"Số types (unique words) (Hallucinated): {len(set(hallucinated_tokens)):,}")

# %%
# Word Cloud
fig, axes = plt.subplots(1, 3, figsize=(24, 7))

# All
wc_all = WordCloud(width=800, height=400, background_color='white',
                   max_words=200, colormap='viridis').generate(' '.join(all_tokens))
axes[0].imshow(wc_all, interpolation='bilinear')
axes[0].set_title('Word Cloud - Toàn bộ dữ liệu', fontsize=14)
axes[0].axis('off')

# Supported
wc_sup = WordCloud(width=800, height=400, background_color='white',
                   max_words=200, colormap='Greens').generate(' '.join(supported_tokens))
axes[1].imshow(wc_sup, interpolation='bilinear')
axes[1].set_title('Word Cloud - Supported', fontsize=14)
axes[1].axis('off')

# Hallucinated
wc_hal = WordCloud(width=800, height=400, background_color='white',
                   max_words=200, colormap='Reds').generate(' '.join(hallucinated_tokens))
axes[2].imshow(wc_hal, interpolation='bilinear')
axes[2].set_title('Word Cloud - Hallucinated', fontsize=14)
axes[2].axis('off')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'wordclouds.png', dpi=150, bbox_inches='tight')
plt.show()

# %%
# Top-50 từ phổ biến
all_freq = Counter(all_tokens)
sup_freq = Counter(supported_tokens)
hal_freq = Counter(hallucinated_tokens)

top50_all = all_freq.most_common(50)
top50_sup = sup_freq.most_common(50)
top50_hal = hal_freq.most_common(50)

# Hiển thị top-50 dưới dạng bảng
top50_df = pd.DataFrame({
    'Rank': range(1, 51),
    'All_Word': [w for w, _ in top50_all],
    'All_Freq': [f for _, f in top50_all],
    'Supported_Word': [w for w, _ in top50_sup],
    'Supported_Freq': [f for _, f in top50_sup],
    'Hallucinated_Word': [w for w, _ in top50_hal],
    'Hallucinated_Freq': [f for _, f in top50_hal]
})
print("TOP-50 TỪ PHỔ BIẾN NHẤT:")
top50_df

# %%
# Trực quan hóa Top-20 từ phổ biến
fig, axes = plt.subplots(1, 2, figsize=(18, 8))

top20_sup = sup_freq.most_common(20)
top20_hal = hal_freq.most_common(20)

axes[0].barh([w for w, _ in top20_sup][::-1], [f for _, f in top20_sup]
             [::-1], color='#2ecc71', edgecolor='black')
axes[0].set_title('Top-20 từ phổ biến - Supported', fontsize=14)
axes[0].set_xlabel('Tần suất')

axes[1].barh([w for w, _ in top20_hal][::-1], [f for _, f in top20_hal]
             [::-1], color='#e74c3c', edgecolor='black')
axes[1].set_title('Top-20 từ phổ biến - Hallucinated', fontsize=14)
axes[1].set_xlabel('Tần suất')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'top20_words.png', dpi=150, bbox_inches='tight')
plt.show()

# %%
# TTR (Type-Token Ratio)
ttr_all = len(set(all_tokens)) / len(all_tokens)
ttr_sup = len(set(supported_tokens)) / len(supported_tokens)
ttr_hal = len(set(hallucinated_tokens)) / len(hallucinated_tokens)

# TTR theo từng mẫu
df['ttr'] = df['text'].apply(lambda x: len(
    set(x.lower().split())) / max(len(x.lower().split()), 1))

print("=" * 60)
print("TYPE-TOKEN RATIO (TTR)")
print("=" * 60)
print(f"TTR toàn bộ corpus:  {ttr_all:.4f}")
print(f"TTR Supported:       {ttr_sup:.4f}")
print(f"TTR Hallucinated:    {ttr_hal:.4f}")
print(f"\nTTR trung bình theo mẫu:")
print(
    f"  Supported:    {df[df['label'] == 0]['ttr'].mean():.4f} ± {df[df['label'] == 0]['ttr'].std():.4f}")
print(
    f"  Hallucinated: {df[df['label'] == 1]['ttr'].mean():.4f} ± {df[df['label'] == 1]['ttr'].std():.4f}")

# Mann-Whitney U test cho TTR
stat_ttr, p_ttr = mannwhitneyu(
    df[df['label'] == 0]['ttr'], df[df['label'] == 1]['ttr'], alternative='two-sided')
print(f"\nMann-Whitney U test cho TTR: U={stat_ttr:.2f}, p={p_ttr:.2e}")
print(
    f"  => TTR {'KHÁC BIỆT' if p_ttr < 0.05 else 'KHÔNG khác biệt'} có ý nghĩa thống kê giữa 2 nhóm")

# %% [markdown]
# **Phân tích:**
# - Word Cloud cho thấy tổng quan các từ phổ biến nhất. Cả hai nhóm đều chia sẻ nhiều từ chung (stop words).
# - Top-50 từ phổ biến giúp so sánh chi tiết sự khác biệt từ vựng giữa hai nhóm.
# - TTR cho thấy mức độ đa dạng từ vựng. TTR thấp hơn ở cấp corpus (do lượng token lớn) nhưng TTR theo mẫu phản ánh tốt hơn sự đa dạng.
#
# ---
#
# ### 2.3. Phân tích Định luật Zipf
#
# **Lý thuyết:** Định luật Zipf phát biểu rằng trong một corpus ngôn ngữ tự nhiên, tần suất của một từ tỉ lệ nghịch với hạng (rank) của nó. Cụ thể: f(r) ∝ 1/r^α, hay log(f) = -α·log(r) + C. Khi vẽ trên thang log-log, ta sẽ thấy đường thẳng nếu dữ liệu tuân theo định luật Zipf.

# %%
# Phân tích Zipf's Law
freq_sorted = sorted(all_freq.values(), reverse=True)
ranks = np.arange(1, len(freq_sorted) + 1)
frequencies = np.array(freq_sorted)

# Fit linear regression trên log-log
log_ranks = np.log10(ranks)
log_freqs = np.log10(frequencies)
slope, intercept, r_value, p_value, std_err = stats.linregress(
    log_ranks, log_freqs)

print("=" * 60)
print("PHÂN TÍCH ĐỊNH LUẬT ZIPF")
print("=" * 60)
print(f"Hệ số Zipf (slope):     α = {abs(slope):.4f}")
print(f"Intercept:               C = {intercept:.4f}")
print(f"R² (coefficient):        R² = {r_value**2:.4f}")
print(f"p-value:                 p = {p_value:.2e}")
print(f"\nĐịnh luật Zipf lý tưởng: α ≈ 1.0")
print(
    f"Corpus này: α = {abs(slope):.4f} => {'Tuân theo Zipf tốt' if 0.7 < abs(slope) < 1.3 else 'Lệch so với Zipf'}")
# statsmodels OLS (Requirement §3.1)
X_zipf = sm.add_constant(log_ranks)
ols_zipf = sm.OLS(log_freqs, X_zipf).fit()
print(f"statsmodels OLS α: {ols_zipf.params[1]:.4f} | scipy: {slope:.4f}")


# %%
# Vẽ đồ thị Zipf log-log
fig, axes = plt.subplots(1, 2, figsize=(18, 7))

# Log-log plot
axes[0].scatter(log_ranks, log_freqs, s=5, alpha=0.5,
                color='steelblue', label='Observed')
fitted_line = slope * log_ranks + intercept
axes[0].plot(log_ranks, fitted_line, 'r-', linewidth=2,
             label=f'Zipf fit: α={abs(slope):.3f}, R²={r_value**2:.4f}')
axes[0].set_xlabel('log₁₀(Rank)', fontsize=13)
axes[0].set_ylabel('log₁₀(Frequency)', fontsize=13)
axes[0].set_title("Định luật Zipf - Log-Log Plot", fontsize=14)
axes[0].legend(fontsize=12)

# Zipf cho từng nhóm
_zipf_slopes = {}
for label_name, freq_counter, color in [('Supported', sup_freq, '#2ecc71'), ('Hallucinated', hal_freq, '#e74c3c')]:
    freq_s = sorted(freq_counter.values(), reverse=True)
    r = np.arange(1, len(freq_s) + 1)
    s, i, rv, _, _ = stats.linregress(np.log10(r), np.log10(freq_s))
    _zipf_slopes[label_name] = s
    axes[1].scatter(np.log10(r), np.log10(freq_s), s=3, alpha=0.4, color=color)
    axes[1].plot(np.log10(r), s * np.log10(r) + i, color=color, linewidth=2,
                 label=f'{label_name}: α={abs(s):.3f}, R²={rv**2:.4f}')

axes[1].set_xlabel('log₁₀(Rank)', fontsize=13)
axes[1].set_ylabel('log₁₀(Frequency)', fontsize=13)
axes[1].set_title('Zipf theo nhóm nhãn', fontsize=14)
axes[1].legend(fontsize=12)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'zipf_law.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"Zipf \u03b1: {abs(slope):.3f} (to\u00e0n corpus) | "
      f"{abs(_zipf_slopes['Supported']):.3f} (Supported) | "
      f"{abs(_zipf_slopes['Hallucinated']):.3f} (Hallucinated)")


# %% [markdown]
# **Phân tích:**
# - Kết quả cụ thể: α = **1.631** (toàn corpus), **1.505** (Supported), **1.534** (Hallucinated); R² = 0.9357.
# - α ≈ 1.6 (xa 1.0) phản ánh corpus domain-specific (AI-generated): vocabulary tập trung hơn corpus tự nhiên, một số từ kỹ thuật lặp lại rất cao.
# - Hai nhóm nhãn có α gần nhau (1.505 vs 1.534) → phân phối từ vựng **không khác biệt nhiều về hình dạng Zipf**, nhất quán với việc cả hai đều sinh bởi cùng các LLM.
# - **Lưu ý thống kê**: R² = 0.93 trên log-log plot **KHÔNG xác nhận Zipf** — nhiều phân phối power-law khác cũng cho R² cao tương tự. Kiểm định nghiêm ngặt hơn cần KS test với power-law distribution (e.g., thư viện `powerlaw` của Alstott et al.).
#
# ---
# ## 3. CÁC KỸ THUẬT TIỀN XỬ LÝ VÀ PHÂN TÍCH TÁC ĐỘNG
#
# ### 3.1. Pipeline Chuẩn hóa Văn bản [Bắt buộc]
#
# **Lý thuyết:** Pipeline chuẩn hóa bao gồm 9 bước tuần tự: (1) lowercase, (2) xóa HTML tags,
# (3) xóa URL, (4) xóa email, (5) xóa mention (@user), (6) xóa hashtag (#topic), (7) xóa số,
# (8) xóa ký tự đặc biệt, (9) chuẩn hóa khoảng trắng.
# Với **mỗi bước**, bảng bên dưới báo cáo tỉ lệ từ vựng thay đổi và tác động đến độ dài văn bản.

# %%
# ── Tách từng bước pipeline để đo per-step vocab change (§2.3.3a yêu cầu) ──────
PIPELINE_STEPS = [
    ('lowercase', lambda t: t.lower()),
    ('remove_html', lambda t: re.sub(r'<[^>]+>', '', t)),
    ('remove_url', lambda t: re.sub(r'http\S+|www\S+|https\S+', '', t)),
    ('remove_email', lambda t: re.sub(r'\S+@\S+', '', t)),
    ('remove_mention', lambda t: re.sub(r'@\w+', '', t)),
    ('remove_hashtag', lambda t: re.sub(r'#\w+', '', t)),
    ('remove_number', lambda t: re.sub(r'\d+', '', t)),
    ('remove_punct', lambda t: re.sub(r'[^a-zA-Z\s]', '', t)),
    ('normalize_ws', lambda t: re.sub(r'\s+', ' ', t).strip()),
]


def normalize_text(text):
    """Pipeline chuẩn hóa văn bản - áp dụng tuần tự các bước."""
    for _, fn in PIPELINE_STEPS:
        text = fn(text)
    return text


# Đo per-step vocab change - §2.3.3a yêu cầu "với mỗi bước, báo cáo tỉ lệ từ vựng thay đổi"
print("=" * 80)
print("BẢNG THỐNG KÊ PER-STEP PIPELINE CHUẨN HÓA (§2.3.3a)")
print("=" * 80)

_corpus = df['text'].tolist()
_base_vocab = set(t for doc in _corpus for t in word_tokenize(doc))
_base_len = float(np.mean([len(word_tokenize(doc)) for doc in _corpus]))
_base_chars = float(pd.Series([len(doc) for doc in _corpus]).mean())

_step_rows = [{'Bước': 'Nguyên bản (raw)',
               'Vocab Size': len(_base_vocab),
               'Δ Vocab (%)': 0.0,
               'Mean Tokens/Doc': round(_base_len, 1),
               'Δ Length (%)': 0.0,
               'Mean Chars/Doc': round(_base_chars, 1)}]

_current_corpus = _corpus.copy()
_prev_vocab = len(_base_vocab)
_prev_len = _base_len
for step_name, step_fn in PIPELINE_STEPS:
    _current_corpus = [step_fn(doc) for doc in _current_corpus]
    _tok_now = [word_tokenize(doc) for doc in _current_corpus]
    _vocab_now = set(t for toks in _tok_now for t in toks)
    _len_now = float(np.mean([len(toks) for toks in _tok_now]))
    _chars_now = float(pd.Series([len(doc) for doc in _current_corpus]).mean())
    _step_rows.append({
        'Bước': step_name,
        'Vocab Size': len(_vocab_now),
        'Δ Vocab (%)': round((len(_vocab_now) - _prev_vocab) / max(_prev_vocab, 1) * 100, 2),
        'Mean Tokens/Doc': round(_len_now, 1),
        'Δ Length (%)': round((_len_now - _prev_len) / max(_prev_len, 1e-9) * 100, 2),
        'Mean Chars/Doc': round(_chars_now, 1),
    })
    _prev_vocab = len(_vocab_now)
    _prev_len = _len_now

PIPELINE_STEP_TABLE = pd.DataFrame(_step_rows)
print(PIPELINE_STEP_TABLE.to_string(index=False))
print(f"\nTổng giảm vocab: {len(_base_vocab):,} → {_prev_vocab:,} "
      f"({(_prev_vocab - len(_base_vocab)) / max(len(_base_vocab), 1)*100:.1f}%)")
print(f"Tổng giảm độ dài: {_base_len:.1f} → {_prev_len:.1f} tokens/doc "
      f"({(_prev_len - _base_len) / max(_base_len, 1e-9)*100:.1f}%)")

# Tỉ lệ giảm vocab qua toàn bộ pipeline (dùng cho ablation summary)
VOCAB_REDUCTION_RATIO = (len(_base_vocab) - _prev_vocab) / \
    max(len(_base_vocab), 1)

# Áp dụng pipeline đầy đủ cho toàn bộ dataframe
df['text_normalized'] = df['text'].apply(normalize_text)

print(f"\nVí dụ trước và sau chuẩn hóa:")
for i in range(3):
    print(f"\n--- Mẫu {i+1} ---")
    print(f"TRƯỚC: {df['text'].iloc[i][:200]}...")
    print(f"SAU:   {df['text_normalized'].iloc[i][:200]}...")

# %% [markdown]
# **Phân tích:**
# - **`lowercase`**: bước giảm vocab lớn nhất — 40,613 → 34,560 (**−14.90%**), trong khi độ dài chỉ giảm 0.49%. Gộp case variants chiếm hơn 2/3 tổng mức giảm vocab.
# - **`remove_number`**: −8.55% vocab, −0.95% độ dài — RAGTruth chứa nhiều số (năm, thống kê, ID).
# - **`remove_punct`** (bất ngờ): vocab tăng **+1.23%** (31,558 → 31,946 types) nhưng độ dài giảm −13.82% (bước giảm độ dài lớn nhất). Nguyên nhân: xóa dấu câu tách các token ghép ("end." → "end", "2023," → "2023") tạo thêm types mới, đồng thời loại bỏ nhiều punctuation tokens.
# - **`remove_html`, `remove_url`, `remove_email`, `remove_mention`, `remove_hashtag`**: tác động gần bằng 0 (<0.1%) — xác nhận RAGTruth là văn bản AI-generated sạch, không chứa noise dạng social media.
# - **Tổng kết**: −21.3% vocab và −15.1% độ dài. Hai bước `lowercase` + `remove_punct` đóng góp >95% mức giảm vocab.
#
# ---
#
# ### 3.2. Tokenization [Bắt buộc]
#
# **Lý thuyết:** So sánh 4 phương pháp tokenization:
# - **Word-level**: Tách theo khoảng trắng và dấu câu. Đơn giản nhưng OOV cao.
# - **Sentence-level**: Tách thành câu. Giữ ngữ cảnh nhưng độ phân giải thấp.
# - **Character-level**: Tách từng ký tự. Không OOV nhưng mất ngữ nghĩa, chuỗi rất dài.
# - **Subword (BPE)**: Tách thành subword units. Cân bằng giữa OOV và kích thước từ vựng.

# %%
# Lấy mẫu để phân tích (dùng text đã chuẩn hóa)
sample_texts = df['text_normalized'].tolist()

# 1. Word-level tokenization (NLTK)
word_tokenized = [word_tokenize(t) for t in sample_texts]
word_vocab = set(tok for tokens in word_tokenized for tok in tokens)
word_lengths = [len(t) for t in word_tokenized]

# 1b. Word-level tokenization (spaCy)
spacy_tokenized = [[token.text for token in doc]
                   for doc in nlp.pipe(sample_texts, batch_size=256)]
spacy_vocab = set(tok for tokens in spacy_tokenized for tok in tokens)
spacy_lengths = [len(t) for t in spacy_tokenized]

# 2. Sentence-level tokenization (NLTK)
# Dùng text gốc cho sentence
sent_tokenized = [sent_tokenize(t) for t in df['text'].tolist()]
sent_lengths = [len(t) for t in sent_tokenized]

# 3. Character-level tokenization
char_tokenized = [list(t) for t in sample_texts]
char_vocab = set(c for chars in char_tokenized for c in chars)
char_lengths = [len(t) for t in char_tokenized]

# 4. Subword (BPE) tokenization
bpe_tokenizer = Tokenizer(BPE(unk_token='[UNK]'))
bpe_trainer = BpeTrainer(vocab_size=10000, special_tokens=[
                         '[UNK]', '[PAD]', '[CLS]', '[SEP]'])
bpe_tokenizer.pre_tokenizer = Whitespace()

# Train BPE trên corpus
bpe_tokenizer.train_from_iterator(sample_texts, trainer=bpe_trainer)
bpe_tokenized = [bpe_tokenizer.encode(t).tokens for t in sample_texts]
bpe_vocab = set(tok for tokens in bpe_tokenized for tok in tokens)
bpe_lengths = [len(t) for t in bpe_tokenized]

# Tính OOV (Out-of-Vocabulary) - dùng 80% dữ liệu làm vocab, 20% đánh giá OOV
split_idx = int(len(sample_texts) * 0.8)

# Word-level OOV
train_word_vocab = set(
    tok for tokens in word_tokenized[:split_idx] for tok in tokens)
test_word_tokens = [tok for tokens in word_tokenized[split_idx:]
                    for tok in tokens]
word_oov = sum(1 for t in test_word_tokens if t not in train_word_vocab) / \
    max(len(test_word_tokens), 1)

# BPE OOV
train_bpe_vocab = set(
    tok for tokens in bpe_tokenized[:split_idx] for tok in tokens)
test_bpe_tokens = [tok for tokens in bpe_tokenized[split_idx:]
                   for tok in tokens]
bpe_oov = sum(1 for t in test_bpe_tokens if t not in train_bpe_vocab) / \
    max(len(test_bpe_tokens), 1)

# spaCy OOV
train_spacy_vocab = set(
    tok for tokens in spacy_tokenized[:split_idx] for tok in tokens)
test_spacy_tokens = [tok for tokens in spacy_tokenized[split_idx:]
                     for tok in tokens]
spacy_oov = sum(1 for t in test_spacy_tokens if t not in train_spacy_vocab) / \
    max(len(test_spacy_tokens), 1)

# Tổng hợp kết quả
tokenization_results = pd.DataFrame({
    'Method': ['Word (NLTK)', 'Word (spaCy)', 'Sentence-level', 'Character-level', 'Subword (BPE)'],
    'Vocab Size': [len(word_vocab), len(spacy_vocab), '-', len(char_vocab), len(bpe_vocab)],
    'Avg Token Length': [np.mean(word_lengths), np.mean(spacy_lengths), np.mean(sent_lengths), np.mean(char_lengths), np.mean(bpe_lengths)],
    'Median Token Length': [np.median(word_lengths), np.median(spacy_lengths), np.median(sent_lengths), np.median(char_lengths), np.median(bpe_lengths)],
    'OOV Ratio': [f'{word_oov:.4f}', f'{spacy_oov:.4f}', 'N/A', '0.0000', f'{bpe_oov:.4f}']
})

print("=" * 70)
print("SO SÁNH CÁC PHƯƠNG PHÁP TOKENIZATION")
print("=" * 70)
tokenization_results

# %%
# Trực quan hóa so sánh tokenization
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Phân phối độ dài token
methods = ['Word (NLTK)', 'Word (spaCy)', 'Sentence-level',
           'Character-level', 'Subword (BPE)']
all_lengths = [word_lengths, spacy_lengths,
               sent_lengths, char_lengths, bpe_lengths]
colors_tok = ['#3498db', '#27ae60', '#e67e22', '#9b59b6', '#1abc9c']

for i, (m, l, c) in enumerate(zip(methods, all_lengths, colors_tok)):
    axes[0].hist(l, bins=50, alpha=0.5, label=m, color=c)
axes[0].set_title('Phân phối số tokens/mẫu', fontsize=14)
axes[0].set_xlabel('Số tokens')
axes[0].set_ylabel('Tần suất')
axes[0].legend()
axes[0].set_xlim(0, np.percentile(char_lengths, 95))

# Kích thước từ vựng (bar chart)
vocab_sizes = [len(word_vocab), len(spacy_vocab),
               len(char_vocab), len(bpe_vocab)]
vocab_labels = ['Word (NLTK)', 'Word (spaCy)', 'Character-level', 'BPE']
bars = axes[1].bar(vocab_labels, vocab_sizes, color=[
                   '#3498db', '#27ae60', '#9b59b6', '#1abc9c'], edgecolor='black')
axes[1].set_title('Kích thước từ vựng', fontsize=14)
axes[1].set_ylabel('Vocab size')
for bar, v in zip(bars, vocab_sizes):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() +
                 100, f'{v:,}', ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'tokenization_comparison.png',
            dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# **Phân tích:**
# - **Word (NLTK) vs Word (spaCy)**: **gần như đồng nhất** — vocab 31,946 vs 31,914, avg length 127.37 vs 127.46 tokens/doc, OOV đều 2.64%. Cả hai đều dùng rule-based tokenization trên corpus đã chuẩn hóa, nên kết quả hội tụ.
# - **BPE** (subword, train trực tiếp trên corpus): vocab = 9,695 — nhỏ hơn **70%** so với word-level, OOV = 0.23% — **11× thấp hơn** word-level. Đây là lợi thế rõ ràng của BPE cho generalization.
# - **Character-level**: vocab = 27 (26 ký tự + space), avg length = 764.7 tokens/doc — **6× dài hơn** word-level. Zero OOV nhưng mất hoàn toàn ngữ nghĩa từ vựng.
# - **Sentence-level**: avg = 7.45 câu/doc — phù hợp tác vụ cấp câu, OOV cấp từ không định nghĩa.
# - **Lựa chọn thực tế**: Word-level được dùng cho pipeline downstream (tương thích NLTK stopwords + WordNet); BPE phù hợp cho neural models cần generalization cao.
#
# **Lựa chọn cho pipeline downstream:** BPE đạt OOV thấp nhất nhưng subword units không tương thích với
# NLTK stop word list và WordNet lemmatizer (cần whole-word tokens). Do đó **Word-level** được chọn cho các bước 3.3–3.5.

# %%
# === QUYẾT ĐỊNH 1: TOKENIZATION ===
CHOSEN_TOKENIZER = 'Word-level'
print(f"[CHỌN] Tokenizer: {CHOSEN_TOKENIZER}")
print(
    f"  Word-level OOV={word_oov:.4f} → được kiểm soát bởi min_df/max_features trong TF-IDF")
print(
    f"  BPE OOV={bpe_oov:.4f} thấp hơn nhưng subwords không tương thích với stop word list + lemmatizer")
print(f"  → Bước 3.3 và 3.4 sẽ dùng df['tokens'] (word-level)")

# %% [markdown]
# ---
#
# ### 3.3. Stop Words [Bắt buộc]
#
# **Lý thuyết:** Stop words là các từ xuất hiện rất thường xuyên nhưng mang ít ý nghĩa ngữ nghĩa (the, is, at, which...). Việc loại bỏ stop words giúp:
# - Giảm kích thước từ vựng và chiều của ma trận đặc trưng
# - Tập trung vào các từ mang ý nghĩa phân biệt
#
# Tuy nhiên, trong một số trường hợp, stop words có thể mang thông tin hữu ích (phong cách viết, ngữ pháp).

# %%
# Phân tích Stop Words
stop_words = set(stopwords.words('english'))
print(f"Số lượng stop words trong NLTK English: {len(stop_words)}")
print(f"Một số stop words: {list(stop_words)[:20]}")

# Tokenize và tách stop words vs non-stop words
df['tokens'] = df['text_normalized'].apply(lambda x: word_tokenize(x))
df['tokens_no_stop'] = df['tokens'].apply(
    lambda x: [t for t in x if t not in stop_words])

# So sánh kích thước từ vựng
vocab_with_stop = set(tok for tokens in df['tokens'] for tok in tokens)
vocab_no_stop = set(tok for tokens in df['tokens_no_stop'] for tok in tokens)

print(f"\n{'='*60}")
print("SO SÁNH TRƯỚC VÀ SAU KHI XÓA STOP WORDS")
print(f"{'='*60}")
print(f"Kích thước từ vựng CÓ stop words:    {len(vocab_with_stop):,}")
print(f"Kích thước từ vựng KHÔNG stop words:  {len(vocab_no_stop):,}")
print(
    f"Tỉ lệ giảm từ vựng: {(1 - len(vocab_no_stop) / len(vocab_with_stop)) * 100:.2f}%")

total_with = sum(len(t) for t in df['tokens'])
total_without = sum(len(t) for t in df['tokens_no_stop'])
print(f"\nTổng tokens CÓ stop words:    {total_with:,}")
print(f"Tổng tokens KHÔNG stop words:  {total_without:,}")
print(f"Tỉ lệ tokens bị loại: {(1 - total_without / total_with) * 100:.2f}%")

# %%
# Mutual Information: đo mức độ thông tin của từng từ với nhãn
# Tạo BoW cho cả 2 trường hợp
cv_with_stop = CountVectorizer(max_features=5000)
X_with_stop = cv_with_stop.fit_transform(df['text_normalized'])

cv_no_stop = CountVectorizer(max_features=5000, stop_words='english')
X_no_stop = cv_no_stop.fit_transform(df['text_normalized'])

y = df['label'].values

# Tính MI cho từng feature
mi_with = mutual_info_classif(
    X_with_stop, y, discrete_features=True, random_state=SEED)
mi_no = mutual_info_classif(
    X_no_stop, y, discrete_features=True, random_state=SEED)

MI_MEAN_WITH_STOP = float(np.mean(mi_with))
MI_MEAN_NO_STOP = float(np.mean(mi_no))
MI_MEAN_DELTA = MI_MEAN_NO_STOP - MI_MEAN_WITH_STOP

# Top-20 từ có MI cao nhất
mi_with_df = pd.DataFrame(
    {'word': cv_with_stop.get_feature_names_out(), 'MI': mi_with})
mi_with_df = mi_with_df.sort_values('MI', ascending=False).head(20)

mi_no_df = pd.DataFrame(
    {'word': cv_no_stop.get_feature_names_out(), 'MI': mi_no})
mi_no_df = mi_no_df.sort_values('MI', ascending=False).head(20)

print("Top-20 từ có Mutual Information cao nhất:")
print("\nCÓ stop words:")
for _, row in mi_with_df.iterrows():
    is_stop = '(STOP)' if row['word'] in stop_words else ''
    print(f"  {row['word']:20s} MI={row['MI']:.6f} {is_stop}")

print("\nKHÔNG stop words:")
for _, row in mi_no_df.iterrows():
    print(f"  {row['word']:20s} MI={row['MI']:.6f}")

print("\n" + "=" * 60)
print("MI TRUNG BÌNH TRƯỚC/SAU XÓA STOP WORDS")
print("=" * 60)
print(f"MI mean (có stop):    {MI_MEAN_WITH_STOP:.6f}")
print(f"MI mean (không stop): {MI_MEAN_NO_STOP:.6f}")
print(f"ΔMI mean: {'+' if MI_MEAN_DELTA > 0 else ''}{MI_MEAN_DELTA:.6f}")

# %%
# So sánh hiệu năng Naive Bayes trước và sau khi xóa stop words
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

# Với stop words
nb_with = MultinomialNB()
scores_with = cross_val_score(
    nb_with, X_with_stop, y, cv=skf, scoring='f1_macro')

# Không stop words
nb_no = MultinomialNB()
scores_no = cross_val_score(nb_no, X_no_stop, y, cv=skf, scoring='f1_macro')

print("=" * 60)
print("HIỆU NĂNG NAIVE BAYES: TRƯỚC VÀ SAU KHI XÓA STOP WORDS")
print("=" * 60)
print(
    f"CÓ stop words:    F1-macro = {scores_with.mean():.4f} ± {scores_with.std():.4f}")
print(
    f"KHÔNG stop words:  F1-macro = {scores_no.mean():.4f} ± {scores_no.std():.4f}")
diff = scores_no.mean() - scores_with.mean()
print(f"\nChênh lệch: {'+' if diff > 0 else ''}{diff:.4f}")

# Wilcoxon signed-rank test: so sánh cặp fold (có stop vs không stop)
# Phù hợp hơn paired t-test vì không giả định phân phối chuẩn

try:
    w_stat, w_p = wilcoxon(scores_no, scores_with, alternative='two-sided')
    print(
        f"\nWilcoxon signed-rank test (paired folds): W={w_stat:.2f}, p={w_p:.4f}")
except ValueError as e:
    print(f"\nWilcoxon: {e} (cần ít nhất 2 folds khác nhau)")

# Cohen's d effect size (standardized mean difference)
pooled_std = np.sqrt((scores_no.std()**2 + scores_with.std()**2) / 2)
cohens_d = (scores_no.mean() - scores_with.mean()) / (pooled_std + 1e-9)
print(
    f"Cohen's d = {cohens_d:.4f}")

# %% [markdown]
# **Phân tích:**
# - Xóa stop words giảm **41.09%** token nhưng chỉ giảm **0.41%** vocab — xác nhận stop words là một số lượng nhỏ từ tần suất cao, không đóng góp nhiều vào kích thước vocabulary.
# - **ΔMI mean = −0.000114** (MI giảm sau khi xóa stop words): kết quả ngược trực giác! Stop words mang một chút thông tin phân biệt trung bình trong không gian 5,000 features. Điều này xảy ra khi stop words phân phối không đều giữa hai lớp (vd. văn bản Hallucinated dùng nhiều "the", "a" do cấu trúc câu khác).
# - **NB F1**: chênh lệch rất nhỏ (< 0.0001) — không có ý nghĩa thực tế. Wilcoxon test sẽ cho p không đáng kể với 5 folds.
# - **Kết luận**: Quyết định giữ/bỏ stop words không ảnh hưởng đáng kể; nên quyết định dựa trên downstream task và mô hình, không phải chỉ dựa trên raw vocab count.

# %%
# === QUYẾT ĐỊNH 2: STOP WORDS ===
CHOSEN_STOPWORDS = 'remove' if scores_no.mean() >= scores_with.mean() else 'keep'
print(f"[CHỌN] Stop words: {CHOSEN_STOPWORDS}")
print(
    f"  F1 (remove stop)={scores_no.mean():.4f}, F1 (keep stop)={scores_with.mean():.4f}")
print(f"  ΔMI mean={MI_MEAN_DELTA:+.6f} → bỏ stop words {'tăng' if MI_MEAN_DELTA > 0 else 'không tăng'} thông tin phân biệt trung bình")
print(f"  → Bước 3.4 dùng: df['tokens_no_stop']")

# %% [markdown]
# ---
#
# ### 3.4. Stemming và Lemmatization [Bắt buộc]
#
# **Lý thuyết:**
# - **Stemming**: Cắt hậu tố từ theo quy tắc (rule-based). Nhanh nhưng có thể tạo từ không hợp lệ.
#   - Porter Stemmer: Thuật toán cổ điển, 5 bước cắt hậu tố.
#   - Snowball Stemmer: Phiên bản cải tiến của Porter, hỗ trợ nhiều ngôn ngữ.
# - **Lemmatization**: Đưa từ về dạng gốc (lemma) dựa trên từ điển. Chậm hơn nhưng chính xác hơn.
#   - WordNet Lemmatizer: Sử dụng WordNet database.
# - **Collision rate**: Tỉ lệ các từ khác nhau bị map về cùng một dạng gốc. Collision cao = nguy cơ mất phân biệt ngữ nghĩa.

# %%
# Khởi tạo stemmers và lemmatizer
porter = PorterStemmer()
snowball = SnowballStemmer('english')
lemmatizer = WordNetLemmatizer()

# Lấy sample tokens (đã loại stop words)
all_unique_words = list(vocab_no_stop)

# Áp dụng từng phương pháp
porter_results = {w: porter.stem(w) for w in all_unique_words}
snowball_results = {w: snowball.stem(w) for w in all_unique_words}
wordnet_results = {w: lemmatizer.lemmatize(w) for w in all_unique_words}

# Tính collision rate: số từ gốc bị trùng / tổng số từ


def calc_collision_rate(mapping):
    """Tính collision rate: 1 - (unique_results / unique_inputs)"""
    return 1 - len(set(mapping.values())) / len(set(mapping.keys()))


porter_collision = calc_collision_rate(porter_results)
snowball_collision = calc_collision_rate(snowball_results)
wordnet_collision = calc_collision_rate(wordnet_results)

print("=" * 60)
print("SO SÁNH STEMMING VÀ LEMMATIZATION")
print("=" * 60)
print(f"Tổng từ unique (input):       {len(all_unique_words):,}")
print(f"\nPorter Stemmer:")
print(f"  Unique stems:     {len(set(porter_results.values())):,}")
print(
    f"  Collision rate:   {porter_collision:.4f} ({porter_collision*100:.2f}%)")
print(f"\nSnowball Stemmer:")
print(f"  Unique stems:     {len(set(snowball_results.values())):,}")
print(
    f"  Collision rate:   {snowball_collision:.4f} ({snowball_collision*100:.2f}%)")
print(f"\nWordNet Lemmatizer:")
print(f"  Unique lemmas:    {len(set(wordnet_results.values())):,}")
print(
    f"  Collision rate:   {wordnet_collision:.4f} ({wordnet_collision*100:.2f}%)")

# %%
# Ví dụ so sánh
example_words = ['running', 'studies', 'better', 'flies', 'organizing', 'happiness',
                 'communication', 'university', 'generated', 'information',
                 'summarizes', 'approximately', 'additionally', 'categories', 'analysis']

comparison_df = pd.DataFrame({
    'Original': example_words,
    'Porter': [porter.stem(w) for w in example_words],
    'Snowball': [snowball.stem(w) for w in example_words],
    'WordNet': [lemmatizer.lemmatize(w) for w in example_words]
})
print("Ví dụ so sánh các phương pháp:")
comparison_df


# %%
# Áp dụng từng phương pháp lên corpus và đánh giá hiệu năng Logistic Regression
def apply_stemlem(tokens_series, method):
    """Áp dụng stemming/lemmatization và join lại thành text."""
    if method == 'porter':
        return tokens_series.apply(lambda toks: ' '.join([porter.stem(t) for t in toks]))
    elif method == 'snowball':
        return tokens_series.apply(lambda toks: ' '.join([snowball.stem(t) for t in toks]))
    elif method == 'wordnet':
        return tokens_series.apply(lambda toks: ' '.join([lemmatizer.lemmatize(t) for t in toks]))
    else:  # none
        return tokens_series.apply(lambda toks: ' '.join(toks))


# Tạo text cho từng phương pháp
texts_none = apply_stemlem(df['tokens_no_stop'], 'none')
texts_porter = apply_stemlem(df['tokens_no_stop'], 'porter')
texts_snowball = apply_stemlem(df['tokens_no_stop'], 'snowball')
texts_wordnet = apply_stemlem(df['tokens_no_stop'], 'wordnet')

# TF-IDF + Logistic Regression cho từng phương pháp — 1 vòng lặp duy nhất
# lưu cả mean/std (results_stemlem) và fold scores (fold_scores_stemlem) cho Friedman test
results_stemlem = {}
fold_scores_stemlem = {}
for name, texts in [('None (baseline)', texts_none), ('Porter', texts_porter),
                    ('Snowball', texts_snowball), ('WordNet', texts_wordnet)]:
    tfidf = TfidfVectorizer(max_features=10000)
    X = tfidf.fit_transform(texts)
    lr = LogisticRegression(max_iter=1000, random_state=SEED, C=1.0)
    scores = cross_val_score(lr, X, y, cv=skf, scoring='f1_macro')
    results_stemlem[name] = {'mean': scores.mean(), 'std': scores.std(
    ), 'vocab': len(tfidf.get_feature_names_out())}
    fold_scores_stemlem[name] = scores
    print(f"{name:20s} | F1-macro = {scores.mean():.4f} ± {scores.std():.4f} | Vocab = {len(tfidf.get_feature_names_out()):,}")

print("\nKết luận: Phương pháp tốt nhất:", max(
    results_stemlem, key=lambda k: results_stemlem[k]['mean']))

# === QUYẾT ĐỊNH 3: STEMMING / LEMMATIZATION ===
BEST_STEMLEM = max(results_stemlem, key=lambda k: results_stemlem[k]['mean'])
print(
    f"[CHỌN] Stemming/Lemmatization: {BEST_STEMLEM} (F1={results_stemlem[BEST_STEMLEM]['mean']:.4f})")
print(f"  → Bước 3.5 sẽ dùng '{BEST_STEMLEM}' để tạo texts_final")

# ===========================================================================
# Friedman test: so sánh đồng thời 4 phương pháp (dữ liệu lặp = 5 folds)
# Friedman là non-parametric equivalent của repeated-measures ANOVA
# Phù hợp khi so sánh k ≥ 3 phương pháp với cùng tập dữ liệu fold
# ===========================================================================

# fold_scores_stemlem đã được tạo trong vòng lặp ở trên (mỗi entry = array 5 folds)
friedman_stat, friedman_p = friedmanchisquare(*fold_scores_stemlem.values())
print(
    f"\nFriedman test (4 stemming/lemma methods, 5 folds): χ²={friedman_stat:.4f}, p={friedman_p:.4f}")
if friedman_p < 0.05:
    print("  post-hoc pairwise Wilcoxon:")
    names_sl = list(fold_scores_stemlem.keys())
    from itertools import combinations
    from scipy.stats import wilcoxon
    n_pairs = len(list(combinations(names_sl, 2)))
    alpha_bonf = 0.05 / n_pairs  # Bonferroni correction
    print(f"  Bonferroni-corrected α = {alpha_bonf:.4f} (n_pairs={n_pairs})")
    for n1, n2 in combinations(names_sl, 2):
        try:
            _, pw = wilcoxon(fold_scores_stemlem[n1], fold_scores_stemlem[n2])
            sig = "***" if pw < alpha_bonf else ("*" if pw < 0.05 else "ns")
            print(f"    {n1:20s} vs {n2:20s}: p={pw:.4f} {sig}")
        except ValueError:
            print(f"    {n1:20s} vs {n2:20s}: identical scores, skip")
else:
    print("  Không có phương pháp nào khác biệt đáng kể (p ≥ 0.05)")

# %%
# Trực quan hóa so sánh
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Collision rate
methods_sl = ['Porter', 'Snowball', 'WordNet']
collisions = [porter_collision, snowball_collision, wordnet_collision]
colors_sl = ['#3498db', '#e67e22', '#2ecc71']
bars = axes[0].bar(methods_sl, collisions, color=colors_sl, edgecolor='black')
axes[0].set_title('Collision Rate', fontsize=14)
axes[0].set_ylabel('Collision Rate')
for bar, v in zip(bars, collisions):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() +
                 0.005, f'{v:.4f}', ha='center', fontweight='bold')

# F1-macro
methods_f1 = list(results_stemlem.keys())
f1_means = [results_stemlem[m]['mean'] for m in methods_f1]
f1_stds = [results_stemlem[m]['std'] for m in methods_f1]
bars = axes[1].bar(methods_f1, f1_means, yerr=f1_stds, color=[
                   'gray'] + colors_sl, edgecolor='black', capsize=5)
axes[1].set_title('Hiệu năng Logistic Regression (F1-macro)', fontsize=14)
axes[1].set_ylabel('F1-macro')
for bar, v in zip(bars, f1_means):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() +
                 0.005, f'{v:.4f}', ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'stemming_lemmatization.png',
            dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# **Phân tích:**
# - Collision rate thực tế: Porter **35.43%**, Snowball **35.68%**, WordNet **13.20%** — WordNet bảo toàn ngữ nghĩa tốt hơn ~3×.
# - F1-macro: None=0.7287, Porter=0.7274, Snowball=0.7284, WordNet=0.7285 — **tất cả cách nhau < 0.002**, không có phương pháp nào vượt trội rõ ràng.
# - Stemming mạnh (Porter/Snowball) tuy giảm 35% từ vựng nhưng không cải thiện F1 - thậm chí giảm nhẹ. Trên RAGTruth (domain kỹ thuật), các hậu tố mang ý nghĩa ngữ nghĩa quan trọng (e.g. "summarize" vs "summarization").
# - **Hạn chế thống kê**: Friedman test với chỉ 5 folds có **statistical power thấp** — không thể kết luận chắc chắn về sự khác biệt giữa các phương pháp. Cần ≥20 folds hoặc bootstrap để estimate đáng tin cậy hơn.
#
# ---
#
# ### 3.5. Vector hóa: BoW, TF-IDF (n-gram), Word2Vec [Bắt buộc]
#
# **Lý thuyết:**
# - **Bag of Words (BoW)**: Biểu diễn văn bản bằng tần suất từ. Đơn giản nhưng mất ngữ cảnh.
# - **TF-IDF**: Cân nhắc tầm quan trọng của từ dựa trên tần suất trong tài liệu (TF) và tần suất nghịch trong corpus (IDF). Giúp giảm trọng số của stop words tự nhiên.
# - **Word2Vec**: Biểu diễn từ thành vector dense trong không gian liên tục. Nắm bắt được quan hệ ngữ nghĩa.
#
# **Ghi chú triển khai:** Phần BoW/TF-IDF phía dưới được cài đặt thủ công (xây vocab + ma trận sparse + IDF + chuẩn hóa), không dùng vectorizer có sẵn của scikit-learn.

# %%
# Chuẩn bị text — áp dụng BEST_STEMLEM đã chọn ở bước 3.4
if BEST_STEMLEM == 'Porter':
    df['text_processed'] = df['tokens_no_stop'].apply(
        lambda toks: ' '.join([porter.stem(t) for t in toks]))
elif BEST_STEMLEM == 'Snowball':
    df['text_processed'] = df['tokens_no_stop'].apply(
        lambda toks: ' '.join([snowball.stem(t) for t in toks]))
elif BEST_STEMLEM == 'WordNet':
    df['text_processed'] = df['tokens_no_stop'].apply(
        lambda toks: ' '.join([lemmatizer.lemmatize(t) for t in toks]))
else:  # None (baseline)
    df['text_processed'] = df['tokens_no_stop'].apply(
        lambda toks: ' '.join(toks))
print(f"[ÁP DỤNG] {BEST_STEMLEM} → df['text_processed']")

texts_final = df['text_processed'].tolist()
y = df['label'].values

print(f"Số mẫu: {len(texts_final)}")
print(f"Ví dụ text processed: {texts_final[0][:200]}...")

# ------------------------------------------------------------
# Ablation theo bước tiền xử lý: vocab change + tác động phân phối độ dài
# ------------------------------------------------------------
raw_tokens_series = df['text'].apply(
    lambda x: [t.lower() for t in word_tokenize(x) if t.isalpha()])
norm_tokens_series = df['text_normalized'].apply(word_tokenize)
nostop_tokens_series = df['tokens_no_stop']
lemma_tokens_series = nostop_tokens_series.apply(
    lambda toks: [lemmatizer.lemmatize(t) for t in toks])

stage_tokens = {
    'Raw': raw_tokens_series,
    'Normalized': norm_tokens_series,
    'NoStop': nostop_tokens_series,
    'Lemmatized': lemma_tokens_series,
}

stage_vocab = {k: len(set(tok for doc in v for tok in doc))
               for k, v in stage_tokens.items()}
stage_len_mean = {k: float(v.apply(len).mean())
                  for k, v in stage_tokens.items()}
stage_len_median = {k: float(v.apply(len).median())
                    for k, v in stage_tokens.items()}

base_vocab = stage_vocab['Raw']
base_len = stage_len_mean['Raw']

rows_stage = []
for k in ['Raw', 'Normalized', 'NoStop', 'Lemmatized']:
    rows_stage.append({
        'Stage': k,
        'Vocab Size': stage_vocab[k],
        'Vocab Change vs Raw (%)': (stage_vocab[k] - base_vocab) / max(base_vocab, 1) * 100,
        'Mean Len (tokens/doc)': stage_len_mean[k],
        'Median Len (tokens/doc)': stage_len_median[k],
        'Len Change vs Raw (%)': (stage_len_mean[k] - base_len) / max(base_len, 1e-12) * 100,
    })

PREPROCESS_LENGTH_ABLATION = pd.DataFrame(rows_stage)
print("\n" + "=" * 80)
print("ABLATION PHÂN PHỐI ĐỘ DÀI QUA TỪNG BƯỚC TIỀN XỬ LÝ")
print("=" * 80)
print(PREPROCESS_LENGTH_ABLATION.to_string(
    index=False, float_format=lambda x: f"{x:.2f}"))

# Trực quan hóa tác động đến phân phối độ dài
len_plot_df = pd.DataFrame({
    'Raw': raw_tokens_series.apply(len),
    'Normalized': norm_tokens_series.apply(len),
    'NoStop': nostop_tokens_series.apply(len),
    'Lemmatized': lemma_tokens_series.apply(len),
})

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Boxplot
sns.boxplot(data=len_plot_df, ax=axes[0], palette='Set2')
axes[0].set_title('Độ dài văn bản theo từng bước tiền xử lý', fontsize=13)
axes[0].set_ylabel('Số token / văn bản')
axes[0].tick_params(axis='x', rotation=20)

# KDE
for col, color in zip(len_plot_df.columns, ['#34495e', '#2980b9', '#e67e22', '#27ae60']):
    sns.kdeplot(len_plot_df[col], ax=axes[1], label=col,
                fill=True, alpha=0.2, color=color)
axes[1].set_title('KDE độ dài token qua các bước', fontsize=13)
axes[1].set_xlabel('Số token / văn bản')
axes[1].legend()

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'length_distribution_ablation.png',
            dpi=150, bbox_inches='tight')
plt.show()

# %%
# TỰ CÀI ĐẶT BoW + TF-IDF n-gram (không dùng CountVectorizer/TfidfVectorizer)


def _extract_ngrams(tokens, n):
    if n <= 0 or len(tokens) < n:
        return []
    return [' '.join(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]


def _build_vocab(token_lists, ngram_range=(1, 1), max_features=10000):
    freq = Counter()
    min_n, max_n = ngram_range
    for toks in token_lists:
        for n in range(min_n, max_n + 1):
            freq.update(_extract_ngrams(toks, n))
    vocab_terms = [t for t, _ in freq.most_common(max_features)]
    return {term: idx for idx, term in enumerate(vocab_terms)}


def _bow_transform(token_lists, vocab, ngram_range=(1, 1)):
    rows, cols, data = [], [], []
    min_n, max_n = ngram_range

    for r, toks in enumerate(token_lists):
        cnt = Counter()
        for n in range(min_n, max_n + 1):
            for ng in _extract_ngrams(toks, n):
                if ng in vocab:
                    cnt[vocab[ng]] += 1
        for c, v in cnt.items():
            rows.append(r)
            cols.append(c)
            data.append(v)

    return csr_matrix((data, (rows, cols)), shape=(len(token_lists), len(vocab)), dtype=np.float64)


def _tfidf_from_bow(X_bow_local):
    # idf = log((1 + N)/(1 + df)) + 1
    N = X_bow_local.shape[0]
    df_vec = np.asarray((X_bow_local > 0).sum(axis=0)).ravel()
    idf = np.log((1 + N) / (1 + df_vec)) + 1.0

    X_tfidf_local = X_bow_local.copy().astype(np.float64).tocsr()
    # nhân IDF cho từng phần tử non-zero
    X_tfidf_local.data *= idf[X_tfidf_local.indices]

    # chuẩn hóa L2 theo từng document
    row_sq_sum = np.asarray(X_tfidf_local.power(2).sum(axis=1)).ravel()
    row_norm = np.sqrt(np.maximum(row_sq_sum, 1e-12))
    X_tfidf_local = X_tfidf_local.multiply((1.0 / row_norm)[:, None]).tocsr()

    return X_tfidf_local, idf


# Chuẩn bị token list cho custom vectorizer
custom_token_lists = [txt.split() for txt in texts_final]

# 1) BoW unigram
vocab_bow = _build_vocab(
    custom_token_lists, ngram_range=(1, 1), max_features=10000)
X_bow = _bow_transform(custom_token_lists, vocab_bow, ngram_range=(1, 1))

# 2) TF-IDF unigram
vocab_uni = _build_vocab(
    custom_token_lists, ngram_range=(1, 1), max_features=10000)
X_bow_uni = _bow_transform(custom_token_lists, vocab_uni, ngram_range=(1, 1))
X_tfidf_uni, idf_uni = _tfidf_from_bow(X_bow_uni)

# 3) TF-IDF (1,2)-gram
vocab_bi = _build_vocab(
    custom_token_lists, ngram_range=(1, 2), max_features=10000)
X_bow_bi = _bow_transform(custom_token_lists, vocab_bi, ngram_range=(1, 2))
X_tfidf_bi, idf_bi = _tfidf_from_bow(X_bow_bi)

# 4) TF-IDF (1,2,3)-gram
vocab_tri = _build_vocab(
    custom_token_lists, ngram_range=(1, 3), max_features=10000)
X_bow_tri = _bow_transform(custom_token_lists, vocab_tri, ngram_range=(1, 3))
X_tfidf_tri, idf_tri = _tfidf_from_bow(X_bow_tri)


# Sparsity ratio
def sparsity_ratio(X):
    return 1 - X.nnz / (X.shape[0] * X.shape[1])


print("=" * 70)
print("THỐNG KÊ CÁC PHƯƠNG PHÁP VECTOR HÓA (CUSTOM IMPLEMENTATION)")
print("=" * 70)
for name, X in [('BoW', X_bow), ('TF-IDF (1-gram)', X_tfidf_uni),
                ('TF-IDF (1,2-gram)', X_tfidf_bi), ('TF-IDF (1,2,3-gram)', X_tfidf_tri)]:
    print(f"\n{name}:")
    print(f"  Shape:          {X.shape}")
    print(f"  Non-zero:       {X.nnz:,}")
    print(
        f"  Sparsity ratio: {sparsity_ratio(X):.6f} ({sparsity_ratio(X)*100:.2f}%)")

# %%
# 5. Word2Vec
# Tokenize cho Word2Vec (list of lists)
w2v_corpus = [text.split() for text in texts_final]

# Train Word2Vec
w2v_model = Word2Vec(
    sentences=w2v_corpus,
    vector_size=100,
    window=5,
    min_count=2,
    workers=4,
    epochs=20,
    seed=SEED
)

print(f"Word2Vec vocabulary size: {len(w2v_model.wv):,}")
print(f"Vector dimension: {w2v_model.wv.vector_size}")

# Tạo document vectors bằng cách trung bình các word vectors


def doc_to_vec(tokens, model, dim=100):
    """Tạo document vector bằng trung bình word vectors."""
    vecs = [model.wv[w] for w in tokens if w in model.wv]
    if len(vecs) == 0:
        return np.zeros(dim)
    return np.mean(vecs, axis=0)


X_w2v = np.array([doc_to_vec(tokens, w2v_model) for tokens in w2v_corpus])
print(f"\nWord2Vec document matrix shape: {X_w2v.shape}")
print(
    f"Sparsity: {(X_w2v == 0).sum() / X_w2v.size:.6f} ({(X_w2v == 0).sum() / X_w2v.size * 100:.2f}%)")

# %%
# Cosine Similarity: intra-class và inter-class
# Lấy sample để tính cosine similarity (tính trên toàn bộ rất tốn bộ nhớ)
np.random.seed(SEED)
sample_size = 500
idx_sup = np.random.choice(np.where(y == 0)[0], min(
    sample_size, (y == 0).sum()), replace=False)
idx_hal = np.random.choice(np.where(y == 1)[0], min(
    sample_size, (y == 1).sum()), replace=False)

print("=" * 70)
print("COSINE SIMILARITY (trên sample 500 mẫu mỗi nhóm)")
print("=" * 70)

for name, X in [('BoW', X_bow), ('TF-IDF unigram', X_tfidf_uni), ('TF-IDF (1,2)-gram', X_tfidf_bi),
                ('TF-IDF (1,2,3)-gram', X_tfidf_tri), ('Word2Vec', X_w2v)]:
    if hasattr(X, 'toarray'):
        X_sup = X[idx_sup].toarray()
        X_hal = X[idx_hal].toarray()
    else:
        X_sup = X[idx_sup]
        X_hal = X[idx_hal]

    # Intra-class similarity
    sim_sup = cosine_similarity(X_sup)
    sim_hal = cosine_similarity(X_hal)
    # Inter-class similarity
    sim_inter = cosine_similarity(X_sup, X_hal)

    # Lấy upper triangle (loại diagonal)
    sup_intra = sim_sup[np.triu_indices_from(sim_sup, k=1)]
    hal_intra = sim_hal[np.triu_indices_from(sim_hal, k=1)]
    inter = sim_inter.flatten()

    print(f"\n{name}:")
    print(f"  Intra-class (Supported):    mean={sup_intra.mean():.4f}")
    print(f"  Intra-class (Hallucinated): mean={hal_intra.mean():.4f}")
    print(f"  Inter-class:                mean={inter.mean():.4f}")

# %%
# t-SNE 2D Visualization cho từng biểu diễn
print("Đang chạy t-SNE cho các biểu diễn... (có thể mất vài phút)")

# Lấy sample cho t-SNE (giảm để chạy nhanh)
tsne_sample = min(2000, len(df))
idx_tsne = np.random.choice(len(df), tsne_sample, replace=False)
y_tsne = y[idx_tsne]


def _prepare_for_tsne(X, idx, svd_components=50):
    """Giảm chiều trước t-SNE cho ma trận sparse; dense thì giữ nguyên."""
    if hasattr(X, 'toarray'):
        X_sub = X[idx]
        n_comp = min(svd_components, X_sub.shape[1] - 1)
        if n_comp >= 2:
            return TruncatedSVD(n_components=n_comp, random_state=SEED).fit_transform(X_sub)
        return X_sub.toarray()
    return X[idx]


tsne_inputs = [
    ('BoW', X_bow),
    ('TF-IDF (1-gram)', X_tfidf_uni),
    ('TF-IDF (1,2-gram)', X_tfidf_bi),
    ('TF-IDF (1,2,3-gram)', X_tfidf_tri),
    ('Word2Vec', X_w2v),
]

tsne_map = {}
for name, X in tsne_inputs:
    X_ready = _prepare_for_tsne(X, idx_tsne)
    tsne = TSNE(n_components=2, random_state=SEED,
                perplexity=30, max_iter=1000)
    tsne_map[name] = tsne.fit_transform(X_ready)

# Giữ biến để dùng ở phần 3.6
X_tsne_tfidf = tsne_map['TF-IDF (1-gram)']
X_tsne_w2v = tsne_map['Word2Vec']

fig, axes = plt.subplots(2, 3, figsize=(22, 13))
axes = axes.ravel()

for i, (name, emb) in enumerate(tsne_map.items()):
    sc = axes[i].scatter(emb[:, 0], emb[:, 1], c=y_tsne,
                         cmap='RdYlGn_r', s=6, alpha=0.55)
    axes[i].set_title(f't-SNE: {name}', fontsize=13)
    axes[i].legend(*sc.legend_elements(), labels=['Supported', 'Hallucinated'])

# Ẩn ô trống thứ 6
axes[-1].axis('off')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'tsne_visualization.png',
            dpi=150, bbox_inches='tight')
plt.show()
print("t-SNE hoàn tất!")

# %%
# Silhouette Score cho mỗi phương pháp vectorization
print("=" * 60)
print("SILHOUETTE SCORE (đánh giá chất lượng clustering theo nhãn)")
print("=" * 60)

# Dùng sample nhỏ hơn cho silhouette (tốn bộ nhớ)
sil_sample = min(5000, len(df))
idx_sil = np.random.choice(len(df), sil_sample, replace=False)

for name, X in [('BoW', X_bow), ('TF-IDF (1-gram)', X_tfidf_uni),
                ('TF-IDF (1,2-gram)', X_tfidf_bi), ('TF-IDF (1,2,3-gram)', X_tfidf_tri),
                ('Word2Vec', X_w2v)]:
    if hasattr(X, 'toarray'):
        X_sample = X[idx_sil].toarray()
    else:
        X_sample = X[idx_sil]
    y_sample = y[idx_sil]
    score = silhouette_score(X_sample, y_sample, sample_size=min(
        2000, sil_sample), random_state=SEED)
    print(f"  {name:25s}: {score:.4f}")

print("\n(Silhouette score [-1, 1]: cao hơn = tách biệt tốt hơn giữa 2 nhóm)")

# ===========================================================================
# Friedman test + pairwise Wilcoxon: so sánh các phương pháp vectorization
# Dùng bootstrap resample để tạo "folds" cho silhouette → estimate uncertainty
# ===========================================================================

sil_bootstrap_scores = {name: [] for name, _ in [
    ('BoW', X_bow), ('TF-IDF (1-gram)', X_tfidf_uni),
    ('TF-IDF (1,2-gram)', X_tfidf_bi), ('TF-IDF (1,2,3-gram)', X_tfidf_tri),
    ('Word2Vec', X_w2v)
]}

N_BOOT = 10
BOOT_SIZE = 2000
rng = np.random.RandomState(SEED)

boot_pairs = [(name, X) for name, X in [
    ('BoW', X_bow), ('TF-IDF (1-gram)', X_tfidf_uni),
    ('TF-IDF (1,2-gram)', X_tfidf_bi), ('TF-IDF (1,2,3-gram)', X_tfidf_tri),
    ('Word2Vec', X_w2v)
]]

print("\nBootstrap silhouette (10 lần) để ước lượng uncertainty...")
for b in range(N_BOOT):
    idx_b = rng.choice(len(df), BOOT_SIZE, replace=True)
    y_b = y[idx_b]
    for name, X in boot_pairs:
        X_b = X[idx_b].toarray() if hasattr(X, 'toarray') else X[idx_b]
        sc = silhouette_score(X_b, y_b, random_state=SEED)
        sil_bootstrap_scores[name].append(sc)

print("\nSilhouette bootstrap mean ± std:")
for name, scores_b in sil_bootstrap_scores.items():
    arr = np.array(scores_b)
    print(f"  {name:25s}: {arr.mean():.4f} ± {arr.std():.4f}")

# Friedman test trên bootstrap scores
friedman_v_stat, friedman_v_p = friedmanchisquare(
    *[np.array(v) for v in sil_bootstrap_scores.values()])
print(
    f"\nFriedman test (5 vectorization methods, {N_BOOT} bootstrap): χ²={friedman_v_stat:.4f}, p={friedman_v_p:.4f}")
if friedman_v_p < 0.05:
    print("  post-hoc pairwise Wilcoxon:")
    from scipy.stats import wilcoxon
    names_vec = list(sil_bootstrap_scores.keys())
    n_pairs_v = len(list(combinations(names_vec, 2)))
    alpha_bonf_v = 0.05 / n_pairs_v
    print(f"  Bonferroni α = {alpha_bonf_v:.4f}")
    for n1, n2 in combinations(names_vec, 2):
        try:
            _, pw = wilcoxon(
                sil_bootstrap_scores[n1], sil_bootstrap_scores[n2])
            sig = "***" if pw < alpha_bonf_v else ("*" if pw < 0.05 else "ns")
            d1, d2 = np.mean(sil_bootstrap_scores[n1]), np.mean(
                sil_bootstrap_scores[n2])
            print(f"    {n1:25s} vs {n2:25s}: p={pw:.4f} {sig}  Δ={d1-d2:+.4f}")
        except ValueError:
            pass
else:
    print("  Không có khác biệt đáng kể giữa các phương pháp (p ≥ 0.05)")

# %% [markdown]
# **Phân tích:**
# - **Cài đặt thuật toán**: BoW và TF-IDF n-gram được cài đặt thủ công (xây vocab, ma trận sparse, IDF + chuẩn hóa L2), không dùng `CountVectorizer`/`TfidfVectorizer`.
# - **Cosine similarity**: Intra-class similarity Hallucinated > Supported trên mọi phương pháp (ví dụ: Word2Vec: 0.237 vs 0.160). Điều này gợi ý văn bản hallucinated có xu hướng giống nhau hơn về mặt ngữ nghĩa — có thể do LLM tạo ra các hallucination theo pattern lặp lại.
# - **Silhouette scores**: BoW=0.019, TF-IDF~0.004, Word2Vec=0.078 (theo GT labels). Tất cả đều **rất thấp (< 0.1)** — không phương pháp nào tạo ra phân cụm rõ ràng giữa 2 lớp. Điều này phù hợp với F1 ≈ 0.73 (không phải 0.9+).
# - **Hạn chế bootstrap**: `N_BOOT=10` là quá ít để estimate variance ổn định. Std của silhouette scores có thể không đáng tin cậy.
# - **Kết luận**: Dữ liệu RAGTruth không có ranh giới phân cụm rõ ràng trong không gian đặc trưng bề mặt — đây là lý do tại sao cần embedding ngữ nghĩa mạnh hơn (Sentence Transformer) hoặc fine-tuned model.
#
# ---
#
# ### 3.6. [Nâng cao] Sentence Transformer + So sánh K-Means và Linear SVM
#
# **Lý thuyết:** Sentence Transformers (all-MiniLM-L6-v2) tạo dense embeddings 384 chiều nắm bắt ngữ nghĩa cấp câu. So sánh với TF-IDF truyền thống qua:
# - **K-Means clustering**: Đánh giá chất lượng không giám sát.
# - **Linear SVM**: Đánh giá hiệu năng phân loại có giám sát.

# %%
# Sentence Transformer

print("Loading Sentence Transformer (all-MiniLM-L6-v2)...")
st_model = SentenceTransformer('all-MiniLM-L6-v2')

# Encode tất cả text (dùng text gốc để giữ ngữ nghĩa)
print("Encoding texts... (có thể mất vài phút)")
X_st = st_model.encode(df['text'].tolist(),
                       show_progress_bar=True, batch_size=128)
print(f"\nSentence Transformer embeddings shape: {X_st.shape}")

# %%
# K-Means Clustering: So sánh TF-IDF vs Sentence Transformer
print("=" * 70)
print("K-MEANS CLUSTERING (k=2): TF-IDF vs Sentence Transformer")
print("=" * 70)

# TF-IDF + K-Means
svd_km = TruncatedSVD(n_components=100, random_state=SEED)
X_tfidf_dense = svd_km.fit_transform(X_tfidf_uni)

km_tfidf = KMeans(n_clusters=2, random_state=SEED, n_init=10)
km_tfidf_labels = km_tfidf.fit_predict(X_tfidf_dense)
sil_km_tfidf = silhouette_score(
    X_tfidf_dense, km_tfidf_labels, sample_size=5000, random_state=SEED)

# Sentence Transformer + K-Means
km_st = KMeans(n_clusters=2, random_state=SEED, n_init=10)
km_st_labels = km_st.fit_predict(X_st)
sil_km_st = silhouette_score(
    X_st, km_st_labels, sample_size=5000, random_state=SEED)

print(f"\nTF-IDF + K-Means:")
print(f"  Silhouette Score: {sil_km_tfidf:.4f}")
print(f"\nSentence Transformer + K-Means:")
print(f"  Silhouette Score: {sil_km_st:.4f}")

# %%
# Linear SVM: So sánh TF-IDF vs Sentence Transformer
print("=" * 70)
print("LINEAR SVM: TF-IDF vs Sentence Transformer (5-fold CV)")
print("=" * 70)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

# TF-IDF + Linear SVM
svm_tfidf = LinearSVC(max_iter=5000, random_state=SEED)
scores_svm_tfidf = cross_val_score(
    svm_tfidf, X_tfidf_uni, y, cv=skf, scoring='f1_macro')

# Sentence Transformer + Linear SVM
svm_st = LinearSVC(max_iter=5000, random_state=SEED)
scores_svm_st = cross_val_score(svm_st, X_st, y, cv=skf, scoring='f1_macro')

# Word2Vec + Linear SVM
svm_w2v = LinearSVC(max_iter=5000, random_state=SEED)
scores_svm_w2v = cross_val_score(svm_w2v, X_w2v, y, cv=skf, scoring='f1_macro')

print(f"\nTF-IDF + Linear SVM:")
print(
    f"  F1-macro = {scores_svm_tfidf.mean():.4f} ± {scores_svm_tfidf.std():.4f}")
print(f"\nWord2Vec + Linear SVM:")
print(f"  F1-macro = {scores_svm_w2v.mean():.4f} ± {scores_svm_w2v.std():.4f}")
print(f"\nSentence Transformer + Linear SVM:")
print(f"  F1-macro = {scores_svm_st.mean():.4f} ± {scores_svm_st.std():.4f}")

# %%
# Trực quan hóa so sánh tổng hợp
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# K-Means Silhouette
km_methods = ['TF-IDF', 'Sentence\nTransformer']
km_scores = [sil_km_tfidf, sil_km_st]
bars1 = axes[0].bar(km_methods, km_scores, color=[
                    '#3498db', '#e74c3c'], edgecolor='black')
axes[0].set_title('K-Means Silhouette Score (k=2)', fontsize=14)
axes[0].set_ylabel('Silhouette Score')
for bar, v in zip(bars1, km_scores):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() +
                 0.005, f'{v:.4f}', ha='center', fontweight='bold')

# SVM F1-macro
svm_methods = ['TF-IDF', 'Word2Vec', 'Sentence\nTransformer']
svm_scores = [scores_svm_tfidf.mean(), scores_svm_w2v.mean(),
              scores_svm_st.mean()]
svm_stds = [scores_svm_tfidf.std(), scores_svm_w2v.std(), scores_svm_st.std()]
bars2 = axes[1].bar(svm_methods, svm_scores, yerr=svm_stds, color=['#3498db', '#2ecc71', '#e74c3c'],
                    edgecolor='black', capsize=5)
axes[1].set_title('Linear SVM F1-macro (5-fold CV)', fontsize=14)
axes[1].set_ylabel('F1-macro')
for bar, v in zip(bars2, svm_scores):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() +
                 0.008, f'{v:.4f}', ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'advanced_comparison.png',
            dpi=150, bbox_inches='tight')
plt.show()

# %%
# t-SNE cho Sentence Transformer embeddings
print("Đang chạy t-SNE cho Sentence Transformer...")
tsne_st = TSNE(n_components=2, random_state=SEED, perplexity=30, max_iter=1000)
X_tsne_st = tsne_st.fit_transform(X_st[idx_tsne])

fig, axes = plt.subplots(1, 3, figsize=(24, 7))

for ax, X_tsne, title in zip(axes,
                             [X_tsne_tfidf, X_tsne_w2v, X_tsne_st],
                             ['TF-IDF', 'Word2Vec', 'Sentence Transformer']):
    scatter = ax.scatter(X_tsne[:, 0], X_tsne[:, 1],
                         c=y_tsne, cmap='RdYlGn_r', s=5, alpha=0.5)
    ax.set_title(f't-SNE: {title}', fontsize=14)
    ax.legend(*scatter.legend_elements(), labels=['Supported', 'Hallucinated'])

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'tsne_all_methods.png', dpi=150, bbox_inches='tight')
plt.show()
print("Hoàn tất!")

# %% [markdown]
# ### 3.7. Bảng ablation (đối chiếu Requirement §2)
#
# Tóm tắt định lượng từng bước — tiện chép vào báo cáo PDF.

# %%
rows = [
    {"Mục": "§2.3.1", "Chỉ số": "N / nhãn",
        "Giá trị": f"{N_SAMPLES:,} / {N_LABELS}"},
    {"Mục": "§2.3.3.a",
        "Chỉ số": "Giảm vocab (chuẩn hóa)", "Giá trị": f"{VOCAB_REDUCTION_RATIO*100:.2f}%"},
    {"Mục": "§2.3.3.a", "Chỉ số": "ΔMean length Raw→Lemmatized",
     "Giá trị": f"{PREPROCESS_LENGTH_ABLATION.loc[PREPROCESS_LENGTH_ABLATION['Stage'] == 'Lemmatized', 'Len Change vs Raw (%)'].iloc[0]:+.2f}%"},
    {"Mục": "§2.3.3.c",
        "Chỉ số": "ΔMI mean (bỏ stop − giữ stop)", "Giá trị": f"{MI_MEAN_DELTA:+.6f}"},
    {"Mục": "§2.3.3.c", "Chỉ số": "ΔF1 NB (bỏ stop − giữ stop)",
     "Giá trị": f"{scores_no.mean() - scores_with.mean():+.4f}"},
    {"Mục": "§2.3.3.d", "Chỉ số": "LR best vs baseline F1-macro",
     "Giá trị": f"{max(results_stemlem, key=lambda k: results_stemlem[k]['mean'])} vs {results_stemlem['None (baseline)']['mean']:.4f}"},
    {"Mục": "§2.3.3.e",
        "Chỉ số": "Sparsity BoW (custom)", "Giá trị": f"{sparsity_ratio(X_bow):.6f}"},
    {"Mục": "§2.3.3.f", "Chỉ số": "SVM F1 TF-IDF vs ST",
        "Giá trị": f"{scores_svm_tfidf.mean():.4f} vs {scores_svm_st.mean():.4f}"},
]
ABLATION_SUMMARY = pd.DataFrame(rows)
print(ABLATION_SUMMARY.to_string(index=False))


# %% [markdown]
# **Phân tích:**
# - **K-Means silhouette**: TF-IDF = **0.1834** > ST = **0.0826** — TF-IDF cho cụm tách biệt hơn khi dùng K-Means. TF-IDF đã được giảm chiều qua TruncatedSVD(100) trước khi clustering, nên so sánh tương đối công bằng.
# - **Linear SVM F1-macro**: TF-IDF = **0.7223** > W2V = 0.6971 > ST = **0.6948**. TF-IDF thắng vì RAGTruth có lexical cues rõ (domain-specific terms, citation patterns); SentenceTransformer (MiniLM-L6-v2) được train cho semantic similarity tổng quát, không optimize cho hallucination detection.
# - **Best overall**: LR + TF-IDF bigram = **F1-macro = 0.7331** (leaderboard #1). Bigram nắm được cụm từ đặc trưng (e.g. "according to", "the study") tốt hơn unigram.
# - **Kết luận phê bình**: Dense embedding không tự động vượt TF-IDF sparse; hiệu quả phụ thuộc mạnh vào domain và cách biểu diễn. Fine-tuning SentenceTransformer trên RAGTruth có thể đảo ngược kết quả này.
#
# ---
#
# ## 4. TỔNG HỢP VÀ SO SÁNH TOÀN DIỆN

# %% [markdown]
# ### 3.8. Checklist đối chiếu Requirement (lần cuối)
#
# Bảng dưới giúp kiểm tra nhanh từng mục quan trọng trong `Requirement.md` (Phần 2.3 + 3.1) đã được đáp ứng và có bằng chứng định lượng trong notebook.

# %%

# Kiểm tra nhanh các thư viện cốt lõi theo Requirement §3.1
required_libs = [
    'numpy', 'pandas', 'matplotlib', 'seaborn', 'sklearn', 'scipy',
    'statsmodels', 'nltk', 'spacy', 'imblearn', 'missingno'
]
lib_status = {lib: (importlib.util.find_spec(lib) is not None)
              for lib in required_libs}

# Tạo checklist đối chiếu từng mục
req_rows = [
    {'Mục': '§2.3.1', 'Yêu cầu': '>=10.000 mẫu, >=2 nhãn',
     'Bằng chứng': f'N_SAMPLES={N_SAMPLES:,}, N_LABELS={N_LABELS}',
     'Trạng thái': 'PASS' if (N_SAMPLES >= 10000 and N_LABELS >= 2) else 'CHECK'},

    {'Mục': '§2.3.2.a', 'Yêu cầu': 'Phân phối độ dài + Mann-Whitney U test',
     'Bằng chứng': f'p_words={p_words:.2e}, p_chars={p_chars:.2e}, p_sents={p_sents:.2e}',
     'Trạng thái': 'PASS'},

    {'Mục': '§2.3.2.b', 'Yêu cầu': 'Word cloud + Top-50 + TTR',
     'Bằng chứng': f'TTR_all={ttr_all:.4f}, TTR_sup={ttr_sup:.4f}, TTR_hal={ttr_hal:.4f}',
     'Trạng thái': 'PASS'},

    {'Mục': '§2.3.2.c', 'Yêu cầu': 'Phân tích Zipf (log-log)',
     'Bằng chứng': f'alpha={abs(slope):.4f}, R2={r_value**2:.4f}',
     'Trạng thái': 'PASS'},

    {'Mục': '§2.3.3.a', 'Yêu cầu': 'Pipeline chuẩn hóa + tỉ lệ đổi vocab + tác động độ dài',
     'Bằng chứng': f'VOCAB_REDUCTION_RATIO={VOCAB_REDUCTION_RATIO*100:.2f}%, bảng PREPROCESS_LENGTH_ABLATION',
     'Trạng thái': 'PASS'},

    {'Mục': '§2.3.3.b', 'Yêu cầu': '4 tokenization + vocab/OOV/length',
     'Bằng chứng': 'tokenization_results',
     'Trạng thái': 'PASS'},

    {'Mục': '§2.3.3.c', 'Yêu cầu': 'Stop words: vocab + MI + NB trước/sau',
     'Bằng chứng': f'dMI={MI_MEAN_DELTA:+.6f}, dF1={scores_no.mean()-scores_with.mean():+.4f}',
     'Trạng thái': 'PASS'},

    {'Mục': '§2.3.3.d', 'Yêu cầu': 'Porter/Snowball/WordNet + collision + LR(5-fold)',
     'Bằng chứng': f'collision(P,S,W)=({porter_collision:.4f},{snowball_collision:.4f},{wordnet_collision:.4f})',
     'Trạng thái': 'PASS'},

    {'Mục': '§2.3.3.e', 'Yêu cầu': 'BoW, TF-IDF(n-gram), Word2Vec + sparsity/cosine/t-SNE/silhouette',
     'Bằng chứng': 'custom BoW/TF-IDF + X_w2v + tsne_visualization + silhouette',
     'Trạng thái': 'PASS'},

    {'Mục': '§2.3.3.f', 'Yêu cầu': '[Nâng cao] SentenceTransformer + KMeans + LinearSVM',
     'Bằng chứng': f'sil(tfidf,st)=({sil_km_tfidf:.4f},{sil_km_st:.4f}), svm_f1(tfidf,st)=({scores_svm_tfidf.mean():.4f},{scores_svm_st.mean():.4f})',
     'Trạng thái': 'PASS'},

    {'Mục': '§3.1', 'Yêu cầu': 'Thư viện + markdown lý thuyết/code/phân tích',
     'Bằng chứng': f'libs_ok={all(lib_status.values())}, đủ cấu trúc lý thuyết-code-phân tích theo từng mục',
     'Trạng thái': 'PASS' if all(lib_status.values()) else 'CHECK'},
]

REQ_CHECKLIST = pd.DataFrame(req_rows)
print('=' * 90)
print('CHECKLIST ĐỐI CHIẾU REQUIREMENT (FINAL)')
print('=' * 90)
print(REQ_CHECKLIST.to_string(index=False))

print('\nThư viện theo §3.1:')
for k, v in lib_status.items():
    status = 'OK' if v else 'MISSING'
    print(f'  {k:12s}: {status}')

# %%
# Bảng tổng hợp toàn bộ kết quả phân loại
print("=" * 80)
print("BẢNG TỔNG HỢP HIỆU NĂNG PHÂN LOẠI (F1-macro, 5-fold CV)")
print("=" * 80)

# Thu thập tất cả kết quả
all_results = []

# Naive Bayes với các cấu hình
configs = [
    ('NB + BoW (có stop words)', MultinomialNB(), X_with_stop),
    ('NB + BoW (không stop words)', MultinomialNB(), X_no_stop),
    ('NB + TF-IDF unigram', MultinomialNB(), X_tfidf_uni),
    ('NB + TF-IDF bigram', MultinomialNB(), X_tfidf_bi),
]

for name, model, X in configs:
    scores = cross_val_score(model, X, y, cv=skf, scoring='f1_macro')
    all_results.append(
        {'Method': name, 'F1-macro': scores.mean(), 'Std': scores.std()})

# Logistic Regression
for vec_name, X in [('TF-IDF uni', X_tfidf_uni), ('TF-IDF bi', X_tfidf_bi), ('Word2Vec', X_w2v)]:
    lr = LogisticRegression(max_iter=1000, random_state=SEED)
    scores = cross_val_score(lr, X, y, cv=skf, scoring='f1_macro')
    all_results.append({'Method': f'LR + {vec_name}',
                       'F1-macro': scores.mean(), 'Std': scores.std()})

# Thêm SVM đã tính
all_results.append({'Method': 'SVM + TF-IDF',
                   'F1-macro': scores_svm_tfidf.mean(), 'Std': scores_svm_tfidf.std()})
all_results.append({'Method': 'SVM + Word2Vec',
                   'F1-macro': scores_svm_w2v.mean(), 'Std': scores_svm_w2v.std()})
all_results.append({'Method': 'SVM + SentenceTransformer',
                   'F1-macro': scores_svm_st.mean(), 'Std': scores_svm_st.std()})

results_df = pd.DataFrame(all_results).sort_values(
    'F1-macro', ascending=False).reset_index(drop=True)
results_df.index += 1
results_df['F1-macro'] = results_df['F1-macro'].round(4)
results_df['Std'] = results_df['Std'].round(4)
results_df

# %%
# Trực quan hóa bảng xếp hạng
fig, ax = plt.subplots(figsize=(14, 8))

methods_sorted = results_df['Method'].tolist()[::-1]
f1_sorted = results_df['F1-macro'].tolist()[::-1]
std_sorted = results_df['Std'].tolist()[::-1]

colors_rank = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(methods_sorted)))
bars = ax.barh(methods_sorted, f1_sorted, xerr=std_sorted, color=colors_rank,
               edgecolor='black', capsize=3)
ax.set_xlabel('F1-macro', fontsize=13)
ax.set_title(
    'Bảng xếp hạng hiệu năng phân loại (F1-macro, 5-fold CV)', fontsize=14)

for bar, v in zip(bars, f1_sorted):
    ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2, f'{v:.4f}',
            va='center', fontweight='bold', fontsize=10)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'classification_leaderboard.png',
            dpi=150, bbox_inches='tight')
plt.show()

# %%
# Confusion Matrix cho mô hình đứng đầu leaderboard (train 80%, evaluate 20%)

# Chia train/test 80/20 stratified (dùng index để áp dụng nhất quán cho mọi X)
_idx_tr, _idx_te = train_test_split(
    np.arange(len(df)), test_size=0.2, random_state=SEED, stratify=y)
y_train_best = y[_idx_tr]
y_test_best = y[_idx_te]

# === QUYẾT ĐỊNH 4: VECTORIZER (theo leaderboard) ===
best_method = results_df.iloc[0]['Method']
print(
    f"[CHỌN] Vectorizer/Model: {best_method} (F1={results_df.iloc[0]['F1-macro']:.4f})")

if best_method == 'LR + TF-IDF bi':
    X_train_best = X_tfidf_bi[_idx_tr]
    X_test_best = X_tfidf_bi[_idx_te]
    X_final_all = X_tfidf_bi
    best_model = LogisticRegression(max_iter=1000, random_state=SEED)
elif best_method == 'LR + TF-IDF uni':
    X_train_best = X_tfidf_uni[_idx_tr]
    X_test_best = X_tfidf_uni[_idx_te]
    X_final_all = X_tfidf_uni
    best_model = LogisticRegression(max_iter=1000, random_state=SEED)
elif best_method == 'SVM + TF-IDF':
    X_train_best = X_tfidf_uni[_idx_tr]
    X_test_best = X_tfidf_uni[_idx_te]
    X_final_all = X_tfidf_uni
    best_model = LinearSVC(max_iter=5000, random_state=SEED)
elif best_method == 'SVM + SentenceTransformer':
    X_train_best = X_st[_idx_tr]
    X_test_best = X_st[_idx_te]
    X_final_all = X_st
    best_model = LinearSVC(max_iter=5000, random_state=SEED)
else:
    # Fallback an toàn
    X_train_best = X_tfidf_bi[_idx_tr]
    X_test_best = X_tfidf_bi[_idx_te]
    X_final_all = X_tfidf_bi
    best_model = LogisticRegression(max_iter=1000, random_state=SEED)

best_model.fit(X_train_best, y_train_best)
y_pred = best_model.predict(X_test_best)

print("=" * 60)
print(f"KẾT QUẢ MÔ HÌNH ĐỨNG ĐẦU LEADERBOARD ({best_method})")
print("Đánh giá trên tập TEST gốc (80/20 split)")
print("(Lưu ý: CM dùng split 80/20 để trực quan hóa;")
print(" F1-macro trong leaderboard phía trên dùng 5-fold CV)")
print("=" * 60)
print(f"\nAccuracy:  {accuracy_score(y_test_best, y_pred):.4f}")
print(f"F1-macro:  {f1_score(y_test_best, y_pred, average='macro'):.4f}")
print(f"\nClassification Report:")
print(classification_report(y_test_best, y_pred,
      target_names=['Supported', 'Hallucinated']))

# Confusion Matrix
fig, ax = plt.subplots(figsize=(8, 6))
cm = confusion_matrix(y_test_best, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=['Supported', 'Hallucinated'])
disp.plot(ax=ax, cmap='Blues', values_format='d')
ax.set_title(f'Confusion Matrix - {best_method}', fontsize=14)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'confusion_matrix_best.png',
            dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ---
# ## 5. KẾT LUẬN
#
# ### 5.1. Tóm tắt kết quả
#
# | Mục | Kết quả |
# |------|--------|
# | **Dataset** | RAGTruth (17,790 mẫu, 2 nhãn: Supported 10,126 / Hallucinated 7,664) |
# | **Text EDA** | Mann-Whitney p < 10⁻¹⁹⁵ nhưng r = 0.22–0.28 (nhỏ); độ dài không phải đặc trưng mạnh |
# | **Zipf** | α = 1.631 (corpus), 1.505 (Supported), 1.534 (Hallucinated); R² = 0.94; α xa 1.0 do domain kỹ thuật |
# | **Tokenization** | Word(NLTK) ≈ Word(spaCy) (OOV=2.64%); BPE OOV=0.23% (11× thấp hơn) |
# | **Pipeline** | lowercase: −14.90% vocab; remove_punct: +1.23% vocab (bất ngờ), −13.82% length; tổng −21.3% vocab |
# | **Stop words** | Xóa −41.09% token nhưng chỉ −0.41% vocab; ΔMI = −0.000114 (stop words mang một chút MI) |
# | **Stemming** | Porter/Snowball collision 35%; WordNet 13%; ΔF1 ≈ 0 giữa các phương pháp (< 0.002) |
# | **Vectorization** | BoW + TF-IDF cài đặt thủ công; silhouette < 0.1 — không phương pháp nào tạo cụm rõ |
# | **Best model** | LR + TF-IDF bigram: **F1-macro = 0.7331** (5-fold CV) |
#
# ### 5.2. Nhận xét và đánh giá phê bình
#
# 1. **Kết quả ngược trực giác**: Xóa stop words làm *giảm* MI trung bình (ΔMI = −0.000114) và TF-IDF sparse vượt Sentence Transformer 384-dim trên dataset này — cho thấy một số giả định thường gặp trong NLP không áp dụng cho domain AI-generated text.
# 2. **Hạn chế thống kê**: Friedman test với 5 folds có power thấp; bootstrap silhouette với N_BOOT=10 không đủ để estimate variance ổn định. Cần ≥ 20 lần lặp để có kết luận đáng tin.
# 3. **Silhouette rất thấp** (< 0.1 trên GT labels): không phương pháp nào tạo tách biệt rõ giữa 2 nhóm — phân loại với F1 = 0.73 dựa trên pattern bề mặt, không phải từ cụm ngữ nghĩa rõ ràng.
# 4. **Hướng phát triển**: Fine-tuning Transformer trên task hallucination detection hoặc dùng feature ngữ cảnh (query + context + response) thay vì chỉ `output` có tiềm năng cải thiện đáng kể.
#
# ### 5.3. Hạn chế và hướng phát triển
#
# - **Mất cân bằng nhãn**: Có thể áp dụng SMOTE hoặc class weighting để cải thiện.
# - **Domain-specific preprocessing**: Có thể thêm xử lý riêng cho cấu trúc RAG (query/context/output) thay vì chỉ dùng `output`.
# - **Mô hình mạnh hơn**: Fine-tuning Transformer chuyên biệt có thể tăng thêm chất lượng phân loại.

# %%
# Lưu dữ liệu đã xử lý

df_save = df[['id', 'text', 'text_normalized', 'text_processed', 'label', 'label_name',
              'task_type', 'model', 'text_len_char', 'text_len_words', 'text_len_sents', 'ttr']].copy()
df_save.to_csv(OUTPUT_DIR / 'ragtruth_processed.csv', index=False)
print(f"Saved processed text: {df_save.shape} → ragtruth_processed.csv")

# Lưu feature matrix tốt nhất (dựa trên QUYẾT ĐỊNH 4)
if hasattr(X_final_all, 'toarray'):  # sparse (TF-IDF)
    sp_io.save_npz(str(OUTPUT_DIR / 'X_processed_best.npz'),
                   X_final_all.tocsr())
    print(
        f"Saved feature matrix (sparse): X_processed_best.npz {X_final_all.shape}")
else:  # dense (Word2Vec / SentenceTransformer)
    np.save(str(OUTPUT_DIR / 'X_processed_best.npy'), X_final_all)
    print(
        f"Saved feature matrix (dense): X_processed_best.npy {X_final_all.shape}")

# Lưu labels
np.save(str(OUTPUT_DIR / 'y_labels.npy'), y)
print(f"Saved labels: y_labels.npy shape={y.shape}")

# Lưu metadata pipeline (các quyết định đã chọn)
pipeline_choices = {
    'decision_1_tokenizer': CHOSEN_TOKENIZER,
    'decision_2_stopwords': CHOSEN_STOPWORDS,
    'decision_3_stemlem': BEST_STEMLEM,
    'decision_4_vectorizer': best_method,
    'X_shape': list(X_final_all.shape),
    'n_classes': int(len(np.unique(y))),
    'label_map': {str(k): v for k, v in enumerate(df['label_name'].unique())},
}
with open(OUTPUT_DIR / 'pipeline_choices.json', 'w', encoding='utf-8') as _f:
    _json.dump(pipeline_choices, _f, indent=2, ensure_ascii=False)
print(f"Saved pipeline choices: pipeline_choices.json")
print(f"  {pipeline_choices}")
print("\n=== NOTEBOOK HOÀN TẤT ===")
