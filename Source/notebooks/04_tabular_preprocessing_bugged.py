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
#     language: python
#     name: python3
# ---
# =============================================================================
# SOURCE/notebooks/04_tabular_preprocessing.py
# Notebook: Pháº§n 2 - Tiá»n xá»­ lÃ½ dá»¯ liá»‡u dáº¡ng báº£ng â€“ IEEE-CIS Fraud Detection
# NgÃ´n ngá»¯: Tiáº¿ng Viá»‡t (markdown) + Python (code)
#
# FEEDBACK tá»« leader (FeedbackFromLeader.pdf):
# [ðŸ”´ P1-CRITICAL] CÃ¡C Ká»¸ THUáº¬T ÄANG APPLY TUáº¦N Tá»° (1â†’2â†’3â†’4â†’5) - SAI YÃŠu Cáº¦U
#   Pháº£i apply Äá»˜C Láº¬P vá»›i tá»«ng method tá»«ng phase
#   Pattern Ä‘Ãºng: df_raw â†’ [impute benchmark] â†’ df_imputed (cá»‘ Ä‘á»‹nh)
#                 df_imputed â†’ [outlier benchmark riÃªng] â†’ compare
#                 df_imputed â†’ [scaling benchmark riÃªng] â†’ compare
# [ðŸ”´ P1] Cáº¥u trÃºc code rá»‘i: Bâ†’Aâ†’B - nÃªn sáº¯p xáº¿p láº¡i thÆ°á»›c EDA â†’ Imputation â†’
#   Outlier â†’ Scaling â†’ Encoding â†’ Feature Selection â†’ Imbalance
# [FIX] Sample size: dÃ¹ng SAMPLE_N = 5000 nháº¥t quÃ¡n, khÃ´ng mix full vs 5k
# [NOTE] MICE tá»‘t nháº¥t nhÆ°ng OOM â†’ ghi rÃµ lÃ½ do dÃ¹ng Median/Mean trong pipeline
# [FIX] Feature selection: Ä‘áº£m báº£o cháº¡y trÃªn Ä‘á»§ features (khÃ´ng chá»‰ 1 feature)
# [NOTE] DÃ¹ khÃ´ng sá»­ dá»¥ng má»™t method, váº«n giá»¯ visualization cá»§a nÃ³
# =============================================================================
#

# %% [markdown] _uuid="f6fc5790-0568-4277-98e6-4a483e1b24a0" _cell_guid="b3b089c3-5524-49d3-a1bd-00a0bd330580" jupyter={"outputs_hidden": false}
# # Pháº§n 2: Tiá»n xá»­ lÃ½ dá»¯ liá»‡u dáº¡ng báº£ng â€“ IEEE-CIS Fraud Detection
# #
# **Dataset:** IEEE-CIS Fraud Detection
# - `train_transaction.csv`, `train_identity.csv`
# - `test_transaction.csv`, `test_identity.csv`
# - BÃ i toÃ¡n phÃ¢n loáº¡i nhá»‹ phÃ¢n: dá»± Ä‘oÃ¡n `isFraud`
# - Dá»¯ liá»‡u cÃ³ cáº£ thuá»™c tÃ­nh sá»‘ vÃ  phÃ¢n loáº¡i, nhiá»u giÃ¡ trá»‹ thiáº¿u, máº¥t cÃ¢n báº±ng lá»›p nghiÃªm trá»ng.

# %% _uuid="bde6d67f-fa82-49a8-9515-433577704435" _cell_guid="87d34020-7f9c-41de-86d7-609545b1f2e1" jupyter={"outputs_hidden": false}
# ============================================================
# 0. CÃ i Ä‘áº·t thÆ° viá»‡n cáº§n thiáº¿t
# ============================================================
# # !pip install missingno pyampute category_encoders umap-learn imbalanced-learn scikit-learn scipy statsmodels

# %% _uuid="420d8d7b-8447-4eec-95b6-7a45827c3ac8" _cell_guid="ffca4767-d2d7-4719-9838-50a483bcce1f" jupyter={"outputs_hidden": false}
import json
from sklearn.metrics import (precision_score, recall_score,
                             f1_score, roc_auc_score, classification_report)
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE, ADASYN
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_selection import f_classif, chi2, mutual_info_classif, RFE
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
import category_encoders as ce
from scipy.stats import levene
from sklearn.preprocessing import (MinMaxScaler, StandardScaler,
                                   RobustScaler, QuantileTransformer)
from scipy.stats import ks_2samp
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn.metrics import mean_squared_error
from sklearn.impute import IterativeImputer
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.feature_selection import mutual_info_classif
import missingno as msno
from scipy import stats
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import gc
import os

warnings.filterwarnings("ignore")
pd.options.display.max_columns = 80
pd.options.display.max_rows = 60
plt.rcParams['figure.figsize'] = (12, 5)
plt.rcParams['font.size'] = 11
SEED = 42
np.random.seed(SEED)

# %% [markdown] _uuid="c08f2368-c4fb-4839-a756-7cd843000063" _cell_guid="f5912673-fa1b-47a7-a201-40c73f4902ca" jupyter={"outputs_hidden": false}
# ## 0. Load & merge dá»¯ liá»‡u

# %% _uuid="22dee88d-1ad8-4f38-ae1b-8e3fc9c8dcff" _cell_guid="1cf14a38-24d2-4897-b81d-2cae7c343684" jupyter={"outputs_hidden": false}
# â”€â”€ ÄÆ°á»ng dáº«n dá»¯ liá»‡u â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

IS_KAGGLE = os.path.exists('/kaggle/input')


def _find_tabular_root() -> str:
    """TÃ¬m thÆ° má»¥c chá»©a train_transaction.csv."""
    if IS_KAGGLE:
        return '/kaggle/input/ieee-fraud-detection'
    candidates = [
        Path.cwd() / 'data' / 'tabular' / 'raw',
        Path.cwd().parent / 'data' / 'tabular' / 'raw',
        Path.cwd().parent.parent / 'data' / 'tabular' / 'raw',
        Path.cwd() / 'Source' / 'Dataset',
    ]
    for p in candidates:
        if (p / 'train_transaction.csv').exists():
            return str(p)
    raise FileNotFoundError(
        "KhÃ´ng tÃ¬m tháº¥y train_transaction.csv!\n"
        "Giáº£i nÃ©n ieee-fraud-detection.zip vÃ o data/tabular/raw/"
    )


DATA_DIR = _find_tabular_root()

if IS_KAGGLE:
    OUTPUT_DIR = '/kaggle/working'
else:
    _out = Path.cwd() / 'output'
    if not _out.exists():
        _out = Path(DATA_DIR).parent / 'output'
    OUTPUT_DIR = str(_out)

os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"{'[Kaggle]' if IS_KAGGLE else '[Local]'} DATA_DIR   = {DATA_DIR}")
print(f"{'[Kaggle]' if IS_KAGGLE else '[Local]'} OUTPUT_DIR = {OUTPUT_DIR}")

print("Äang táº£i dá»¯ liá»‡u...")
train_transaction = pd.read_csv(
    os.path.join(DATA_DIR, 'train_transaction.csv'))
train_identity = pd.read_csv(os.path.join(DATA_DIR, 'train_identity.csv'))
test_transaction = pd.read_csv(os.path.join(DATA_DIR, 'test_transaction.csv'))
test_identity = pd.read_csv(os.path.join(DATA_DIR, 'test_identity.csv'))

# Merge transaction + identity theo TransactionID (left join)
train = pd.merge(train_transaction, train_identity,
                 on='TransactionID', how='left')
test = pd.merge(test_transaction,  test_identity,
                on='TransactionID', how='left')

del train_transaction, train_identity, test_transaction, test_identity
gc.collect()

print(f"Train: {train.shape[0]:,} dÃ²ng Ã— {train.shape[1]} cá»™t")
print(f"Test : {test.shape[0]:,} dÃ²ng Ã— {test.shape[1]} cá»™t")


# %% _uuid="5238513a-c5a4-4fae-92b8-eb7eb0c7b792" _cell_guid="7a10f196-7e48-46aa-bc60-3b19ed4aa112" jupyter={"outputs_hidden": false}
# â”€â”€ Giáº£m sá»­ dá»¥ng bá»™ nhá»› (downcast kiá»ƒu dá»¯ liá»‡u) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def reduce_mem_usage(df, verbose=True):
    """Giáº£m bá»™ nhá»› báº±ng cÃ¡ch chuyá»ƒn kiá»ƒu dtype nhá» hÆ¡n."""
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage(deep=True).sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min, c_max = df[col].min(), df[col].max()
            if str(col_type)[:3] == 'int':
                for dtype in [np.int8, np.int16, np.int32, np.int64]:
                    if c_min > np.iinfo(dtype).min and c_max < np.iinfo(dtype).max:
                        df[col] = df[col].astype(dtype)
                        break
            else:
                if (c_min > np.finfo(np.float32).min and
                        c_max < np.finfo(np.float32).max):
                    df[col] = df[col].astype(np.float32)
    end_mem = df.memory_usage(deep=True).sum() / 1024**2
    if verbose:
        print(f"  Bá»™ nhá»› giáº£m tá»« {start_mem:.1f} MB -> {end_mem:.1f} MB "
              f"({100*(start_mem-end_mem)/start_mem:.1f}%)")
    return df


train = reduce_mem_usage(train)
test = reduce_mem_usage(test)
gc.collect()

# %% [markdown] _uuid="f34a13c8-1c82-4496-add6-eaf948315454" _cell_guid="50d2d328-9a42-44e2-9b65-a62c2a6758e6" jupyter={"outputs_hidden": false}
# #### Cell 1 â€“ Shape, dtypes, memory usage

# %% _uuid="400be013-ae56-44f8-b206-fdc2842f8b5b" _cell_guid="b992f93d-c7b8-457c-bd48-6bac3ec772a4" jupyter={"outputs_hidden": false}
print(f"Train shape : {train.shape}")
print(f"Test  shape : {test.shape}\n")

# PhÃ¢n loáº¡i dtype
dtype_summary = train.dtypes.value_counts()
print("Kiá»ƒu dá»¯ liá»‡u:")
print(dtype_summary.to_string())

mem_mb = train.memory_usage(deep=True).sum() / 1024**2
print(f"\nBá»™ nhá»› train: {mem_mb:.1f} MB")

# %% [markdown] _uuid="d3b21416-8d51-49f8-9665-a5a637343bcd" _cell_guid="2a072991-4bc5-4801-be53-d985d587802f" jupyter={"outputs_hidden": false}
# #### Cell 2 â€“ PhÃ¢n nhÃ³m cá»™t theo Ã½ nghÄ©a nghiá»‡p vá»¥
# #
# Dataset IEEE-CIS gá»“m **~400 cá»™t** Ä‘Æ°á»£c Vesta Corporation Ä‘áº·t tÃªn cÃ³ chá»§ Ä‘Ã­ch.
# Ta cÃ³ thá»ƒ chia toÃ n bá»™ thuá»™c tÃ­nh thÃ nh **9 nhÃ³m** dá»±a theo Ã½ nghÄ©a nghiá»‡p vá»¥:
# #
# | NhÃ³m | Cá»™t Ä‘áº¡i diá»‡n | Ã nghÄ©a |
# |---|---|---|
# | **Transaction** | `TransactionDT`, `TransactionAmt`, `ProductCD` | ThÃ´ng tin cÆ¡ báº£n cá»§a giao dá»‹ch: thá»i Ä‘iá»ƒm, giÃ¡ trá»‹, loáº¡i sáº£n pháº©m |
# | **Card** | `card1`â€“`card6` | ThÃ´ng tin tháº» thanh toÃ¡n: ID, máº¡ng tháº», loáº¡i tháº», ngÃ¢n hÃ ng phÃ¡t hÃ nh |
# | **Address/Dist** | `addr1`, `addr2`, `dist1`, `dist2` | Äá»‹a chá»‰ thanh toÃ¡n vÃ  khoáº£ng cÃ¡ch Ä‘á»‹a lÃ½ liÃªn quan |
# | **Email** | `P_emaildomain`, `R_emaildomain` | Domain email cá»§a ngÆ°á»i mua (P) vÃ  ngÆ°á»i nháº­n (R) |
# | **C (count)** | `C1`â€“`C14` | Äáº¿m cÃ¡c sá»± kiá»‡n liÃªn quan Ä‘áº¿n tháº»/Ä‘á»‹a chá»‰ (Vesta khÃ´ng cÃ´ng bá»‘ chi tiáº¿t) |
# | **D (timedelta)** | `D1`â€“`D15` | Khoáº£ng thá»i gian giá»¯a cÃ¡c sá»± kiá»‡n (ngÃ y tá»« giao dá»‹ch trÆ°á»›c, tá»« khi má»Ÿ tháº», v.v.) |
# | **M (match)** | `M1`â€“`M9` | Biáº¿n nhá»‹ phÃ¢n "match": khá»›p tÃªn, Ä‘á»‹a chá»‰, v.v. giá»¯a cÃ¡c báº£n ghi |
# | **V (Vesta)** | `V1`â€“`V339` | Äáº·c trÆ°ng ká»¹ thuáº­t do Vesta tÃ­nh toÃ¡n ná»™i bá»™ (enriched features) |
# | **Identity** | `id_01`â€“`id_38`, `DeviceType`, `DeviceInfo` | ThÃ´ng tin thiáº¿t bá»‹ vÃ  danh tÃ­nh: OS, trÃ¬nh duyá»‡t, loáº¡i thiáº¿t bá»‹ |
# #
# ### Táº¡i sao phÃ¢n nhÃ³m nhÆ° váº­y?
# #
# **1. Transaction** â€“ Ba cá»™t thuáº§n giao dá»‹ch táº¡o thÃ nh "xÆ°Æ¡ng sá»‘ng": thá»i Ä‘iá»ƒm (`TransactionDT`)
# vÃ  giÃ¡ trá»‹ (`TransactionAmt`) lÃ  hai Ä‘áº·c trÆ°ng quan trá»ng nháº¥t; `ProductCD` cho biáº¿t loáº¡i hÃ ng hÃ³a.
# PhÃ¢n nhÃ³m riÃªng Ä‘á»ƒ dá»… so sÃ¡nh phÃ¢n phá»‘i theo lá»›p (Fraud vs. Normal).
# #
# **2. Card** â€“ SÃ¡u cá»™t `card1`â€“`card6` Ä‘á»u mÃ´ táº£ tháº» thanh toÃ¡n tá»« cÃ¡c gÃ³c Ä‘á»™ khÃ¡c nhau
# (ID tháº», máº¡ng Visa/MC, tháº» debit/credit, ngÃ¢n hÃ ng phÃ¡t hÃ nh, quá»‘c gia).
# NhÃ³m láº¡i vÃ¬ chÃºng cÃ¹ng xÃ¡c Ä‘á»‹nh **danh tÃ­nh cá»§a phÆ°Æ¡ng tiá»‡n thanh toÃ¡n**.
# #
# **3. Address/Dist** â€“ Hai cá»™t Ä‘á»‹a chá»‰ (`addr1`, `addr2`) vÃ  hai khoáº£ng cÃ¡ch (`dist1`, `dist2`)
# Ä‘á»u liÃªn quan Ä‘áº¿n vá»‹ trÃ­ Ä‘á»‹a lÃ½. Gian láº­n thÆ°á»ng xáº£y ra khi Ä‘á»‹a chá»‰ giao hÃ ng khÃ¡c xa
# Ä‘á»‹a chá»‰ tháº» â†’ nhÃ³m láº¡i Ä‘á»ƒ phÃ¢n tÃ­ch tÆ°Æ¡ng quan khÃ´ng gian.
# #
# **4. Email** â€“ Hai domain email (purchaser vs. recipient) pháº£n Ã¡nh **hÃ nh vi táº¡o tÃ i khoáº£n**.
# Khi P_email â‰  R_email hoáº·c dÃ¹ng domain báº¥t thÆ°á»ng â†’ tÃ­n hiá»‡u gian láº­n. NhÃ³m riÃªng Ä‘á»ƒ
# táº¡o feature `email_match` vÃ  phÃ¢n tÃ­ch fraud rate theo domain.
# #
# **5. C (count)** â€“ Theo tÃ i liá»‡u Vesta, C1â€“C14 lÃ  cÃ¡c **biáº¿n Ä‘áº¿m** (count) vá» lá»‹ch sá»­
# giao dá»‹ch liÃªn quan Ä‘áº¿n tháº»/Ä‘á»‹a chá»‰ trong quÃ¡ khá»©. GiÃ¡ trá»‹ lá»›n = tháº»/Ä‘á»‹a chá»‰ Ä‘Ã£ dÃ¹ng nhiá»u.
# NhÃ³m láº¡i vÃ¬ táº¥t cáº£ cÃ¹ng kiá»ƒu "lá»‹ch sá»­ táº§n suáº¥t".
# #
# **6. D (timedelta)** â€“ D1â€“D15 lÃ  **khoáº£ng thá»i gian** (ngÃ y) giá»¯a cÃ¡c sá»± kiá»‡n:
# D1 = sá»‘ ngÃ y tá»« giao dá»‹ch trÆ°á»›c trÃªn cÃ¹ng tháº», D4/D5 = tá»« khi má»Ÿ tháº», v.v.
# NhÃ³m láº¡i vÃ¬ chÃºng Ä‘á»u Ä‘o **temporal gap** vÃ  cÃ³ cáº¥u trÃºc missing khÃ¡c nhau theo tá»«ng cá»™t.
# #
# **7. M (match)** â€“ M1â€“M9 lÃ  **cá» khá»›p** (match flag) dáº¡ng T/F/NaN:
# M1 = khá»›p tÃªn tháº», M2 = khá»›p Ä‘á»‹a chá»‰. NhÃ³m láº¡i vÃ¬ táº¥t cáº£ Ä‘á»u lÃ  biáº¿n phÃ¢n loáº¡i nhá»‹ phÃ¢n
# vÃ  cáº§n xá»­ lÃ½ encoding Ä‘á»“ng nháº¥t.
# #
# **8. V (Vesta)** â€“ 339 cá»™t V1â€“V339 lÃ  **Ä‘áº·c trÆ°ng bÃ­ máº­t** do Vesta tÃ­nh toÃ¡n. ChÃºng cÃ³ cáº¥u trÃºc
# missing ráº¥t rÃµ rÃ ng theo 11 sub-group (G1â€“G11) â†’ nhÃ³m láº¡i Ä‘á»ƒ phÃ¢n tÃ­ch missing pattern
# vÃ  giáº£m chiá»u báº±ng PCA/feature selection theo sub-group.
# #
# **9. Identity** â€“ ToÃ n bá»™ cá»™t `id_` vÃ  `DeviceType`/`DeviceInfo` Ä‘áº¿n tá»« báº£ng `train_identity`
# (left-join â†’ ~40% transaction khÃ´ng cÃ³ identity record). NhÃ³m riÃªng Ä‘á»ƒ phÃ¢n tÃ­ch tÃ¡c Ä‘á»™ng
# cá»§a "cÃ³/khÃ´ng cÃ³ thiáº¿t bá»‹" lÃªn fraud rate vÃ  xÃ¢y dá»±ng feature `has_identity`.

# %% _uuid="6fe1a133-00e2-428e-b355-3ff48a8569d2" _cell_guid="b68d9eea-8963-44fd-9000-b90c12f04196" jupyter={"outputs_hidden": false}
# â”€â”€ Äá»‹nh nghÄ©a 9 nhÃ³m cá»™t vÃ  in ra thÃ´ng tin tá»•ng quÃ¡t â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
groups = {
    'Transaction': ['TransactionDT', 'TransactionAmt', 'ProductCD'],
    'Card': [f'card{i}' for i in range(1, 7)],
    'Address/Dist': ['addr1', 'addr2', 'dist1', 'dist2'],
    'Email': ['P_emaildomain', 'R_emaildomain'],
    'C (count)': [f'C{i}' for i in range(1, 15)],
    'D (timedelta)': [f'D{i}' for i in range(1, 16)],
    'M (match)': [f'M{i}' for i in range(1, 10)],
    'V (Vesta)': [c for c in train.columns if c.startswith('V')],
    'Identity': [c for c in train.columns if c.startswith('id_') or c in ('DeviceType', 'DeviceInfo')],
}

print("=" * 70)
print(f"{'Tá»”NG QUAN 9 NHÃ“M Cá»˜T â€“ IEEE-CIS Fraud Detection':^70}")
print("=" * 70)

total_defined = 0
for grp_name, grp_cols in groups.items():
    exist = [c for c in grp_cols if c in train.columns]
    missing_cols = [c for c in grp_cols if c not in train.columns]
    miss_rate_grp = train[exist].isnull().mean().mean() * \
        100 if exist else float('nan')
    dtypes_grp = train[exist].dtypes.value_counts().to_dict() if exist else {}
    dtype_str = ', '.join(f'{str(k)}Ã—{v}' for k, v in dtypes_grp.items())
    total_defined += len(exist)

    print(f"\nâ–¶ [{grp_name}]")
    print(f"   Sá»‘ cá»™t Ä‘á»‹nh nghÄ©a : {len(grp_cols)}")
    print(f"   CÃ³ trong dataset  : {len(exist)} cá»™t")
    if missing_cols:
        print(f"   KhÃ´ng tÃ¬m tháº¥y   : {missing_cols}")
    print(f"   % thiáº¿u TB        : {miss_rate_grp:.1f}%")
    print(f"   Kiá»ƒu dá»¯ liá»‡u      : {dtype_str}")
    print(
        f"   Cá»™t               : {', '.join(exist[:10])}{'...' if len(exist) > 10 else ''}")

uncategorized = [c for c in train.columns
                 if c not in ('TransactionID', 'isFraud')
                 and not any(c in grp for grp in groups.values())]
print(f"\n{'=' * 70}")
print(f"Tá»•ng cá»™t Ä‘Ã£ phÃ¢n nhÃ³m : {total_defined}")
print(f"Cá»™t chÆ°a phÃ¢n nhÃ³m    : {len(uncategorized)}")
if uncategorized:
    print(
        f"  â†’ {uncategorized[:20]}{'...' if len(uncategorized) > 20 else ''}")
print("=" * 70)

# %% [markdown] _uuid="ab7524a5-2b34-4c44-94e9-1667d128a5fa" _cell_guid="62a769c1-471b-4e87-a2de-83062df22918" jupyter={"outputs_hidden": false}
# ---
# ## 2.2.2. PhÃ¢n tÃ­ch thá»‘ng kÃª khÃ¡m phÃ¡ (EDA)

# %% [markdown] _uuid="90316245-071a-4636-ac63-64628c0beb01" _cell_guid="ee40f037-5e34-4e02-a1da-b361a5e38091" jupyter={"outputs_hidden": false}
# ### a) Kiá»ƒm tra phÃ¢n phá»‘i â€“ D'Agostino-Pearson (vÃ¬ n > 5000)
# #
# #### Kiá»ƒm tra phÃ¢n phá»‘i lÃ  gÃ¬ vÃ  táº¡i sao cáº§n?
# #
# **Kiá»ƒm tra phÃ¢n phá»‘i** (normality test) xÃ¡c Ä‘á»‹nh liá»‡u má»™t thuá»™c tÃ­nh cÃ³ tuÃ¢n theo **phÃ¢n phá»‘i chuáº©n** $\mathcal{N}(\mu, \sigma^2)$ hay khÃ´ng.
# Káº¿t quáº£ áº£nh hÆ°á»Ÿng trá»±c tiáº¿p Ä‘áº¿n cÃ¡c bÆ°á»›c tiáº¿p theo:
# #
# | Náº¿u phÃ¢n phá»‘i... | DÃ¹ng correlation | DÃ¹ng scaler |
# |---|---|---|
# | Gáº§n chuáº©n (symmetric) | Pearson | StandardScaler |
# | Lá»‡ch / cÃ³ outlier | Spearman | RobustScaler hoáº·c QuantileTransformer |
# #
# #### Shapiro-Wilk lÃ  gÃ¬?
# #
# **Shapiro-Wilk** (1965) lÃ  kiá»ƒm Ä‘á»‹nh phÃ¢n phá»‘i chuáº©n phá»• biáº¿n nháº¥t, hoáº¡t Ä‘á»™ng báº±ng cÃ¡ch
# so sÃ¡nh cÃ¡c **order statistics** (giÃ¡ trá»‹ Ä‘Ã£ sáº¯p xáº¿p) cá»§a máº«u vá»›i cÃ¡c order statistics ká»³ vá»ng
# cá»§a phÃ¢n phá»‘i chuáº©n lÃ½ thuyáº¿t:
# #
# $$W = \frac{\left(\sum_{i=1}^{n} a_i x_{(i)}\right)^2}{\sum_{i=1}^{n}(x_i - \bar{x})^2}$$
# #
# $W$ cÃ ng gáº§n 1 thÃ¬ dá»¯ liá»‡u cÃ ng gáº§n chuáº©n. Tuy nhiÃªn, **Shapiro-Wilk máº¥t hiá»‡u lá»±c khi $n > 5\,000$**:
# vá»›i máº«u lá»›n, power cá»§a test tÄƒng ráº¥t cao khiáº¿n test phÃ¡t hiá»‡n ra nhá»¯ng lá»‡ch chuáº©n cá»±c nhá»
# khÃ´ng cÃ³ Ã½ nghÄ©a thá»±c táº¿ vÃ  gáº§n nhÆ° luÃ´n bÃ¡c bá» $H_0$.
# #
# #### D'Agostino-Pearson lÃ  gÃ¬?
# #
# **D'Agostino-Pearson** (`scipy.stats.normaltest`) lÃ  kiá»ƒm Ä‘á»‹nh phÃ¹ há»£p hÆ¡n cho dataset lá»›n.
# Thay vÃ¬ so sÃ¡nh toÃ n bá»™ phÃ¢n phá»‘i, nÃ³ chá»‰ kiá»ƒm tra hai Ä‘áº·c trÆ°ng hÃ¬nh dáº¡ng Ä‘áº·c trÆ°ng cá»§a phÃ¢n phá»‘i chuáº©n:
# #
# - **Skewness** $\hat{\gamma}_1$: Ä‘o tÃ­nh Ä‘á»‘i xá»©ng cá»§a phÃ¢n phá»‘i. PhÃ¢n phá»‘i chuáº©n hoÃ n toÃ n Ä‘á»‘i xá»©ng nÃªn $\gamma_1 = 0$. GiÃ¡ trá»‹ dÆ°Æ¡ng â†’ Ä‘uÃ´i pháº£i dÃ i hÆ¡n; Ã¢m â†’ Ä‘uÃ´i trÃ¡i dÃ i hÆ¡n.
# - **Excess kurtosis** $\hat{\gamma}_2$: Ä‘o Ä‘á»™ "nhá»n" vÃ  Ä‘á»™ náº·ng cá»§a Ä‘uÃ´i. PhÃ¢n phá»‘i chuáº©n cÃ³ $\gamma_2 = 0$. GiÃ¡ trá»‹ dÆ°Æ¡ng â†’ Ä‘uÃ´i náº·ng hÆ¡n chuáº©n (leptokurtic); Ã¢m â†’ Ä‘uÃ´i nháº¹ hÆ¡n (platykurtic).
# #
# Hai Ä‘áº¡i lÆ°á»£ng nÃ y Ä‘Æ°á»£c chuáº©n hÃ³a thÃ nh $Z_1, Z_2$ rá»“i káº¿t há»£p thÃ nh thá»‘ng kÃª tá»•ng há»£p phÃ¢n phá»‘i $\chi^2(2)$:
# #
# $$K^2 = Z_1(\hat{\gamma}_1)^2 + Z_2(\hat{\gamma}_2)^2 \;\sim\; \chi^2(2) \quad \text{dÆ°á»›i } H_0$$
# #
# - **$H_0$**: dá»¯ liá»‡u tuÃ¢n theo phÃ¢n phá»‘i chuáº©n
# - $p > 0.05$: khÃ´ng bÃ¡c bá» $H_0$ â€“ thuá»™c tÃ­nh gáº§n chuáº©n
# - $p \leq 0.05$: bÃ¡c bá» $H_0$ â€“ thuá»™c tÃ­nh lá»‡ch hoáº·c cÃ³ Ä‘uÃ´i náº·ng

# %% _uuid="070a09b3-1132-42af-8e78-de979f92eb29" _cell_guid="ece8b423-01f5-4da9-9116-e8353cfbef6f" jupyter={"outputs_hidden": false}

# Chá»‰ láº¥y cÃ¡c cá»™t sá»‘ (khÃ´ng pháº£i target, khÃ´ng pháº£i ID)
num_cols = [c for c in train.select_dtypes(include=[np.number]).columns
            if c not in ('isFraud', 'TransactionID', 'TransactionDT')]

print(f"Sá»‘ cá»™t sá»‘: {len(num_cols)}")

# Cháº¡y D'Agostino-Pearson test trÃªn máº«u tá»‘i Ä‘a 20,000 dÃ²ng Ä‘á»ƒ tiáº¿t kiá»‡m thá»i gian
SAMPLE_N = min(20_000, len(train))
sample_df = train[num_cols].dropna(
    axis=1, how='all').sample(SAMPLE_N, random_state=SEED)

normality_results = []
for col in sample_df.columns:
    col_data = sample_df[col].dropna()
    if len(col_data) < 8:
        continue
    stat, p = stats.normaltest(col_data)
    normality_results.append({
        'feature': col,
        'statistic': round(stat, 4),
        'p_value': round(p, 6),
        'is_normal': p > 0.05
    })

normality_df = pd.DataFrame(normality_results)
n_normal = normality_df['is_normal'].sum()
print(
    f"\nSá»‘ thuá»™c tÃ­nh phÃ¢n phá»‘i chuáº©n (p>0.05): {n_normal}/{len(normality_df)}")
print(
    f"Sá»‘ thuá»™c tÃ­nh KHÃ”NG phÃ¢n phá»‘i chuáº©n  : {len(normality_df)-n_normal}/{len(normality_df)}")
print("\n10 thuá»™c tÃ­nh cÃ³ p-value cao nháº¥t (gáº§n chuáº©n nháº¥t):")
print(normality_df.sort_values(
    'p_value', ascending=False).head(10).to_string(index=False))

# %% [markdown] _uuid="89111427-5b03-4369-b8c6-c35ef2251ed7" _cell_guid="dd49472f-82ee-4308-b0ae-c433c80b9d3d" jupyter={"outputs_hidden": false}
# #### Nháº­n xÃ©t káº¿t quáº£ â€“ D'Agostino-Pearson test
# #
# | Chá»‰ sá»‘ | GiÃ¡ trá»‹ |
# |---|---|
# | Tá»•ng cá»™t sá»‘ kiá»ƒm tra | **400** |
# | PhÃ¢n phá»‘i chuáº©n ($p > 0.05$) | **1 / 400** (0.25%) |
# | KhÃ´ng phÃ¢n phá»‘i chuáº©n | **399 / 400** (99.75%) |
# #
# **Káº¿t luáº­n**: Gáº§n nhÆ° **toÃ n bá»™ dataset khÃ´ng tuÃ¢n theo phÃ¢n phá»‘i chuáº©n** â€“ Ä‘Ã¢y lÃ  káº¿t quáº£
# Ä‘iá»ƒn hÃ¬nh vá»›i dá»¯ liá»‡u tÃ i chÃ­nh/giao dá»‹ch thá»±c táº¿. Cá»™t duy nháº¥t gáº§n chuáº©n lÃ  **`id_25`**
# ($p = 0.627$), vá»‘n lÃ  má»™t cá»™t identity thÆ°a dá»¯ liá»‡u vÃ  Ã­t giÃ¡ trá»‹ khÃ¡c nhau.
# #
# **PhÃ¢n tÃ­ch top 10 cá»™t:**
# #
# - **`id_25`** ($p = 0.627$): Cá»™t identity duy nháº¥t qua Ä‘Æ°á»£c ngÆ°á»¡ng $p > 0.05$.
#   Nhiá»u kháº£ nÄƒng do phÃ¢n phá»‘i Ä‘á»“ng Ä‘á»u trÃªn táº­p giÃ¡ trá»‹ nhá», khÃ´ng pháº£i phÃ¢n phá»‘i chuáº©n thá»±c sá»±.
# - **`TransactionAmt`** ($K^2 = 25{,}679$, $p \approx 0$): Lá»‡ch pháº£i cá»±c máº¡nh â€“
#   giao dá»‹ch nhá» ráº¥t nhiá»u, giao dá»‹ch lá»›n ráº¥t Ã­t (log-normal Ä‘iá»ƒn hÃ¬nh).
# - **`V231`â€“`V234`**, **`V225`** ($K^2 > 5{,}000$): CÃ¡c Ä‘áº·c trÆ°ng Vesta cÃ³ phÃ¢n phá»‘i
#   Ä‘uÃ´i náº·ng, nhiá»u giÃ¡ trá»‹ báº±ng 0 hoáº·c táº­p trung cá»±c ká»³ táº¡i má»™t Ä‘iá»ƒm.
# #
# **Há»‡ quáº£ cho cÃ¡c bÆ°á»›c tiáº¿p theo:**
# #
# | BÆ°á»›c | Quyáº¿t Ä‘á»‹nh | LÃ½ do |
# |---|---|---|
# | **Correlation** | DÃ¹ng **Spearman** lÃ m máº·c Ä‘á»‹nh | 399/400 cá»™t khÃ´ng chuáº©n |
# | **Scaling** | **RobustScaler** hoáº·c **QuantileTransformer** | TrÃ¡nh bá»‹ kÃ©o bá»Ÿi outlier |
# | **Imputation** | **Median** (khÃ´ng pháº£i Mean) | Mean nháº¡y cáº£m vá»›i lá»‡ch pháº£i |
# | **Feature engineering** | Log-transform `TransactionAmt`, cÃ¡c cá»™t C/D/V | Giáº£m skewness trÆ°á»›c khi Ä‘Æ°a vÃ o mÃ´ hÃ¬nh tuyáº¿n tÃ­nh |

# %% _uuid="98415b03-e4d1-42d2-b55e-42a4d0b1284c" _cell_guid="822a319a-e13c-4a5c-b111-86fa543b6aa2" jupyter={"outputs_hidden": false}

# Biá»ƒu Ä‘á»“ phÃ¢n phá»‘i cá»§a 12 thuá»™c tÃ­nh sá»‘ Ä‘áº§u tiÃªn
fig, axes = plt.subplots(3, 4, figsize=(20, 12))
# KHÃ”NG dÃ¹ng .dropna() trÃªn toÃ n bá»™ 12 cá»™t -> cÃ³ thá»ƒ máº¥t háº¿t row do NaN chÃ©o nhau
sample_plot = train[num_cols[:12]].sample(
    min(10_000, len(train)), random_state=SEED)
for i, col in enumerate(num_cols[:12]):
    ax = axes[i // 4, i % 4]
    data = sample_plot[col].dropna()   # dropna chá»‰ cho cá»™t nÃ y
    if len(data) == 0:
        ax.set_title(f'{col} (no data)', fontsize=9)
        continue
    ax.hist(data, bins=50, edgecolor='none', color='steelblue', alpha=0.8)
    ax.set_title(col, fontsize=9)
    ax.set_xlabel('')
    is_norm = normality_df.loc[normality_df.feature == col, 'is_normal']
    label = "Chuan" if (
        len(is_norm) > 0 and is_norm.values[0]) else "Khong chuan"
    ax.text(0.97, 0.95, label, ha='right', va='top', transform=ax.transAxes,
            fontsize=8, color='green' if 'Chuan' == label else 'red')
plt.suptitle(
    "PhÃ¢n phá»‘i 12 thuá»™c tÃ­nh sá»‘ Ä‘áº§u â€“ D'Agostino-Pearson test", fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'fig_01_distributions.png'),
            dpi=100, bbox_inches='tight')
plt.show()

# %% [markdown] _uuid="b0a083f1-6799-48a7-bde2-bd9c6e4cebef" _cell_guid="69bfc56a-db65-4c06-8365-6a55324dfcf3" jupyter={"outputs_hidden": false}
# ### b) PhÃ¢n tÃ­ch tÆ°Æ¡ng quan Ä‘a biáº¿n â€“ Spearman
# #
# | PhÆ°Æ¡ng phÃ¡p | CÃ´ng thá»©c | Giáº£ Ä‘á»‹nh | Khi nÃ o dÃ¹ng |
# |---|---|---|---|
# | **Pearson** | $r = \frac{\sum(x_i-\bar{x})(y_i-\bar{y})}{\sqrt{\sum(x_i-\bar{x})^2 \cdot \sum(y_i-\bar{y})^2}}$ | Tuyáº¿n tÃ­nh, phÃ¢n phá»‘i chuáº©n | Dá»¯ liá»‡u chuáº©n, quan há»‡ tuyáº¿n tÃ­nh |
# | **Spearman** | $r_s = 1 - \frac{6\sum d_i^2}{n(n^2-1)}$ | ÄÆ¡n Ä‘iá»‡u, phi tham sá»‘ | Dá»¯ liá»‡u lá»‡ch, cÃ³ outlier |
# #
# > **Lá»±a chá»n phÆ°Æ¡ng phÃ¡p:** Káº¿t quáº£ D'Agostino-Pearson test á»Ÿ bÆ°á»›c trÃªn cho tháº¥y **399/400**
# > cá»™t sá»‘ khÃ´ng tuÃ¢n theo phÃ¢n phá»‘i chuáº©n. VÃ¬ váº­y **Spearman** Ä‘Æ°á»£c chá»n lÃ m phÆ°Æ¡ng phÃ¡p
# > máº·c Ä‘á»‹nh thay vÃ¬ Pearson â€“ Spearman hoáº¡t Ä‘á»™ng trÃªn rank thay vÃ¬ giÃ¡ trá»‹ thÃ´ nÃªn
# > khÃ´ng bá»‹ áº£nh hÆ°á»Ÿng bá»Ÿi outlier vÃ  phÃ¢n phá»‘i lá»‡ch.
# #
# **PhÃ¡t hiá»‡n Ä‘a cá»™ng tuyáº¿n** ($|r_s| > 0.9$): hai Ä‘áº·c trÆ°ng cÃ³ Spearman rank correlation cao
# chá»©a thÃ´ng tin gáº§n nhÆ° trÃ¹ng láº·p. Giáº£i phÃ¡p: loáº¡i má»™t trong hai cá»™t hoáº·c dÃ¹ng PCA Ä‘á»ƒ káº¿t há»£p.

# %% _uuid="635f4241-f4ec-4836-9c0c-552e08deb237" _cell_guid="c7a758da-3bc6-4906-9ecf-9236778fbd89" jupyter={"outputs_hidden": false}
# Chá»n tá»‘i Ä‘a 30 thuá»™c tÃ­nh sá»‘ cÃ³ Ã­t giÃ¡ trá»‹ thiáº¿u nháº¥t Ä‘á»ƒ váº½ heatmap
miss_rate = train[num_cols].isnull().mean()
top_num_cols = miss_rate.sort_values().head(30).index.tolist()

sample_corr = train[top_num_cols].sample(
    min(10_000, len(train)), random_state=SEED)

# DÃ¹ng Spearman vÃ¬ 399/400 cá»™t khÃ´ng phÃ¢n phá»‘i chuáº©n (xem káº¿t quáº£ D'Agostino-Pearson)
spearman_corr = sample_corr.corr(method='spearman')
pearson_corr = sample_corr.corr(method='pearson')

# Váº½ Cáº¢ HAI Pearson vÃ  Spearman (Ä‘á» yÃªu cáº§u Â§2.2.2b)
fig, axes = plt.subplots(1, 2, figsize=(28, 14))
mask = np.triu(np.ones_like(spearman_corr, dtype=bool))

sns.heatmap(pearson_corr, mask=mask, cmap='coolwarm', center=0, vmin=-1, vmax=1,
            ax=axes[0], square=True, linewidths=0.3,
            cbar_kws={"shrink": 0.8}, xticklabels=True, yticklabels=True)
axes[0].set_title(
    'TÆ°Æ¡ng quan Pearson â€“ 30 thuá»™c tÃ­nh sá»‘ Ã­t thiáº¿u nháº¥t', fontsize=12)
axes[0].tick_params(axis='x', rotation=90, labelsize=7)
axes[0].tick_params(axis='y', labelsize=7)

sns.heatmap(spearman_corr, mask=mask, cmap='coolwarm', center=0, vmin=-1, vmax=1,
            ax=axes[1], square=True, linewidths=0.3,
            cbar_kws={"shrink": 0.8}, xticklabels=True, yticklabels=True)
axes[1].set_title(
    'TÆ°Æ¡ng quan Spearman â€“ 30 thuá»™c tÃ­nh sá»‘ Ã­t thiáº¿u nháº¥t', fontsize=12)
axes[1].tick_params(axis='x', rotation=90, labelsize=7)
axes[1].tick_params(axis='y', labelsize=7)

plt.suptitle('So sÃ¡nh Pearson vs Spearman Correlation Matrix', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'fig_02_correlation_heatmap.png'),
            dpi=100, bbox_inches='tight')
plt.show()

# So sÃ¡nh Pearson vs Spearman â€” phÃ¢n tÃ­ch Ä‘á»™ng
diff_corr = (pearson_corr - spearman_corr).abs()
max_diff = diff_corr.values[np.tril_indices_from(diff_corr.values, k=-1)].max()
mean_diff = diff_corr.values[np.tril_indices_from(
    diff_corr.values, k=-1)].mean()
print(
    f"Pearson vs Spearman: max|Î”r| = {max_diff:.4f}, mean|Î”r| = {mean_diff:.4f}")
if max_diff > 0.1:
    print("  => CÃ³ Ã­t nháº¥t 1 cáº·p thuá»™c tÃ­nh cÃ³ tÆ°Æ¡ng quan tuyáº¿n tÃ­nh vÃ  Ä‘Æ¡n Ä‘iá»‡u khÃ¡c nhau Ä‘Ã¡ng ká»ƒ")
    print("  => Spearman Ä‘Æ°á»£c Æ°u tiÃªn vÃ¬ dá»¯ liá»‡u khÃ´ng chuáº©n vÃ  cÃ³ outlier")
else:
    print("  => Pearson vÃ  Spearman nháº¥t quÃ¡n â€” phÃ¢n phá»‘i gáº§n tuyáº¿n tÃ­nh")

# %% _uuid="c7aafb4f-8b90-4b50-9bd7-2a63b480572c" _cell_guid="2a9e1000-4c99-48c3-819e-f1de5d078ff4" jupyter={"outputs_hidden": false}
# PhÃ¡t hiá»‡n Ä‘a cá»™ng tuyáº¿n máº¡nh |r_s| > 0.9 (dÃ¹ng Spearman)
high_corr_pairs = []
for i in range(len(spearman_corr.columns)):
    for j in range(i + 1, len(spearman_corr.columns)):
        r = spearman_corr.iloc[i, j]
        if abs(r) > 0.9:
            high_corr_pairs.append({
                'feature_1': spearman_corr.columns[i],
                'feature_2': spearman_corr.columns[j],
                'spearman_r': round(r, 4)
            })

high_corr_df = pd.DataFrame(high_corr_pairs)
print(f"Sá»‘ cáº·p thuá»™c tÃ­nh cÃ³ |Spearman r| > 0.9: {len(high_corr_df)}")
if len(high_corr_df) > 0:
    print(high_corr_df.to_string(index=False))
    print("\n-> Äá» xuáº¥t: loáº¡i bá» má»™t trong má»—i cáº·p Ä‘a cá»™ng tuyáº¿n máº¡nh khi huáº¥n luyá»‡n mÃ´ hÃ¬nh.")
else:
    print("-> KhÃ´ng phÃ¡t hiá»‡n Ä‘a cá»™ng tuyáº¿n máº¡nh trong 30 thuá»™c tÃ­nh kháº£o sÃ¡t.")

# %% [markdown] _uuid="33213665-284a-498d-a77a-c582660b47dc" _cell_guid="57acf54e-ea12-4ec0-bf11-5fdd2cac3e94" jupyter={"outputs_hidden": false}
# ### c) PhÃ¢n tÃ­ch giÃ¡ trá»‹ thiáº¿u â€“ missingno + Little's MCAR test
# #
# CÃ³ ba cÆ¡ cháº¿ thiáº¿u dá»¯ liá»‡u cáº§n phÃ¢n biá»‡t Ä‘á»ƒ chá»n chiáº¿n lÆ°á»£c xá»­ lÃ½ phÃ¹ há»£p:
# #
# | CÆ¡ cháº¿ | Ã nghÄ©a | Chiáº¿n lÆ°á»£c khuyáº¿n nghá»‹ |
# |---|---|---|
# | **MCAR** â€“ Missing Completely At Random | XÃ¡c suáº¥t thiáº¿u khÃ´ng phá»¥ thuá»™c vÃ o báº¥t ká»³ biáº¿n nÃ o | Mean/Median imputation lÃ  an toÃ n |
# | **MAR** â€“ Missing At Random | XÃ¡c suáº¥t thiáº¿u phá»¥ thuá»™c vÃ o biáº¿n khÃ¡c Ä‘Ã£ quan sÃ¡t Ä‘Æ°á»£c | kNN, MICE (khai thÃ¡c cáº¥u trÃºc) |
# | **MNAR** â€“ Missing Not At Random | XÃ¡c suáº¥t thiáº¿u phá»¥ thuá»™c vÃ o chÃ­nh giÃ¡ trá»‹ bá»‹ thiáº¿u | Cáº§n domain knowledge; khÃ³ xá»­ lÃ½ |
# #
# **Little's MCAR test** kiá»ƒm Ä‘á»‹nh giáº£ thuyáº¿t MCAR báº±ng thá»‘ng kÃª $\chi^2$:
# so sÃ¡nh mean cá»§a tá»«ng *missing pattern* (nhÃ³m hÃ ng cÃ³ cÃ¹ng vá»‹ trÃ­ thiáº¿u) vá»›i grand mean.
# #
# $$\chi^2 = \sum_{g} n_g \sum_{j \in O_g} \frac{(\bar{x}_{gj} - \bar{x}_j)^2}{\hat{\sigma}_j^2}$$
# #
# Náº¿u $p < 0.05$ â†’ bÃ¡c bá» MCAR â†’ dá»¯ liá»‡u cÃ³ kháº£ nÄƒng lÃ  MAR hoáº·c MNAR â†’ nÃªn dÃ¹ng kNN/MICE.

# %% [markdown] _uuid="12ca3124-6c34-44be-855e-eac174829633" _cell_guid="2635efcd-e6fa-4d75-b881-8375aa207437" jupyter={"outputs_hidden": false}
# #### Cell 3 â€“ Tá»‰ lá»‡ thiáº¿u theo nhÃ³m cá»™t

# %% _uuid="9d3df8d8-5442-494a-8384-a03680fb28e9" _cell_guid="19999e4d-be7f-447c-be5b-6c227fbbcbe9" jupyter={"outputs_hidden": false}
# Tá»‰ lá»‡ thiáº¿u trung bÃ¬nh theo nhÃ³m
group_missing = {}
for grp, cols in groups.items():
    exist = [c for c in cols if c in train.columns]
    if exist:
        group_missing[grp] = train[exist].isnull().mean().mean() * 100

miss_series = pd.Series(group_missing).sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(10, 5))
colors = ['tomato' if v > 50 else 'orange' if v >
          20 else 'steelblue' for v in miss_series.values]
miss_series.plot(kind='bar', ax=ax, color=colors, edgecolor='white')
ax.axhline(50, color='red',    linestyle='--', alpha=0.6, label='50%')
ax.axhline(20, color='orange', linestyle='--', alpha=0.6, label='20%')
ax.set_ylabel('% thiáº¿u trung bÃ¬nh')
ax.set_title('Tá»‰ lá»‡ thiáº¿u trung bÃ¬nh theo nhÃ³m cá»™t')
ax.legend()
ax.tick_params(axis='x', rotation=20)
for i, v in enumerate(miss_series.values):
    ax.text(i, v + 0.5, f'{v:.1f}%', ha='center', fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'eda_02_missing_by_group.png'),
            dpi=100, bbox_inches='tight')
plt.show()

# %% [markdown] _uuid="442e3df9-2073-493e-a02e-456c8aa7aa0e" _cell_guid="191b740e-f427-46e8-9c33-c82237888346" jupyter={"outputs_hidden": false}
# #### Nháº­n xÃ©t â€“ Tá»‰ lá»‡ thiáº¿u theo nhÃ³m cá»™t
# #
# | NhÃ³m | % thiáº¿u TB | Má»©c Ä‘á»™ | Há»‡ quáº£ |
# |---|---|---|---|
# | **Identity** | 84.5% | ðŸ”´ NghiÃªm trá»ng | ~40% transaction khÃ´ng cÃ³ identity record |
# | **D (timedelta)** | 58.2% | ðŸ”´ NghiÃªm trá»ng | D1â€“D15 thiáº¿u khÃ´ng Ä‘á»“ng Ä‘á»u, cáº§n xá»­ lÃ½ riÃªng tá»«ng cá»™t |
# | **M (match)** | 49.9% | ðŸŸ  Cao | M1â€“M9 gáº§n 50% thiáº¿u; chá»‰ bÃ¡o thiáº¿u cÃ³ thá»ƒ lÃ  feature |
# | **Email** | 46.4% | ðŸŸ  Cao | Thiáº¿u email thÆ°á»ng cÃ³ thá»ƒ lÃ  giao dá»‹ch khÃ´ng Ä‘Äƒng nháº­p |
# | **Address/Dist** | 43.9% | ðŸŸ  Cao | dist1, dist2 thiáº¿u nhiá»u hÆ¡n addr1, addr2 |
# | **V (Vesta)** | 43.0% | ðŸŸ  Cao | Thiáº¿u theo tá»«ng sub-group (G1â€“G11), khÃ´ng ngáº«u nhiÃªn |
# | **Card** | 0.5% | ðŸŸ¢ Tháº¥p | Gáº§n Ä‘áº§y Ä‘á»§; thiáº¿u Ã­t á»Ÿ card4/card6 |
# | **Transaction** | 0.0% | ðŸŸ¢ Äáº§y Ä‘á»§ | Ba cá»™t cá»‘t lÃµi luÃ´n hiá»‡n diá»‡n |
# | **C (count)** | 0.0% | ðŸŸ¢ Äáº§y Ä‘á»§ | C1â€“C14 khÃ´ng cÃ³ giÃ¡ trá»‹ thiáº¿u |
# #
# **Nháº­n xÃ©t chÃ­nh:**
# #
# - **Identity** thiáº¿u 84.5% khÃ´ng pháº£i do lá»—i thu tháº­p mÃ  vÃ¬ Ä‘Ã¢y lÃ  **left join**:
#   chá»‰ ~60% transaction Ä‘i kÃ¨m identity record. Viá»‡c thiáº¿u nÃ y mang Ã½ nghÄ©a thá»±c sá»±
#   (giao dá»‹ch khÃ´ng thiáº¿t bá»‹ = khÃ³ theo dÃµi = tÄƒng nguy cÆ¡ fraud) â†’ nÃªn táº¡o feature `has_identity`.
# #
# - **D (timedelta)** thiáº¿u 58.2% trung bÃ¬nh nhÆ°ng **khÃ´ng Ä‘á»“ng Ä‘á»u**: D1 thiáº¿u Ã­t, D8â€“D15
#   thiáº¿u ráº¥t nhiá»u. Má»—i cá»™t D Ä‘o khoáº£ng cÃ¡ch thá»i gian khÃ¡c nhau, thiáº¿u = sá»± kiá»‡n Ä‘Ã³ chÆ°a
#   tá»«ng xáº£y ra trÆ°á»›c Ä‘Ã³ â†’ **khÃ´ng nÃªn impute báº±ng median**, nÃªn dÃ¹ng -1 hoáº·c táº¡o flag.
# #
# - **M (match)** vÃ  **Email** thiáº¿u ~46â€“50%: cÃ³ thá»ƒ do giao dá»‹ch khÃ¡ch vÃ£ng lai (guest checkout)
#   khÃ´ng cáº§n xÃ¡c thá»±c email hay Ä‘á»‹a chá»‰ â†’ missing cÃ³ thá»ƒ lÃ  **MNAR** (thiáº¿u phá»¥ thuá»™c vÃ o loáº¡i
#   giao dá»‹ch), cáº§n cáº©n tháº­n vá»›i imputation.
# #
# - **C (count)** vÃ  **Transaction** Ä‘áº§y Ä‘á»§ 100% â†’ Ä‘Ã¢y lÃ  nhÃ³m feature **Ä‘Ã¡ng tin cáº­y nháº¥t**
#   cho model, khÃ´ng cáº§n xá»­ lÃ½ thiáº¿u.

# %% [markdown] _uuid="3132a202-2e03-4d6d-95a0-fb7c75c86de2" _cell_guid="a52b03e2-8b47-42f1-a2d0-b44fc1cba8ce" jupyter={"outputs_hidden": false}
# #### Cell 4 â€“ Heatmap missing pattern cá»§a V1â€“V339 (xÃ¡c nháº­n 11 nhÃ³m)

# %% _uuid="3f0d3e62-a1cd-48c9-a5c6-618751f52c8c" _cell_guid="16f52937-da5a-4820-b6a7-65d8dce88e87" jupyter={"outputs_hidden": false}
v_cols = [c for c in train.columns if c.startswith('V')]
# Tá»‰ lá»‡ thiáº¿u tá»«ng cá»™t V
v_missing = train[v_cols].isnull().mean() * 100

fig, ax = plt.subplots(figsize=(20, 3))
ax.bar(range(len(v_cols)), v_missing.values,
       color='steelblue', alpha=0.8, width=1.0)
ax.set_xlabel('V columns (V1 -> V339)')
ax.set_ylabel('% thiáº¿u')
ax.set_title('Tá»‰ lá»‡ thiáº¿u tá»«ng cá»™t V â€“ nhÃ³m theo missing pattern')

# ÄÃ¡nh dáº¥u ranh giá»›i 11 nhÃ³m theo Kaggle
group_boundaries = [11, 34, 52, 74, 94, 137, 166, 216, 278, 321, 339]
group_labels = [f'G{i+1}' for i in range(11)]
prev = 0
for b, lbl in zip(group_boundaries, group_labels):
    ax.axvline(b, color='red', linestyle='--', alpha=0.5, linewidth=0.8)
    ax.text((prev + b) / 2, v_missing.max() * 0.85, lbl,
            ha='center', fontsize=8, color='red')
    prev = b

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'eda_03_v_missing_pattern.png'),
            dpi=100, bbox_inches='tight')
plt.show()

# %% _uuid="ecfa2747-c7bb-4850-8189-e38130707972" _cell_guid="7e8e58b1-be7f-4b58-8991-fc0f5458fbb6" jupyter={"outputs_hidden": false}

# Ma tráº­n thiáº¿u dá»¯ liá»‡u (láº¥y máº«u 500 dÃ²ng Ä‘á»ƒ trá»±c quan)
print("=== PhÃ¢n tÃ­ch giÃ¡ trá»‹ thiáº¿u ===")
missing_pct = train.isnull().mean().sort_values(ascending=False)
print(f"Sá»‘ cá»™t cÃ³ giÃ¡ trá»‹ thiáº¿u: {(missing_pct > 0).sum()}")
print(f"Sá»‘ cá»™t cÃ³ >50% thiáº¿u   : {(missing_pct > 0.5).sum()}")
print(f"Sá»‘ cá»™t cÃ³ >80% thiáº¿u   : {(missing_pct > 0.8).sum()}")
print(f"Sá»‘ cá»™t cÃ³ >90% thiáº¿u   : {(missing_pct > 0.9).sum()}")

# %% _uuid="906ec339-2429-47b2-b2cc-7e064fa4dc7e" _cell_guid="ca6fb7c4-d2b7-4745-91aa-142dd574ac6d" jupyter={"outputs_hidden": false}
fig, axes = plt.subplots(1, 2, figsize=(20, 8))

# Chá»‰ hiá»ƒn thá»‹ cá»™t cÃ³ giÃ¡ trá»‹ thiáº¿u, giá»›i háº¡n 60 cá»™t Ä‘áº§u Ä‘á»ƒ plot Ä‘á»c Ä‘Æ°á»£c
miss_cols = [c for c in train.columns if train[c].isnull().any()][:60]
sample_miss = train[miss_cols].sample(500, random_state=SEED)
msno.matrix(sample_miss, ax=axes[0], fontsize=7, sparkline=False)
axes[0].set_title(
    f'Ma tráº­n giÃ¡ trá»‹ thiáº¿u (500 dÃ²ng máº«u, {len(miss_cols)} cá»™t cÃ³ NaN)', fontsize=11)

# Bar: top 40 cá»™t thiáº¿u nhiá»u nháº¥t
sample_miss_bar = train[miss_cols[:40]].sample(500, random_state=SEED)
msno.bar(sample_miss_bar, ax=axes[1], fontsize=7, color='steelblue')
axes[1].set_title(
    'Tá»‰ lá»‡ dá»¯ liá»‡u cÃ³ máº·t (top 40 cá»™t thiáº¿u)', fontsize=11)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'fig_03_missing_matrix.png'),
            dpi=100, bbox_inches='tight')
plt.show()

# %% _uuid="79d213bd-3f0c-4197-9f32-d30dcb68710d" _cell_guid="2e15a4ad-82d2-4c02-95c8-49b314303222" jupyter={"outputs_hidden": false}
# Biá»ƒu Ä‘á»“ phÃ¢n phá»‘i tá»‰ lá»‡ thiáº¿u
fig, ax = plt.subplots(figsize=(14, 5))
missing_pct[missing_pct > 0].plot(kind='bar', ax=ax, color='tomato', alpha=0.8)
ax.axhline(0.05, color='green', linestyle='--', label='5% ngÆ°á»¡ng')
ax.axhline(0.5,  color='orange', linestyle='--', label='50% ngÆ°á»¡ng')
ax.axhline(0.9,  color='red',    linestyle='--', label='90% ngÆ°á»¡ng')
ax.set_xlabel('Cá»™t')
ax.set_ylabel('Tá»‰ lá»‡ thiáº¿u')
ax.set_title('Tá»‰ lá»‡ giÃ¡ trá»‹ thiáº¿u theo tá»«ng cá»™t')
ax.legend()
ax.tick_params(axis='x', rotation=90, labelsize=6)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'fig_04_missing_bar.png'),
            dpi=100, bbox_inches='tight')
plt.show()


# %% _uuid="2aae2ec1-d032-48ca-8ec8-00941ebe65be" _cell_guid="d8357571-da90-4ded-9b51-c641dda22fc2" jupyter={"outputs_hidden": false}
# â”€â”€ Little's MCAR test (gáº§n Ä‘Ãºng báº±ng chi-square trÃªn táº­p con) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# âš ï¸  LÆ°u Ã½ thiáº¿t káº¿: pháº£i dÃ¹ng cÃ¡c cá»™t CÃ“ missing thá»±c sá»± (5â€“95%).
#    Náº¿u dÃ¹ng cá»™t Ã­t missing nháº¥t (nhÆ° C count, card), háº§u háº¿t row cÃ³ cÃ¹ng
#    1 pattern (khÃ´ng thiáº¿u gÃ¬) â†’ chá»‰ 1 nhÃ³m â†’ chi2=0, p=1 (káº¿t quáº£ suy biáº¿n).
def littles_mcar_approx(df, n_sample=5000, random_state=42):
    """
    Gáº§n Ä‘Ãºng Little's MCAR test báº±ng kiá»ƒm Ä‘á»‹nh chi-square.
    Input: df chá»‰ gá»“m cÃ¡c cá»™t sá»‘ cÃ³ tá»‰ lá»‡ thiáº¿u trong khoáº£ng (5%, 95%).
    - Má»—i missing pattern lÃ  má»™t nhÃ³m; so sÃ¡nh mean nhÃ³m vá»›i grand mean.
    - p > 0.05 â†’ khÃ´ng bÃ¡c bá» MCAR.
    - Tráº£ vá» (chi2_stat, p_value, n_patterns): n_patterns < 2 â†’ test vÃ´ nghÄ©a.
    """
    df = df.sample(min(n_sample, len(df)), random_state=random_state)
    df = df.dropna(axis=1, how='all')

    r = df.isnull().astype(int)
    patterns = r.apply(lambda row: tuple(row), axis=1)
    grp_obj = df.groupby(patterns)
    n_patterns = grp_obj.ngroups

    chi2 = 0.0
    df_deg = 0
    grand_means = df.mean()
    grand_vars = df.var()

    for _, grp in grp_obj:
        n_g = len(grp)
        if n_g < 2:
            continue
        obs_cols = grp.columns[grp.notna().all()].tolist()
        for col in obs_cols:
            gv = grand_vars[col]
            if gv == 0 or np.isnan(gv):
                continue
            chi2 += n_g * (grp[col].mean() - grand_means[col]) ** 2 / gv
            df_deg += 1

    if df_deg == 0:
        return np.nan, np.nan, n_patterns
    return round(chi2, 4), round(1 - stats.chi2.cdf(chi2, df=df_deg), 6), n_patterns


# â”€â”€ Chá»n cá»™t cÃ³ missing thá»±c sá»± (5%â€“95%) trong tá»«ng nhÃ³m quan tÃ¢m â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Dá»±a vÃ o phÃ¢n tÃ­ch nhÃ³m cá»™t: D, M, addr/dist, id_ cÃ³ missing cao vÃ  cÃ³ Ã½ nghÄ©a
num_miss_cols = [c for c in train.select_dtypes(include=[np.number]).columns
                 if 0.05 < train[c].isnull().mean() < 0.95]

# Láº¥y Ä‘áº¡i diá»‡n má»—i nhÃ³m (tá»‘i Ä‘a 5 cá»™t/nhÃ³m) Ä‘á»ƒ trÃ¡nh quÃ¡ nhiá»u pattern
test_groups = {
    'D (timedelta)': [c for c in [f'D{i}' for i in range(1, 16)] if c in num_miss_cols][:5],
    'V (Vesta)': [c for c in train.columns if c.startswith('V') and c in num_miss_cols][:5],
    'id_ (numeric)': [c for c in train.columns if c.startswith('id_') and c in num_miss_cols][:5],
    'addr/dist': [c for c in ['addr1', 'addr2', 'dist1', 'dist2'] if c in num_miss_cols],
}

_title = "LITTLE'S MCAR TEST THEO Tá»ªNG NHÃ“M Cá»˜T"
print("=" * 65)
print(f"{_title:^65}")
print("=" * 65)
print(f"{'NhÃ³m':<18} {'Cá»™t test':<6} {'Patterns':<10} {'Chi2':>12} {'p-value':>10}  {'Káº¿t luáº­n'}")
print("-" * 65)

for grp_name, cols in test_groups.items():
    if not cols:
        print(
            f"{grp_name:<18} {'â€“':^6}  {'â€“':^9}  {'â€“':>12}  {'â€“':>10}  KhÃ´ng Ä‘á»§ cá»™t")
        continue
    chi2_s, p_s, n_pat = littles_mcar_approx(train[cols])
    if np.isnan(chi2_s):
        verdict = "Suy biáº¿n (1 pattern)"
    elif n_pat < 2:
        verdict = "Suy biáº¿n (1 pattern)"
    elif p_s > 0.05:
        verdict = "KhÃ´ng bÃ¡c bá» MCAR"
    else:
        verdict = "âŒ BÃ¡c bá» MCAR â†’ MAR/MNAR"
    print(f"{grp_name:<18} {len(cols):<6}  {n_pat:<9}  {str(chi2_s):>12}  {str(p_s):>10}  {verdict}")

print("=" * 65)
print("\nâš ï¸  LÆ°u Ã½: Little's MCAR test chá»‰ há»¯u Ã­ch khi cÃ³ â‰¥ 2 missing pattern.")
print("   Káº¿t quáº£ suy biáº¿n (1 pattern) = táº¥t cáº£ row cÃ¹ng 1 cáº¥u trÃºc thiáº¿u")
print("   â†’ cáº§n phÃ¢n tÃ­ch tÆ°Æ¡ng quan missing indicator vs target (cell dÆ°á»›i).")

# %% _uuid="31904c12-1a79-417e-8569-d4e161aac79d" _cell_guid="68c16a82-db0c-4e01-9c9d-f4ad3b2e52ea" jupyter={"outputs_hidden": false}
# â”€â”€ PhÃ¢n loáº¡i cÆ¡ cháº¿ thiáº¿u: tÆ°Æ¡ng quan missing indicator vs isFraud â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# ÄÃ¢y lÃ  kiá»ƒm tra trá»±c tiáº¿p hÆ¡n: náº¿u corr(is_missing, isFraud) cao
# â†’ viá»‡c thiáº¿u liÃªn quan Ä‘áº¿n outcome â†’ MNAR hoáº·c MAR informative
print("\n=== TÆ°Æ¡ng quan Missing Indicator vs isFraud ===")
print("(Cháº¡y trÃªn toÃ n bá»™ cá»™t sá»‘ cÃ³ missing, khÃ´ng giá»›i háº¡n top 50)\n")

miss_indicator_corr = {}
for col in train.select_dtypes(include=[np.number]).columns:
    if col in ('isFraud', 'TransactionID'):
        continue
    miss_rate_col = train[col].isnull().mean()
    if 0.01 < miss_rate_col < 0.99:          # chá»‰ cá»™t cÃ³ missing thá»±c sá»±
        indicator = train[col].isnull().astype(int)
        miss_indicator_corr[col] = round(indicator.corr(train['isFraud']), 4)

miss_indicator_df = pd.DataFrame.from_dict(
    miss_indicator_corr, orient='index', columns=['corr_with_isFraud']
).sort_values('corr_with_isFraud', key=abs, ascending=False)

n_high = (miss_indicator_df['corr_with_isFraud'].abs() > 0.05).sum()
print(
    f"Sá»‘ cá»™t cÃ³ |corr(missing, isFraud)| > 0.05 : {n_high}/{len(miss_indicator_df)}")
print(f"\nTop 15 cá»™t cÃ³ missing indicator tÆ°Æ¡ng quan máº¡nh nháº¥t vá»›i isFraud:")
print(miss_indicator_df.head(15).to_string())

# Káº¿t luáº­n tá»•ng há»£p
print("\n" + "=" * 65)
if n_high > len(miss_indicator_df) * 0.3:
    print("â†’ Káº¾T LUáº¬N: Pháº§n lá»›n missing CÃ“ tÆ°Æ¡ng quan vá»›i isFraud")
    print("  â†’ CÆ¡ cháº¿ thiáº¿u lÃ  MAR hoáº·c MNAR (khÃ´ng pháº£i MCAR).")
    print("  â†’ Chiáº¿n lÆ°á»£c xá»­ lÃ½:")
    print("     â€¢ Táº¡o binary flag '_was_missing' cho D, M, Identity, addr/dist")
    print("     â€¢ DÃ¹ng Median impute (khÃ´ng dÃ¹ng Mean â€“ dá»¯ liá»‡u lá»‡ch)")
    print("     â€¢ KHÃ”NG bá» cá»™t chá»‰ vÃ¬ thiáº¿u nhiá»u â€“ missing báº£n thÃ¢n lÃ  tÃ­n hiá»‡u")
else:
    print("â†’ Káº¾T LUáº¬N: Missing Ã­t tÆ°Æ¡ng quan vá»›i isFraud â†’ gáº§n MCAR hÆ¡n.")
    print("  â†’ CÃ³ thá»ƒ dÃ¹ng Median/Mode impute an toÃ n.")
print("=" * 65)

# %% [markdown] _uuid="403b8620-c266-4906-be6e-fea5ece803ee" _cell_guid="025d6087-490e-41ba-bb64-94d7d79973a5" jupyter={"outputs_hidden": false}
# #### Cell 10 â€“ D1â€“D15: phÃ¢n phá»‘i vÃ  tÆ°Æ¡ng quan missing vá»›i Fraud

# %% _uuid="abcd5b0b-1ae4-4f52-9479-b3bb555ccf6f" _cell_guid="b4a95900-7f3f-4db2-b7c0-65c13f1a57df" jupyter={"outputs_hidden": false}
d_cols = [f'D{i}' for i in range(1, 16) if f'D{i}' in train.columns]

# Tá»‰ lá»‡ thiáº¿u
d_missing = train[d_cols].isnull().mean() * 100

# TÆ°Æ¡ng quan giá»¯a missing indicator vÃ  isFraud
d_miss_corr = {}
for col in d_cols:
    if train[col].isnull().any():
        indicator = train[col].isnull().astype(int)
        d_miss_corr[col] = indicator.corr(train['isFraud'])

fig, axes = plt.subplots(1, 2, figsize=(16, 5))

d_missing.plot(
    kind='bar', ax=axes[0], color='steelblue', alpha=0.8, edgecolor='white')
axes[0].set_title('Tá»‰ lá»‡ thiáº¿u D1â€“D15 (%)')
axes[0].set_ylabel('% thiáº¿u')
axes[0].tick_params(axis='x', rotation=30)
for i, v in enumerate(d_missing.values):
    axes[0].text(i, v + 0.5, f'{v:.0f}%', ha='center', fontsize=8)

corr_series = pd.Series(d_miss_corr).sort_values(key=abs, ascending=False)
corr_series.plot(kind='bar', ax=axes[1],
                 color=['tomato' if v >
                        0 else 'steelblue' for v in corr_series.values],
                 alpha=0.8, edgecolor='white')
axes[1].axhline(0, color='black', linestyle='--', alpha=0.4)
axes[1].set_title(
    'TÆ°Æ¡ng quan (missing indicator vs isFraud)\n>0: thiáº¿u -> kháº£ nÄƒng Fraud cao hÆ¡n')
axes[1].set_ylabel('Pearson r')
axes[1].tick_params(axis='x', rotation=30)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'eda_08_d_cols.png'),
            dpi=100, bbox_inches='tight')
plt.show()

# %% [markdown] _uuid="82e84564-d7a4-4126-a93b-9f287ebb89a0" _cell_guid="02bd9923-68a6-46a7-a0fe-658099092390" jupyter={"outputs_hidden": false}
# ### d) PhÃ¢n tÃ­ch feature vs target

# %% [markdown] _uuid="5b5071c3-0fd7-46c9-8969-4eed75307d69" _cell_guid="cd4c07c9-ad37-4c33-b9db-39dcb4b39b0f" jupyter={"outputs_hidden": false}
# #### Cell 6 â€“ TransactionAmt: phÃ¢n phá»‘i Fraud vs Normal

# %% _uuid="ea1173e7-1762-4d45-be63-eacc738e62a7" _cell_guid="bae72298-95b3-41f6-a6b4-cf33c2c731f4" jupyter={"outputs_hidden": false}
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Log scale boxplot
for label, grp in train.groupby('isFraud')['TransactionAmt']:
    axes[0].boxplot(grp.clip(upper=grp.quantile(0.99)),
                    positions=[label], widths=0.4,
                    patch_artist=True,
                    boxprops=dict(facecolor='tomato' if label else 'steelblue', alpha=0.7))
axes[0].set_xticks([0, 1])
axes[0].set_xticklabels(['Normal', 'Fraud'])
axes[0].set_title('TransactionAmt â€“ Boxplot (clip 99th pct)')
axes[0].set_ylabel('Amount')

# Histogram log-scale
for label, color in [(0, 'steelblue'), (1, 'tomato')]:
    data = train[train['isFraud'] == label]['TransactionAmt'].clip(upper=5000)
    axes[1].hist(data, bins=80, alpha=0.5, color=color,
                 label=f'{"Fraud" if label else "Normal"}', density=True)
axes[1].set_title('TransactionAmt â€“ Distribution (clip $5000)')
axes[1].set_xlabel('Amount')
axes[1].legend()

plt.suptitle('TransactionAmt: Fraud vs Normal', fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'eda_04_transamt.png'),
            dpi=100, bbox_inches='tight')
plt.show()

print("Thá»‘ng kÃª TransactionAmt theo lá»›p:")
print(train.groupby('isFraud')[
      'TransactionAmt'].describe().round(2).to_string())

# %% [markdown] _uuid="c43f5e4e-d37a-41f0-8678-3e0c1d7ed06d" _cell_guid="1c6f803e-78cf-4766-82fe-643a25636f96" jupyter={"outputs_hidden": false}
# #### Cell 7 â€“ Categorical features vs Fraud: ProductCD, card4, card6, M1â€“M9

# %% _uuid="ba840683-3e4f-47b5-8869-d3cdd23bf3ef" _cell_guid="c9e2895c-d961-4e9c-bc67-3050f663c1cc" jupyter={"outputs_hidden": false}
cat_fraud_cols = ['ProductCD', 'card4', 'card6'] + \
    [f'M{i}' for i in range(1, 10)]
cat_fraud_cols = [c for c in cat_fraud_cols if c in train.columns]

n = len(cat_fraud_cols)
ncols = 4
nrows = (n + ncols - 1) // ncols
fig, axes = plt.subplots(nrows, ncols, figsize=(18, nrows * 4))
axes = axes.flatten()

for i, col in enumerate(cat_fraud_cols):
    fraud_rate = train.groupby(
        col)['isFraud'].mean().sort_values(ascending=False)
    counts = train[col].value_counts()
    # Chá»‰ hiá»ƒn thá»‹ top 10 giÃ¡ trá»‹ theo count
    top_vals = counts.head(10).index
    fraud_rate_top = fraud_rate.reindex(top_vals).dropna()
    bars = axes[i].bar(fraud_rate_top.index.astype(str), fraud_rate_top.values * 100,
                       color='tomato', alpha=0.8, edgecolor='white')
    axes[i].set_title(col, fontsize=11)
    axes[i].set_ylabel('Fraud rate (%)')
    axes[i].tick_params(axis='x', rotation=30, labelsize=8)
    for bar in bars:
        h = bar.get_height()
        axes[i].text(bar.get_x() + bar.get_width()/2, h + 0.2,
                     f'{h:.1f}%', ha='center', fontsize=7)

for j in range(i+1, len(axes)):
    axes[j].set_visible(False)

plt.suptitle('Fraud rate theo categorical features', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'eda_05_cat_fraud_rate.png'),
            dpi=100, bbox_inches='tight')
plt.show()

# %% [markdown] _uuid="13d1c698-8eaf-4288-b704-b90c427bb0d2" _cell_guid="cc251dae-0590-4e3d-b775-54692925e590" jupyter={"outputs_hidden": false}
# #### Cell 8 â€“ Email domain vs Fraud

# %% _uuid="d2972b89-e4d0-48e5-85cc-52d5d5cd5a20" _cell_guid="1a9caca0-3989-4910-ba7b-ac4796a76146" jupyter={"outputs_hidden": false}
fig, axes = plt.subplots(1, 2, figsize=(16, 5))

for ax, col in zip(axes, ['P_emaildomain', 'R_emaildomain']):
    top_domains = train[col].value_counts().head(10).index
    fraud_rate = (train[train[col].isin(top_domains)]
                  .groupby(col)['isFraud'].mean()
                  .reindex(top_domains)
                  .sort_values(ascending=False))
    ax.barh(fraud_rate.index.astype(str), fraud_rate.values * 100,
            color='tomato', alpha=0.8, edgecolor='white')
    ax.set_xlabel('Fraud rate (%)')
    ax.set_title(f'{col} â€“ Top 10 domains')
    for j, v in enumerate(fraud_rate.values):
        ax.text(v + 0.1, j, f'{v*100:.1f}%', va='center', fontsize=8)

# Feature: email match
train['email_match'] = (train['P_emaildomain'] ==
                        train['R_emaildomain']).astype(int)
match_rate = train.groupby('email_match')['isFraud'].mean()
print("Fraud rate â€“ P_email == R_email:")
print(f"  Email khÃ¡c nhau (0): {match_rate.get(0, 0)*100:.2f}%")
print(f"  Email giá»‘ng nhau (1): {match_rate.get(1, 0)*100:.2f}%")
train.drop(columns=['email_match'], inplace=True)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'eda_06_email_fraud.png'),
            dpi=100, bbox_inches='tight')
plt.show()

# %% [markdown] _uuid="ac4a7988-a3de-4ccd-ae94-d4df311131f3" _cell_guid="55ab95d5-8869-4b68-abe3-6bfb76ca1025" jupyter={"outputs_hidden": false}
# #### Cell 9 â€“ C1â€“C14: so sÃ¡nh mean/median giá»¯a Fraud vs Normal

# %% _uuid="083d8e83-f80b-4632-8f88-583357ff0db3" _cell_guid="0c26d288-2d8d-4a1f-9a21-9b3937ff5bb9" jupyter={"outputs_hidden": false}
c_cols = [f'C{i}' for i in range(1, 15) if f'C{i}' in train.columns]

stats_c = train.groupby('isFraud')[c_cols].median()
ratio_c = (stats_c.loc[1] / stats_c.loc[0]).sort_values(ascending=False)

fig, axes = plt.subplots(1, 2, figsize=(16, 5))

stats_c.T.plot(kind='bar', ax=axes[0], color=['steelblue', 'tomato'],
               edgecolor='white', alpha=0.8)
axes[0].set_title('Median C1â€“C14: Normal vs Fraud')
axes[0].set_ylabel('Median value')
axes[0].tick_params(axis='x', rotation=30)
axes[0].legend(['Normal', 'Fraud'])

ratio_c.plot(kind='bar', ax=axes[1],
             color=['tomato' if v > 1 else 'steelblue' for v in ratio_c.values],
             edgecolor='white', alpha=0.8)
axes[1].axhline(1, color='black', linestyle='--', alpha=0.5)
axes[1].set_title(
    'Fraud/Normal median ratio (C columns)\n>1: Fraud cÃ³ giÃ¡ trá»‹ cao hÆ¡n')
axes[1].set_ylabel('Ratio Fraud / Normal')
axes[1].tick_params(axis='x', rotation=30)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'eda_07_c_cols.png'),
            dpi=100, bbox_inches='tight')
plt.show()
print("Fraud/Normal median ratio:\n", ratio_c.round(3).to_string())

# %% [markdown] _uuid="8055cb3c-3df6-4a0d-8d1a-73c30aec89df" _cell_guid="744f2509-fa57-4cf5-b388-22db10977d5a" jupyter={"outputs_hidden": false}
# #### Cell 11 â€“ Identity: id_01, id_02 vÃ  tÃ¡c Ä‘á»™ng cá»§a viá»‡c CÃ“/KHÃ”NG cÃ³ identity

# %% _uuid="11f21451-4909-4e08-837e-6ff636c689fe" _cell_guid="fda2ddeb-2752-462c-938c-f88e3b93fed3" jupyter={"outputs_hidden": false}
# TÃ¡c Ä‘á»™ng cá»§a viá»‡c cÃ³ identity record
train['has_identity'] = (~train['id_01'].isnull()).astype(int)
id_fraud_rate = train.groupby('has_identity')['isFraud'].mean()
print("Fraud rate theo cÃ³/khÃ´ng cÃ³ identity record:")
print(f"  KhÃ´ng cÃ³ identity (0): {id_fraud_rate.get(0, 0)*100:.2f}%")
print(f"  CÃ³ identity     (1): {id_fraud_rate.get(1, 0)*100:.2f}%")
train.drop(columns=['has_identity'], inplace=True)

# id_01, id_02 distribution
id_num_cols = [c for c in ['id_01', 'id_02', 'id_03', 'id_04', 'id_05', 'id_06']
               if c in train.columns]

fig, axes = plt.subplots(2, 3, figsize=(16, 8))
axes = axes.flatten()
for i, col in enumerate(id_num_cols):
    for label, color in [(0, 'steelblue'), (1, 'tomato')]:
        data = train[train['isFraud'] == label][col].dropna()
        axes[i].hist(data, bins=50, alpha=0.5, color=color, density=True,
                     label=f'{"Fraud" if label else "Normal"}')
    axes[i].set_title(col)
    axes[i].legend(fontsize=8)

for j in range(i+1, len(axes)):
    axes[j].set_visible(False)

plt.suptitle('Identity features (id_01â€“id_06): Fraud vs Normal', fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'eda_09_identity.png'),
            dpi=100, bbox_inches='tight')
plt.show()

# %% [markdown] _uuid="e668f8cd-a310-4289-8922-297ab9c75049" _cell_guid="d77160bd-69d1-401d-b9a9-2e1734a6d579" jupyter={"outputs_hidden": false}
# #### Cell 12 â€“ TÃ³m táº¯t: báº£ng xáº¿p háº¡ng feature theo má»©c phÃ¢n biá»‡t Fraud

# %% _uuid="7a8e1e37-45c5-479a-9ede-5260586429a3" _cell_guid="b517c006-c871-4909-9506-19bb830be7ae" jupyter={"outputs_hidden": false}

# Chá»n má»™t táº­p features Ä‘áº¡i diá»‡n Ä‘á»ƒ tÃ­nh MI score
summary_cols = (
    ['TransactionAmt', 'TransactionDT']
    + [f'card{i}' for i in range(1, 7) if f'card{i}' in train.columns]
    + ['addr1', 'addr2', 'dist1', 'dist2']
    + [f'C{i}' for i in range(1, 15) if f'C{i}' in train.columns]
    + [f'D{i}' for i in range(1, 16) if f'D{i}' in train.columns]
    + train[[c for c in train.columns if c.startswith('V')]].isnull(
    ).mean().sort_values().head(5).index.tolist()
    + [c for c in ['id_01', 'id_02', 'id_03'] if c in train.columns]
)
summary_cols = [c for c in summary_cols if c in train.columns]

X_sum = train[summary_cols].copy()
# Encode categorical columns thÃ nh sá»‘ nguyÃªn Ä‘á»ƒ mutual_info_classif cháº¥p nháº­n
for col in X_sum.select_dtypes(include='object').columns:
    X_sum[col] = X_sum[col].astype('category').cat.codes
X_sum = X_sum.fillna(-999)
mi_sum = mutual_info_classif(X_sum, train['isFraud'], random_state=SEED)
mi_rank = pd.DataFrame({'feature': summary_cols, 'MI_score': mi_sum})
mi_rank = mi_rank.sort_values(
    'MI_score', ascending=False).reset_index(drop=True)

print("Top 20 features theo Mutual Information:")
print(mi_rank.head(20).to_string(index=False))

fig, ax = plt.subplots(figsize=(10, 8))
mi_rank.head(20).plot(x='feature', y='MI_score', kind='barh',
                      ax=ax, color='steelblue', legend=False)
ax.set_title('Top 20 Features â€“ Mutual Information vs isFraud')
ax.set_xlabel('MI Score')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'eda_10_mi_summary.png'),
            dpi=100, bbox_inches='tight')
plt.show()

# %% [markdown] _uuid="9e0653aa-3a0a-4601-943f-9e8b97644403" _cell_guid="7f47be29-faca-47f4-976a-d0362dd43085" jupyter={"outputs_hidden": false}
# ---
# ## 2.2.3a. Xá»­ lÃ½ giÃ¡ trá»‹ thiáº¿u cÃ³ kiá»ƒm soÃ¡t â€“ So sÃ¡nh 5 chiáº¿n lÆ°á»£c
# #
# ### CÃ¡c chiáº¿n lÆ°á»£c Ä‘iá»n khuyáº¿t
# #
# | Chiáº¿n lÆ°á»£c | CÆ¡ cháº¿ | Äá»™ phá»©c táº¡p | PhÃ¹ há»£p khi |
# |---|---|---|---|
# | **Mean / Median / Mode** | Thay báº±ng thá»‘ng kÃª tÃ³m táº¯t cá»§a cá»™t | $O(n)$ | MCAR, cáº§n tá»‘c Ä‘á»™ |
# | **kNN Imputation** | Äiá»n tá»« $k$ hÃ ng xÃ³m gáº§n nháº¥t (khoáº£ng cÃ¡ch Euclidean trÃªn cÃ¡c cá»™t Ä‘áº§y Ä‘á»§) | $O(n^2 d)$ | MAR, dataset vá»«a |
# | **MICE** (Iterative Imputer) | Há»“i quy vÃ²ng láº·p: má»—i cá»™t thiáº¿u Ä‘Æ°á»£c dá»± Ä‘oÃ¡n tá»« cÃ¡c cá»™t khÃ¡c, láº·p Ä‘áº¿n há»™i tá»¥ | $O(n \cdot d^2 \cdot \text{iter})$ | MAR/MNAR, cáº§n Ä‘á»™ chÃ­nh xÃ¡c cao |
# #
# ### PhÆ°Æ¡ng phÃ¡p Ä‘Ã¡nh giÃ¡ (Benchmark)
# Táº¡o nhÃ¢n táº¡o 10% MCAR trÃªn táº­p hoÃ n chá»‰nh, Ä‘iá»n láº¡i, rá»“i tÃ­nh **RMSE** chá»‰ trÃªn cÃ¡c Ã´ bá»‹ lÃ m thiáº¿u:
# #
# $$\text{RMSE} = \sqrt{\frac{1}{|\mathcal{M}|}\sum_{(i,j)\in\mathcal{M}} (x_{ij} - \hat{x}_{ij})^2}$$
# #
# Chiáº¿n lÆ°á»£c cÃ³ RMSE tháº¥p nháº¥t Ä‘Æ°á»£c chá»n. Tuy nhiÃªn, vá»›i dataset lá»›n (~400 cá»™t Ã— 590k dÃ²ng),
# **Median imputation** Ä‘Æ°á»£c Æ°u tiÃªn cho production vÃ¬ bá»n vá»¯ng vá»›i outlier vÃ  trÃ¡nh OOM.

# %% _uuid="7afee0b4-1555-4f97-adf2-a388db84f0a0" _cell_guid="09ef0eca-9909-4760-a993-ae643f45c954" jupyter={"outputs_hidden": false}
from sklearn.experimental import enable_iterative_imputer  # noqa

# Chá»n má»™t táº­p con nhá» hÆ¡n Ä‘á»ƒ benchmark imputation (cÃ¡c cá»™t sá»‘, Ã­t thiáº¿u <60%)
imp_cols = [c for c in num_cols
            if 0.01 < train[c].isnull().mean() < 0.6][:15]
print(f"Sá»‘ cá»™t dÃ¹ng Ä‘á»ƒ so sÃ¡nh imputation: {len(imp_cols)}")

imp_sample = train[imp_cols].dropna().sample(min(5000, len(train.dropna(subset=imp_cols))),
                                             random_state=SEED)
print(f"DÃ²ng máº«u khÃ´ng thiáº¿u: {len(imp_sample)}")


def benchmark_imputation(df_complete, cols, missing_frac=0.10, seed=42):
    """
    Táº¡o nhÃ¢n táº¡o missing_frac% giÃ¡ trá»‹ thiáº¿u (MCAR),
    Ã¡p dá»¥ng 5 chiáº¿n lÆ°á»£c, tÃ­nh RMSE so vá»›i giÃ¡ trá»‹ gá»‘c.
    """
    rng = np.random.default_rng(seed)
    df_orig = df_complete[cols].copy().reset_index(drop=True)
    df_miss = df_orig.copy()

    # Táº¡o MCAR mask
    mask = rng.random(df_orig.shape) < missing_frac
    df_miss[mask] = np.nan

    results = {}

    strategies = {
        'Mean': SimpleImputer(strategy='mean'),
        'Median': SimpleImputer(strategy='median'),
        'Mode': SimpleImputer(strategy='most_frequent'),
        'kNN-3': KNNImputer(n_neighbors=3),
        'kNN-5': KNNImputer(n_neighbors=5),
        'kNN-10': KNNImputer(n_neighbors=10),
        'MICE': IterativeImputer(max_iter=5, random_state=seed),
    }

    for name, imputer in strategies.items():
        df_imp = pd.DataFrame(imputer.fit_transform(df_miss), columns=cols)
        # TÃ­nh RMSE chá»‰ trÃªn cÃ¡c Ã´ bá»‹ lÃ m thiáº¿u nhÃ¢n táº¡o
        rmse_list = []
        for col in cols:
            col_mask = mask[:, cols.index(col)]
            if col_mask.sum() == 0:
                continue
            rmse = np.sqrt(mean_squared_error(df_orig[col][col_mask],
                                              df_imp[col][col_mask]))
            rmse_list.append(rmse)
        results[name] = round(np.mean(rmse_list), 4) if rmse_list else np.nan

    return results


print("\nÄang cháº¡y benchmark imputation (10% MCAR)...")
imp_results = benchmark_imputation(imp_sample, imp_cols)

imp_compare_df = pd.DataFrame.from_dict(imp_results, orient='index',
                                        columns=['RMSE_trung_bÃ¬nh'])
imp_compare_df = imp_compare_df.sort_values('RMSE_trung_bÃ¬nh')
print("\n=== Báº£ng so sÃ¡nh chiáº¿n lÆ°á»£c Ä‘iá»n khuyáº¿t ===")
print(imp_compare_df.to_string())
best_strategy = imp_compare_df.index[0]
print(
    f"\n-> Chiáº¿n lÆ°á»£c tá»‘t nháº¥t (benchmark): {best_strategy} (RMSE = {imp_compare_df.iloc[0, 0]})")

# Chá»n chiáº¿n lÆ°á»£c scalable tá»‘t nháº¥t cho production (KNN/MICE khÃ´ng phÃ¹ há»£p 590k Ã— 400+ cols)
scalable = {k: v for k, v in imp_results.items() if k in ('Mean',
                                                          'Median', 'Mode')}
best_scalable = min(scalable, key=scalable.get)
print(
    f"-> Chiáº¿n lÆ°á»£c scalable tá»‘t nháº¥t: {best_scalable} (RMSE = {scalable[best_scalable]:.4f})")
print("   (KNN/MICE khÃ´ng Ã¡p dá»¥ng production â€” O(nÂ·d) memory exceed vá»›i 400+ cá»™t Ã— 590k dÃ²ng)")

# Friedman test â€” kiá»ƒm Ä‘á»‹nh sá»± khÃ¡c biá»‡t giá»¯a cÃ¡c chiáº¿n lÆ°á»£c imputation
# (phi tham sá»‘, n máº«u nhá» â€” sá»‘ cá»™t imputed)
if len(imp_results) >= 3:
    from scipy.stats import friedmanchisquare, wilcoxon
    # Má»—i phÆ°Æ¡ng phÃ¡p cÃ³ 1 RMSE scalar â€” cáº§n chuáº©n bá»‹ per-column RMSE
    # (benchmark_imputation Ä‘Ã£ tÃ­nh rmse_list cho má»—i cá»™t â€” reload vá»›i per-col detail)

    def benchmark_imputation_detail(df_orig, cols, n_masked=0.10, seed=42):
        np.random.seed(seed)
        df_miss = df_orig[cols].copy()
        masks = {}
        for col in cols:
            n_m = max(1, int(len(df_orig) * n_masked))
            idx = np.random.choice(df_orig.index, n_m, replace=False)
            df_miss.loc[idx, col] = np.nan
            masks[col] = idx
        strategies = {
            'Mean': SimpleImputer(strategy='mean'),
            'Median': SimpleImputer(strategy='median'),
            'Mode': SimpleImputer(strategy='most_frequent'),
            'kNN-5': KNNImputer(n_neighbors=5),
            'MICE': IterativeImputer(max_iter=5, random_state=seed),
        }
        per_col = {name: [] for name in strategies}
        for name, imputer in strategies.items():
            df_imp = pd.DataFrame(imputer.fit_transform(df_miss), columns=cols)
            for col in cols:
                col_mask = df_orig.index.isin(masks[col])
                rmse = np.sqrt(mean_squared_error(
                    df_orig[col][col_mask], df_imp[col][col_mask]))
                per_col[name].append(rmse)
        return per_col

    print("\nFriedman test â€” kiá»ƒm Ä‘á»‹nh sá»± khÃ¡c biá»‡t giá»¯a cÃ¡c chiáº¿n lÆ°á»£c (per-column RMSE):")
    per_col_rmse = benchmark_imputation_detail(imp_sample, imp_cols)
    strats_ordered = list(per_col_rmse.keys())
    groups_rmse = [np.array(per_col_rmse[s]) for s in strats_ordered]
    try:
        fstat, fp = friedmanchisquare(*groups_rmse)
        print(f"  Friedman: Ï‡Â²={fstat:.3f}, p={fp:.4e}")
        if fp < 0.05:
            print(
                "  => CÃ³ Ã­t nháº¥t 1 chiáº¿n lÆ°á»£c khÃ¡c biá»‡t cÃ³ Ã½ nghÄ©a â€” cháº¡y post-hoc Wilcoxon:")
            n_p = len(strats_ordered) * (len(strats_ordered)-1) // 2
            alpha_b = 0.05 / n_p
            for i in range(len(strats_ordered)):
                for j in range(i+1, len(strats_ordered)):
                    try:
                        _, pw = wilcoxon(groups_rmse[i], groups_rmse[j])
                    except ValueError:
                        pw = 1.0
                    sig = "*" if pw < alpha_b else "ns"
                    print(
                        f"    {strats_ordered[i]:8s} vs {strats_ordered[j]:8s}: p={pw:.4e} {sig} (Bonf Î±={alpha_b:.4f})")
        else:
            print(
                "  => KhÃ´ng cÃ³ sá»± khÃ¡c biá»‡t cÃ³ Ã½ nghÄ©a thá»‘ng kÃª giá»¯a cÃ¡c chiáº¿n lÆ°á»£c")
    except Exception as e:
        print(f"  Friedman test khÃ´ng cháº¡y Ä‘Æ°á»£c: {e}")

# %% _uuid="42623781-a843-4f09-93dc-008dd46c1be3" _cell_guid="d40748ab-ad3e-4e86-a77e-8f4a23441876" jupyter={"outputs_hidden": false}
# Biá»ƒu Ä‘á»“ so sÃ¡nh RMSE
fig, ax = plt.subplots(figsize=(10, 5))
imp_compare_df['RMSE_trung_bÃ¬nh'].plot(kind='barh', ax=ax,
                                        color='steelblue', edgecolor='white')
ax.set_xlabel('RMSE trung bÃ¬nh')
ax.set_title('So sÃ¡nh chiáº¿n lÆ°á»£c Ä‘iá»n khuyáº¿t (10% MCAR nhÃ¢n táº¡o)')
ax.axvline(imp_compare_df['RMSE_trung_bÃ¬nh'].min(), color='red', linestyle='--',
           label=f'Min RMSE = {imp_compare_df["RMSE_trung_bÃ¬nh"].min()}')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'fig_05_imputation_comparison.png'),
            dpi=100, bbox_inches='tight')
plt.show()

# %% [markdown] _uuid="6695f4e6-c616-4504-a455-8ad19629b235" _cell_guid="194804d8-f774-41a6-b91e-385bc19dba46" jupyter={"outputs_hidden": false}
# ### Ãp dá»¥ng chiáº¿n lÆ°á»£c Ä‘iá»n khuyáº¿t lÃªn toÃ n bá»™ dá»¯ liá»‡u
# #
# **LÆ°u Ã½ ká»¹ thuáº­t:** KNNImputer vÃ  MICE chá»‰ phÃ¹ há»£p Ä‘á»ƒ *benchmark* trÃªn táº­p con nhá».
# Vá»›i ~400+ cá»™t sá»‘ vÃ  590k dÃ²ng, Ã¡p dá»¥ng toÃ n bá»™ sáº½ gÃ¢y **out-of-memory**.
# -> DÃ¹ng chiáº¿n lÆ°á»£c scalable tá»‘t nháº¥t (Mean/Median/Mode theo RMSE benchmark) cho production pipeline
#   (KNN/MICE khÃ´ng phÃ¹ há»£p 400+ cá»™t Ã— 590k dÃ²ng â€” OOM)

# %% _uuid="f5454d47-5959-4f32-8e0a-38c45c254e8c" _cell_guid="08fa5613-36ef-4ddc-8787-a5cb673c8d48" jupyter={"outputs_hidden": false}
print(
    f"\nBenchmark tá»‘t nháº¥t: {best_strategy}. Scalable tá»‘t nháº¥t: {best_scalable}.")
print(
    f"Ãp dá»¥ng {best_scalable} imputation cho toÃ n bá»™ dá»¯ liá»‡u (400+ cols Ã— 590k rows)...")

_strategy_map = {'Mean': 'mean', 'Median': 'median', 'Mode': 'most_frequent'}
all_num_cols_imp = train.select_dtypes(include=[np.number]).columns.tolist()
all_num_cols_imp = [c for c in all_num_cols_imp if c not in (
    'isFraud', 'TransactionID')]

# Má»™t sá»‘ cá»™t identity (id_01..id_32) cÃ³ trong train nhÆ°ng KHÃ”NG cÃ³ trong test
# (test_identity cÃ³ thá»ƒ thiáº¿u má»™t sá»‘ cá»™t so vá»›i train_identity)
# -> Chá»‰ impute trÃªn giao cá»§a hai táº­p Ä‘á»ƒ trÃ¡nh KeyError
all_num_cols_imp_test = [c for c in all_num_cols_imp if c in test.columns]
train_only_cols = [c for c in all_num_cols_imp if c not in test.columns]
if train_only_cols:
    print(
        f"  Cá»™t chá»‰ cÃ³ trong train (bá» qua transform test): {train_only_cols}")

# DÃ¹ng best scalable strategy â€” robust vá»›i outlier, trÃ¡nh OOM
prod_imputer = SimpleImputer(strategy=_strategy_map[best_scalable])
train[all_num_cols_imp] = prod_imputer.fit_transform(train[all_num_cols_imp])
# transform test chá»‰ trÃªn cá»™t chung
prod_imputer_test = SimpleImputer(strategy=_strategy_map[best_scalable])
test[all_num_cols_imp_test] = prod_imputer_test.fit_transform(
    test[all_num_cols_imp_test])

# Äiá»n mode cho cá»™t phÃ¢n loáº¡i (object)
cat_cols_all = train.select_dtypes(include=['object']).columns.tolist()
for col in cat_cols_all:
    mode_val = train[col].mode()
    if len(mode_val) > 0:
        train[col] = train[col].fillna(mode_val[0])
        if col in test.columns:   # má»™t sá»‘ cá»™t identity chá»‰ cÃ³ trong train
            test[col] = test[col].fillna(mode_val[0])

print(f"Sau imputation â€“ train NaN cÃ²n láº¡i: {train.isnull().sum().sum()}")
print(f"-> {len(all_num_cols_imp)} cá»™t sá»‘ Ä‘Ã£ Ä‘Æ°á»£c Ä‘iá»n khuyáº¿t báº±ng Median.")
gc.collect()

# %% [markdown] _uuid="601a91af-0802-4d8a-9548-74a06db26225" _cell_guid="b74e3439-5b7e-4007-8bf7-91162848e9b6" jupyter={"outputs_hidden": false}
# ---
# ## 2.2.3b. PhÃ¡t hiá»‡n vÃ  xá»­ lÃ½ ngoáº¡i lai â€“ So sÃ¡nh cÃ¡c phÆ°Æ¡ng phÃ¡p
# #
# | PhÆ°Æ¡ng phÃ¡p | NguyÃªn lÃ½ | Tham sá»‘ | Loáº¡i |
# |---|---|---|---|
# | **IQR** | Ngoáº¡i lai náº¿u $x < Q_1 - 1.5 \cdot IQR$ hoáº·c $x > Q_3 + 1.5 \cdot IQR$ | factor = 1.5 | Univariate |
# | **Z-score** | Ngoáº¡i lai náº¿u $|z| > 3$, vá»›i $z = (x - \mu)/\sigma$ | ngÆ°á»¡ng = 3 | Univariate |
# | **Isolation Forest** | CÃ´ láº­p Ä‘iá»ƒm báº±ng cÃ¢y ngáº«u nhiÃªn; Ä‘iá»ƒm dá»… cÃ´ láº­p (Ä‘Æ°á»ng Ä‘i ngáº¯n) = ngoáº¡i lai | contamination | Multivariate |
# | **LOF** | So sÃ¡nh máº­t Ä‘á»™ cá»¥c bá»™ cá»§a Ä‘iá»ƒm vá»›i $k$ hÃ ng xÃ³m; máº­t Ä‘á»™ tháº¥p hÆ¡n nhiá»u = ngoáº¡i lai | n_neighbors | Multivariate |
# | **DBSCAN** | Äiá»ƒm khÃ´ng thuá»™c cluster nÃ o (label = $-1$) lÃ  ngoáº¡i lai | eps, min\_samples | Multivariate |
# #
# **ÄÃ¡nh giÃ¡ tÃ¡c Ä‘á»™ng** báº±ng **KS test (Kolmogorov-Smirnov)**:
# so sÃ¡nh phÃ¢n phá»‘i trÆ°á»›c vÃ  sau khi loáº¡i ngoáº¡i lai:
# #
# $$D = \sup_x \left| F_1(x) - F_2(x) \right|$$
# #
# Náº¿u $p < 0.05$ â†’ phÃ¢n phá»‘i bá»‹ biáº¿n dáº¡ng Ä‘Ã¡ng ká»ƒ â†’ phÆ°Æ¡ng phÃ¡p Ä‘Ã³ quÃ¡ hung hÄƒng.
# Æ¯u tiÃªn **IQR clipping** (giá»›i háº¡n thay vÃ¬ xÃ³a dÃ²ng) Ä‘á»ƒ báº£o toÃ n sá»‘ lÆ°á»£ng máº«u.

# %% _uuid="8032e199-e56c-483c-825b-3e3aedf95743" _cell_guid="8d884d03-1eb1-4e32-8d37-725cb05a7e53" jupyter={"outputs_hidden": false}

# DÃ¹ng táº­p con nhá» Ä‘á»ƒ benchmark ngoáº¡i lai (sá»‘ cá»™t khÃ´ng thiáº¿u, n dÃ²ng)
outlier_cols = [c for c in num_cols if train[c].isnull().sum() == 0][:20]
outlier_sample = train[outlier_cols].sample(min(10_000, len(train[outlier_cols].dropna())),
                                            random_state=SEED).reset_index(drop=True)
print(
    f"Benchmark ngoáº¡i lai: {outlier_sample.shape[0]} dÃ²ng Ã— {len(outlier_cols)} cá»™t")
# LÃ½ do subsample: LOF vÃ  DBSCAN cÃ³ Ä‘á»™ phá»©c táº¡p O(nÂ²); vá»›i 590k dÃ²ng sáº½ khÃ´ng Ä‘á»§ RAM.
# 10,000 dÃ²ng (~1.7% dataset): Ä‘á»§ Ä‘á»ƒ náº¯m báº¯t phÃ¢n phá»‘i, theo heuristic phá»• biáº¿n
# cho outlier benchmark khi n > 100k (Han et al., Data Mining: Concepts and Techniques).

scaler_out = StandardScaler()
X_scaled = scaler_out.fit_transform(outlier_sample)

# â”€â”€ 1. IQR â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


def iqr_outlier_mask(df):
    mask = pd.Series([False] * len(df))
    for col in df.columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        col_mask = (df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)
        mask = mask | col_mask
    return mask.values


iqr_mask = iqr_outlier_mask(outlier_sample)

# â”€â”€ 2. Z-score â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
zscore_mat = np.abs(stats.zscore(outlier_sample, nan_policy='omit'))
zscore_mask = (zscore_mat > 3).any(axis=1)

# â”€â”€ 3. Isolation Forest â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
if_results = {}
for cont in [0.01, 0.05, 0.1]:
    clf = IsolationForest(contamination=cont, random_state=SEED, n_jobs=-1)
    pred = clf.fit_predict(X_scaled)
    if_results[f'IF_c{cont}'] = (pred == -1)

# â”€â”€ 4. LOF â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
lof_results = {}
for k in [10, 20, 50]:
    lof = LocalOutlierFactor(n_neighbors=k, n_jobs=-1)
    pred = lof.fit_predict(X_scaled)
    lof_results[f'LOF_k{k}'] = (pred == -1)

# â”€â”€ 5. DBSCAN â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
dbscan = DBSCAN(eps=3.0, min_samples=5, n_jobs=-1)
db_labels = dbscan.fit_predict(X_scaled)
dbscan_mask = (db_labels == -1)

# %% _uuid="7eb7f34e-30ac-408f-b90d-d6d345f1bc55" _cell_guid="1de7d84e-97f8-43d2-8e38-d469300b017e" jupyter={"outputs_hidden": false}
# Tá»•ng há»£p tá»‰ lá»‡ phÃ¡t hiá»‡n ngoáº¡i lai
all_masks = {
    'IQR': iqr_mask,
    'Z-score': zscore_mask,
    **if_results,
    **lof_results,
    'DBSCAN': dbscan_mask,
}

outlier_rates = {name: mask.mean() for name, mask in all_masks.items()}
print("=== Tá»‰ lá»‡ phÃ¡t hiá»‡n ngoáº¡i lai ===")
for name, rate in outlier_rates.items():
    print(f"  {name:<15}: {rate:.2%}")


# %% _uuid="2ec6ba58-996c-42b4-8e72-ef5060d984d7" _cell_guid="9a2f8be1-082b-4403-874a-e1641a2767f1" jupyter={"outputs_hidden": false}
# Jaccard similarity giá»¯a cÃ¡c phÆ°Æ¡ng phÃ¡p
def jaccard(a, b):
    intersection = np.sum(a & b)
    union = np.sum(a | b)
    return round(intersection / union, 4) if union > 0 else 0.0


method_names = list(all_masks.keys())
jaccard_mat = pd.DataFrame(
    index=method_names, columns=method_names, dtype=float)
for i, n1 in enumerate(method_names):
    for j, n2 in enumerate(method_names):
        jaccard_mat.loc[n1, n2] = jaccard(all_masks[n1], all_masks[n2])

fig, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(jaccard_mat.astype(float), annot=True, fmt='.2f', cmap='YlOrRd',
            vmin=0, vmax=1, ax=ax, linewidths=0.5)
ax.set_title(
    'Jaccard Similarity giá»¯a cÃ¡c phÆ°Æ¡ng phÃ¡p phÃ¡t hiá»‡n ngoáº¡i lai', fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'fig_06_outlier_jaccard.png'),
            dpi=100, bbox_inches='tight')
plt.show()

# %% _uuid="e573a9af-f1fb-4a95-b503-b955f44a9f36" _cell_guid="2dd6b460-51d2-4cda-a55c-506e6ab4f9fd" jupyter={"outputs_hidden": false}
# ÄÃ¡nh giÃ¡ tÃ¡c Ä‘á»™ng loáº¡i bá» ngoáº¡i lai qua KS test
print("\n=== KS test: tÃ¡c Ä‘á»™ng loáº¡i bá» ngoáº¡i lai lÃªn phÃ¢n phá»‘i TransactionAmt ===")
ref_col = 'TransactionAmt'
if ref_col in outlier_sample.columns:
    original_vals = outlier_sample[ref_col].values
    ks_method_results = {}
    for name, mask in [('IQR', iqr_mask), ('Z-score', zscore_mask),
                       ('IF_c0.05', if_results['IF_c0.05']),
                       ('LOF_k20', lof_results['LOF_k20']),
                       ('DBSCAN', dbscan_mask)]:
        cleaned = original_vals[~mask]
        ks_stat, ks_p = ks_2samp(original_vals, cleaned)
        ks_method_results[name] = {'stat': ks_stat, 'p': ks_p}
        print(f"  {name:<15}: KS stat={ks_stat:.4f}, p={ks_p:.4f}  "
              f"({'phÃ¢n phá»‘i thay Ä‘á»•i Ä‘Ã¡ng ká»ƒ' if ks_p < 0.05 else 'phÃ¢n phá»‘i á»•n Ä‘á»‹nh'})")

    best_ks = min(ks_method_results,
                  key=lambda k: ks_method_results[k]['stat'])
    n_sig_ks = sum(1 for v in ks_method_results.values() if v['p'] < 0.05)
    print(f"\n=> KS: phÆ°Æ¡ng phÃ¡p '{best_ks}' lÃ m thay Ä‘á»•i phÃ¢n phá»‘i Ã­t nháº¥t "
          f"(stat={ks_method_results[best_ks]['stat']:.4f})")
    print(f"=> {n_sig_ks}/{len(ks_method_results)} phÆ°Æ¡ng phÃ¡p thay Ä‘á»•i phÃ¢n phá»‘i Ä‘Ã¡ng ká»ƒ (KS p<0.05)")

# Táº¡o mask ngoáº¡i lai cuá»‘i cÃ¹ng (báº£o thá»§: giao cá»§a IQR vÃ  IF_c0.05)
final_outlier_mask_sample = iqr_mask & if_results['IF_c0.05']
print(f"\n-> Ngoáº¡i lai Ä‘Æ°á»£c xÃ¡c nháº­n bá»Ÿi cáº£ IQR + IF: "
      f"{final_outlier_mask_sample.mean():.2%} ({final_outlier_mask_sample.sum()} dÃ²ng)")

# %% [markdown]
# **LÃ½ do chá»n IQR clipping cho production:**
#
# KS test cho tháº¥y DBSCAN thay Ä‘á»•i phÃ¢n phá»‘i Ã­t nháº¥t â€” khÃ´ng pháº£i vÃ¬ nÃ³ phÃ¡t hiá»‡n tá»‘t hÆ¡n,
# mÃ  vÃ¬ vá»›i `eps=3.0` trong khÃ´ng gian nhiá»u chiá»u (20 cá»™t), DBSCAN gáº§n nhÆ° khÃ´ng Ä‘Ã¡nh dáº¥u
# Ä‘iá»ƒm nÃ o lÃ  ngoáº¡i lai (tá»‰ lá»‡ phÃ¡t hiá»‡n ráº¥t tháº¥p) â€” thay Ä‘á»•i phÃ¢n phá»‘i Ã­t vÃ¬ loáº¡i ráº¥t Ã­t Ä‘iá»ƒm.
# IQR lÃ m thay Ä‘á»•i nhiá»u hÆ¡n VÃŒ Ä‘Ã¢Y LÃ€ TÃN HIá»†U Tá»T: nÃ³ Ä‘ang xá»­ lÃ½ Ä‘Æ°á»£c outlier thá»±c sá»±.
#
# Chá»n **IQR clipping** (giá»›i háº¡n thay vÃ¬ xÃ³a dÃ²ng) vÃ¬:
# 1. Báº£o toÃ n sá»‘ dÃ²ng (khÃ´ng giáº£m dataset tá»« 590k xuá»‘ng cÃ²n Ã­nhau);
# 2. Interpretá»ble vÃ  deterministic (khÃ´ng phá»¥ thuá»™c vÃ o hyperparameter eps nhÆ° DBSCAN);
# 3. Scale tá»‘t vá»›i 400+ cá»™t sá»‘ (O(nÂ·d), khÃ´ng O(nÂ²) nhÆ° LOF/DBSCAN);
# 4. KS stat váº«n cháº¥p nháº­n Ä‘Æ°á»£c: IQR chá»‰ cáº¯t Ä‘uá»™i phÃ¢n phá»‘i, giá»¯ hÃ¬nh dáº¡ng tá»•ng thá»ƒ.

# %%


def apply_iqr_clip(df, cols, factor=1.5):
    """Giá»›i háº¡n (clip) giÃ¡ trá»‹ cá»™t vá» [Q1 - factor*IQR, Q3 + factor*IQR].
    Thay vÃ¬ xÃ³a dÃ²ng, giá»¯ láº¡i nhÆ°ng cap giÃ¡ trá»‹ â€” báº£o toÃ n sá»‘ lÆ°á»£ng máº«u."""
    df = df.copy()
    for col in cols:
        if col not in df.columns:
            continue
        Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        IQR = Q3 - Q1
        df[col] = df[col].clip(lower=Q1 - factor * IQR,
                               upper=Q3 + factor * IQR)
    return df


clip_cols = [c for c in outlier_cols if c != 'isFraud']
train = apply_iqr_clip(train, clip_cols)
print(
    f"\n-> ÄÃ£ Ã¡p dá»¥ng IQR clipping trÃªn {len(clip_cols)} cá»™t sá»‘ cá»§a train")

# %% [markdown] _uuid="93459f5d-fe94-4948-8e4c-fa82b673a0e3" _cell_guid="87a88e09-4555-49ee-8ba0-6d89d82507ba" jupyter={"outputs_hidden": false}
# ---
# ## 2.2.3c. Chuáº©n hÃ³a dá»¯ liá»‡u cÃ³ kiá»ƒm Ä‘á»‹nh
# #
# Chuáº©n hÃ³a Ä‘Æ°a cÃ¡c Ä‘áº·c trÆ°ng vá» cÃ¹ng thang Ä‘o, giÃºp cÃ¡c mÃ´ hÃ¬nh dá»±a trÃªn khoáº£ng cÃ¡ch (kNN, SVM, LR)
# hoáº¡t Ä‘á»™ng á»•n Ä‘á»‹nh. So sÃ¡nh 5 phÆ°Æ¡ng phÃ¡p:
# #
# | PhÆ°Æ¡ng phÃ¡p | CÃ´ng thá»©c | Nháº¡y vá»›i outlier | PhÃ¢n phá»‘i output |
# |---|---|---|---|
# | **Min-Max** | $x' = \frac{x - x_{\min}}{x_{\max} - x_{\min}} \in [0,1]$ | Ráº¥t nháº¡y | Giá»¯ nguyÃªn hÃ¬nh dáº¡ng |
# | **Z-score (Standard)** | $x' = \frac{x - \mu}{\sigma}$ | KhÃ¡ nháº¡y | Trung bÃ¬nh 0, std 1 |
# | **Robust** | $x' = \frac{x - Q_2}{Q_3 - Q_1}$ | Bá»n vá»¯ng (dÃ¹ng median/IQR) | Giá»¯ nguyÃªn hÃ¬nh dáº¡ng |
# | **Quantile-Uniform** | Map sang $\text{Uniform}(0,1)$ qua CDF thá»±c nghiá»‡m | Bá»n vá»¯ng | Äá»u Ä‘á»u |
# | **Quantile-Normal** | Map sang $\mathcal{N}(0,1)$ qua CDF thá»±c nghiá»‡m | Bá»n vá»¯ng | Chuáº©n |
# #
# **Levene's test** kiá»ƒm tra Ä‘á»“ng nháº¥t phÆ°Æ¡ng sai (homoscedasticity) sau chuáº©n hÃ³a:
# #
# $$H_0: \sigma_1^2 = \sigma_2^2 = \cdots = \sigma_k^2$$
# #
# Náº¿u $p > 0.05$ â†’ homoscedastic â†’ chuáº©n hÃ³a hiá»‡u quáº£.
# **RobustScaler** Ä‘Æ°á»£c chá»n cho production vÃ¬ bá»n vá»¯ng vá»›i outlier cÃ²n sÃ³t sau bÆ°á»›c 2.2.3b.

# %% _uuid="0b324a82-beee-4a40-ac55-10a674a29a55" _cell_guid="269a62a8-6021-4d71-a908-ec9c535aa962" jupyter={"outputs_hidden": false}

# Chá»n táº­p cá»™t sá»‘ benchmark (khÃ´ng thiáº¿u sau imputation)
scale_cols = [c for c in outlier_cols if c != 'isFraud'][:10]
scale_sample = train[scale_cols].sample(
    min(5000, len(train)), random_state=SEED).copy()

scalers = {
    'Min-Max': MinMaxScaler(),
    'Z-score': StandardScaler(),
    'Robust': RobustScaler(),
    'Quantile-Uniform': QuantileTransformer(output_distribution='uniform',
                                            random_state=SEED),
    'Quantile-Normal': QuantileTransformer(output_distribution='normal',
                                           random_state=SEED),
}

scaled_dfs = {}
levene_results = {}

for name, scaler in scalers.items():
    arr = scaler.fit_transform(scale_sample)
    scaled_dfs[name] = pd.DataFrame(arr, columns=scale_cols)
    # Levene's test Ä‘Ã¡nh giÃ¡ Ä‘á»“ng nháº¥t phÆ°Æ¡ng sai giá»¯a cÃ¡c thuá»™c tÃ­nh
    groups = [scaled_dfs[name][c].dropna().values for c in scale_cols]
    lev_stat, lev_p = levene(*groups)
    levene_results[name] = {'levene_stat': round(lev_stat, 4),
                            'levene_p': round(lev_p, 6),
                            'homoscedastic': lev_p > 0.05}

levene_df = pd.DataFrame(levene_results).T
print("=== Levene's test Ä‘Ã¡nh giÃ¡ homoscedasticity sau chuáº©n hÃ³a ===")
print(levene_df.to_string())
print("\n-> Quantile Transform thÆ°á»ng Ä‘áº¡t homoscedasticity tá»‘t nháº¥t vÃ¬ chuáº©n hÃ³a phÃ¢n phá»‘i.")

# %% _uuid="d738db4c-6279-40de-8ed6-27b12dc7b2e5" _cell_guid="fbe6218d-2c40-40c4-a9b5-cc4cf11c5db8" jupyter={"outputs_hidden": false}
# Violin plot phÃ¢n phá»‘i sau tá»«ng phÆ°Æ¡ng phÃ¡p chuáº©n hÃ³a (táº¥t cáº£ cá»™t benchmark)
n_vcols = len(scale_cols)
ncols_grid = 3
nrows_grid = (n_vcols + ncols_grid - 1) // ncols_grid
fig, axes = plt.subplots(nrows_grid, ncols_grid,
                         figsize=(7 * ncols_grid, 5 * nrows_grid))
axes_flat = axes.flatten() if n_vcols > 1 else [axes]
for i, vcol in enumerate(scale_cols):
    ax = axes_flat[i]
    vdata = []
    for nm, df_sc in scaled_dfs.items():
        vdata.append(pd.DataFrame({'value': df_sc[vcol], 'method': nm}))
    vdf = pd.concat(vdata, ignore_index=True)
    sns.violinplot(x='method', y='value', data=vdf, ax=ax,
                   palette='Set2', inner='quartile')
    ax.set_title(f'"{vcol}"', fontsize=10)
    ax.set_xlabel('')
    ax.tick_params(axis='x', rotation=30)
for j in range(i + 1, len(axes_flat)):
    axes_flat[j].set_visible(False)
fig.suptitle(
    'PhÃ¢n phá»‘i sau cÃ¡c phÆ°Æ¡ng phÃ¡p chuáº©n hÃ³a (táº¥t cáº£ cá»™t benchmark)', fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'fig_07_scaling_violin.png'),
            dpi=100, bbox_inches='tight')
plt.show()

# %% _uuid="f5444bfc-82d2-45bf-8859-2157f10e6aba" _cell_guid="10d9c6f7-d9a3-4295-bddd-710d6482a558" jupyter={"outputs_hidden": false}
# Ãp dá»¥ng chuáº©n hÃ³a RobustScaler lÃªn TOÃ€N Bá»˜ cá»™t sá»‘ (khÃ´ng chá»‰ 10 cá»™t benchmark)
print("\nÃp dá»¥ng RobustScaler lÃªn táº¥t cáº£ cá»™t sá»‘ cá»§a train & test...")
all_num_for_scale = [
    c for c in all_num_cols_imp if c in train.columns and c in test.columns]
final_scaler = RobustScaler()
train[all_num_for_scale] = final_scaler.fit_transform(train[all_num_for_scale])
test[all_num_for_scale] = final_scaler.transform(test[all_num_for_scale])
print(
    f"  HoÃ n thÃ nh: {len(all_num_for_scale)} cá»™t Ä‘Ã£ Ä‘Æ°á»£c chuáº©n hÃ³a báº±ng RobustScaler.")

# %% [markdown]
# ---
# ## 2.2.3d. MÃ£ hÃ³a biáº¿n phÃ¢n loáº¡i nÃ¢ng cao
# #
# CÃ¡c mÃ´ hÃ¬nh ML yÃªu cáº§u Ä‘áº§u vÃ o dáº¡ng sá»‘. Chiáº¿n lÆ°á»£c mÃ£ hÃ³a phá»¥ thuá»™c vÃ o **cardinality** (sá»‘ giÃ¡ trá»‹ duy nháº¥t):
# #
# | PhÆ°Æ¡ng phÃ¡p | Cardinality | CÆ¡ cháº¿ | LÆ°u Ã½ |
# |---|---|---|---|
# | **One-Hot Encoding** | Tháº¥p ($\leq 20$) | Má»—i giÃ¡ trá»‹ thÃ nh má»™t cá»™t nhá»‹ phÃ¢n | Táº¡o ma tráº­n thÆ°a; khÃ´ng cÃ³ thá»© tá»± |
# | **Ordinal Encoding** | Báº¥t ká»³ | Map giÃ¡ trá»‹ sang sá»‘ nguyÃªn $0, 1, \ldots, k-1$ | Giáº£ Ä‘á»‹nh thá»© tá»±; phÃ¹ há»£p cho tree models |
# | **Target Encoding (CV)** | Cao | $\hat{x}_i = \mathbb{E}[y \mid x = x_i]$ Æ°á»›c tÃ­nh qua CV | **Báº¯t buá»™c dÃ¹ng CV** Ä‘á»ƒ trÃ¡nh data leakage |
# | **Binary Encoding** | Cao ($> 20$) | MÃ£ hÃ³a ordinal rá»“i biá»ƒu diá»…n nhá»‹ phÃ¢n | Sá»‘ cá»™t: $\lceil \log_2 k \rceil$ thay vÃ¬ $k$ |
# | **Frequency Encoding** | Báº¥t ká»³ | $\hat{x}_i = P(x = x_i)$ | ÄÆ¡n giáº£n, khÃ´ng phá»¥ thuá»™c target |
# #
# **VIF (Variance Inflation Factor)** sau má»—i phÆ°Æ¡ng phÃ¡p encode Ä‘á»ƒ phÃ¡t hiá»‡n Ä‘a cá»™ng tuyáº¿n má»›i:
# $$\text{VIF}_j = \frac{1}{1 - R_j^2}$$
# $\text{VIF} > 10$ â†’ Ä‘áº·c trÆ°ng $j$ bá»‹ giáº£i thÃ­ch gáº§n nhÆ° hoÃ n toÃ n bá»Ÿi cÃ¡c Ä‘áº·c trÆ°ng khÃ¡c.

# %% [markdown]
# ### BÆ°á»›c 1: PhÃ¢n tÃ­ch cardinality & Demo so sÃ¡nh 5 phÆ°Æ¡ng phÃ¡p (khÃ´ng Ä‘á»™t biáº¿n train/test)
# #
# Äá»ƒ so sÃ¡nh khÃ¡ch quan, ta Ã¡p dá»¥ng tá»«ng phÆ°Æ¡ng phÃ¡p trÃªn **cÃ¹ng má»™t táº­p demo** (copy, khÃ´ng thay Ä‘á»•i train/test gá»‘c), sau Ä‘Ã³ Ä‘o VIF trÃªn cÃ¡c cá»™t Ä‘Æ°á»£c táº¡o ra.

# %%

# XÃ¡c Ä‘á»‹nh cá»™t phÃ¢n loáº¡i (object) cÃ³ trong Cáº¢ train VÃ€ test
train_cat_set = set(train.select_dtypes(include=['object']).columns.tolist())
test_cat_set = set(test.select_dtypes(include=['object']).columns.tolist())

# Cá»™t object chá»‰ cÃ³ trong train â†’ thÃªm NaN placeholder vÃ o test
train_only_cat = train_cat_set - test_cat_set
if train_only_cat:
    print(
        f"Cá»™t object chá»‰ cÃ³ trong train (thÃªm NaN vÃ o test): {sorted(train_only_cat)}")
    for c in train_only_cat:
        test[c] = np.nan

all_cat_cols = [c for c in train.select_dtypes(include=['object']).columns
                if c in test.columns]

cardinality = {col: train[col].nunique() for col in all_cat_cols}
low_card = [c for c, v in cardinality.items() if v <= 20]
high_card = [c for c, v in cardinality.items() if v > 20]

print(f"Tá»•ng cá»™t phÃ¢n loáº¡i (chung train & test): {len(all_cat_cols)}")
print(
    f"  Low-cardinality  (â‰¤20 giÃ¡ trá»‹): {len(low_card)} cá»™t â†’ dÃ¹ng OHE")
print(
    f"  High-cardinality (>20 giÃ¡ trá»‹): {len(high_card)} cá»™t â†’ dÃ¹ng Binary Encoding")
print(f"\nTop 5 low-card cols : {low_card[:5]}")
print(f"Top 5 high-card cols: {high_card[:5]}")

# %%
# â”€â”€ DEMO: Ãp dá»¥ng tá»«ng phÆ°Æ¡ng phÃ¡p lÃªn df_demo (khÃ´ng thay Ä‘á»•i train/test) â”€â”€
# Chá»n 3 cá»™t low-card + 2 cá»™t high-card Ä‘á»ƒ minh há»a
demo_low = low_card[:3]
demo_high = high_card[:2]
demo_cols = demo_low + demo_high
print(f"Demo columns: {demo_cols}")
print(f"Cardinality : {[cardinality[c] for c in demo_cols]}")


def compute_vif_summary(df_encoded, n_sample=3000, seed=SEED):
    """TÃ­nh VIF trÃªn tá»‘i Ä‘a n_sample dÃ²ng, tráº£ vá» (mean_vif, max_vif)."""
    num_cols = df_encoded.select_dtypes(include=[np.number]).columns.tolist()
    valid = [c for c in num_cols if df_encoded[c].std() > 0]
    if len(valid) < 2:
        return np.nan, np.nan
    sample = df_encoded[valid].dropna().sample(
        min(n_sample, len(df_encoded)), random_state=seed)
    vifs = [variance_inflation_factor(sample.values, i)
            for i in range(sample.shape[1])]
    finite_vifs = [v for v in vifs if np.isfinite(v)]
    if not finite_vifs:
        return np.nan, np.nan
    return round(np.mean(finite_vifs), 2), round(np.max(finite_vifs), 2)


def target_encode_cv(df_tr, col, target_col, n_splits=5, seed=42):
    """Target encoding vá»›i cross-validation Ä‘á»ƒ trÃ¡nh data leakage."""
    out = np.zeros(len(df_tr))
    global_mean = df_tr[target_col].mean()
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr_idx, val_idx in kf.split(df_tr):
        means = df_tr.iloc[tr_idx].groupby(col)[target_col].mean()
        out[val_idx] = df_tr.iloc[val_idx][col].map(
            means).fillna(global_mean).values
    return out


df_demo_base = train[demo_cols + ['isFraud']].copy()
vif_comparison = []

# â”€â”€â”€ PhÆ°Æ¡ng phÃ¡p 1: One-Hot Encoding â”€â”€â”€
df1 = df_demo_base.copy()
ohe_demo = ce.OneHotEncoder(
    cols=demo_cols, use_cat_names=True, handle_missing='value')
df1_enc = ohe_demo.fit_transform(df1.drop(columns=['isFraud']))
n_cols_ohe = df1_enc.shape[1]
mean_v, max_v = compute_vif_summary(df1_enc)
vif_comparison.append({'PhÆ°Æ¡ng phÃ¡p': 'One-Hot Encoding', 'Sá»‘ cá»™t táº¡o ra': n_cols_ohe,
                       'VIF trung bÃ¬nh': mean_v, 'VIF cao nháº¥t': max_v})
print(
    f"OHE      : {len(demo_cols)} cá»™t gá»‘c â†’ {n_cols_ohe} cá»™t | VIF mean={mean_v}, max={max_v}")

# â”€â”€â”€ PhÆ°Æ¡ng phÃ¡p 2: Ordinal Encoding â”€â”€â”€
df2 = df_demo_base.copy()
for col in demo_cols:
    le = LabelEncoder()
    le.fit(df2[col].astype(str))
    df2[f'{col}_ord'] = le.transform(df2[col].astype(str))
df2_enc = df2[[f'{c}_ord' for c in demo_cols]]
mean_v, max_v = compute_vif_summary(df2_enc)
vif_comparison.append({'PhÆ°Æ¡ng phÃ¡p': 'Ordinal Encoding', 'Sá»‘ cá»™t táº¡o ra': len(demo_cols),
                       'VIF trung bÃ¬nh': mean_v, 'VIF cao nháº¥t': max_v})
print(
    f"Ordinal  : {len(demo_cols)} cá»™t gá»‘c â†’ {len(demo_cols)} cá»™t | VIF mean={mean_v}, max={max_v}")

# â”€â”€â”€ PhÆ°Æ¡ng phÃ¡p 3: Target Encoding (5-fold CV) â”€â”€â”€
df3 = df_demo_base.copy()
for col in demo_cols:
    df3[f'{col}_te'] = target_encode_cv(df3, col, 'isFraud', seed=SEED)
df3_enc = df3[[f'{c}_te' for c in demo_cols]]
mean_v, max_v = compute_vif_summary(df3_enc)
vif_comparison.append({'PhÆ°Æ¡ng phÃ¡p': 'Target Encoding (CV)', 'Sá»‘ cá»™t táº¡o ra': len(demo_cols),
                       'VIF trung bÃ¬nh': mean_v, 'VIF cao nháº¥t': max_v})
print(
    f"Target CV: {len(demo_cols)} cá»™t gá»‘c â†’ {len(demo_cols)} cá»™t | VIF mean={mean_v}, max={max_v}")

# â”€â”€â”€ PhÆ°Æ¡ng phÃ¡p 4: Binary Encoding â”€â”€â”€
df4 = df_demo_base.copy()
bin_demo = ce.BinaryEncoder(cols=demo_cols, handle_missing='value')
df4_enc = bin_demo.fit_transform(df4[demo_cols])
n_cols_bin = df4_enc.shape[1]
mean_v, max_v = compute_vif_summary(df4_enc)
vif_comparison.append({'PhÆ°Æ¡ng phÃ¡p': 'Binary Encoding', 'Sá»‘ cá»™t táº¡o ra': n_cols_bin,
                       'VIF trung bÃ¬nh': mean_v, 'VIF cao nháº¥t': max_v})
print(
    f"Binary   : {len(demo_cols)} cá»™t gá»‘c â†’ {n_cols_bin} cá»™t | VIF mean={mean_v}, max={max_v}")

# â”€â”€â”€ PhÆ°Æ¡ng phÃ¡p 5: Frequency Encoding â”€â”€â”€
df5 = df_demo_base.copy()
for col in demo_cols:
    freq_map = df5[col].value_counts(normalize=True)
    df5[f'{col}_freq'] = df5[col].map(freq_map).fillna(0)
df5_enc = df5[[f'{c}_freq' for c in demo_cols]]
mean_v, max_v = compute_vif_summary(df5_enc)
vif_comparison.append({'PhÆ°Æ¡ng phÃ¡p': 'Frequency Encoding', 'Sá»‘ cá»™t táº¡o ra': len(demo_cols),
                       'VIF trung bÃ¬nh': mean_v, 'VIF cao nháº¥t': max_v})
print(
    f"Frequency: {len(demo_cols)} cá»™t gá»‘c â†’ {len(demo_cols)} cá»™t | VIF mean={mean_v}, max={max_v}")

# %%
# Báº£ng so sÃ¡nh vÃ  biá»ƒu Ä‘á»“ VIF
vif_comp_df = pd.DataFrame(vif_comparison)
print("\n=== Báº£ng so sÃ¡nh 5 phÆ°Æ¡ng phÃ¡p mÃ£ hÃ³a (demo trÃªn cÃ¹ng táº­p cá»™t) ===")
print(vif_comp_df.to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
colors = ['steelblue', 'darkorange', 'tomato', 'forestgreen', 'mediumorchid']
methods = vif_comp_df['PhÆ°Æ¡ng phÃ¡p'].tolist()

axes[0].bar(methods, vif_comp_df['VIF trung bÃ¬nh'].fillna(0),
            color=colors, edgecolor='white')
axes[0].axhline(10, color='red', linestyle='--', label='NgÆ°á»¡ng VIF=10')
axes[0].set_title('VIF Trung BÃ¬nh theo PhÆ°Æ¡ng PhÃ¡p MÃ£ HÃ³a', fontsize=12)
axes[0].set_ylabel('VIF trung bÃ¬nh')
axes[0].tick_params(axis='x', rotation=20)
axes[0].legend()
axes[0].grid(axis='y', alpha=0.3)

axes[1].bar(methods, vif_comp_df['VIF cao nháº¥t'].fillna(0),
            color=colors, edgecolor='white')
axes[1].axhline(10, color='red', linestyle='--', label='NgÆ°á»¡ng VIF=10')
axes[1].set_title('VIF Cao Nháº¥t theo PhÆ°Æ¡ng PhÃ¡p MÃ£ HÃ³a', fontsize=12)
axes[1].set_ylabel('VIF cao nháº¥t')
axes[1].tick_params(axis='x', rotation=20)
axes[1].legend()
axes[1].grid(axis='y', alpha=0.3)

plt.suptitle('So sÃ¡nh Ä‘a cá»™ng tuyáº¿n (VIF) sau má»—i phÆ°Æ¡ng phÃ¡p mÃ£ hÃ³a\n'
             f'(demo trÃªn {len(demo_cols)} cá»™t: {demo_cols})', fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(
    OUTPUT_DIR, 'fig_07b_vif_encoding_comparison.png'), dpi=100, bbox_inches='tight')
plt.show()

# %% [markdown]
# **Nháº­n xÃ©t VIF:**
# - **One-Hot Encoding** cÃ³ thá»ƒ táº¡o Ä‘a cá»™ng tuyáº¿n hoÃ n háº£o (dummy variable trap) náº¿u khÃ´ng bá» má»™t cá»™t;
#   category_encoders tá»± Ä‘á»™ng xá»­ lÃ½ Ä‘iá»u nÃ y.
# - **Target Encoding (CV)** táº¡o ra VIF tháº¥p vÃ¬ má»—i cá»™t lÃ  trung bÃ¬nh má»™t phÃ¢n phá»‘i khÃ¡c nhau.
# - **Binary Encoding** táº¡o cá»™t nhá»‹ phÃ¢n Ä‘á»™c láº­p (VIF tháº¥p) Ä‘á»“ng thá»i tiáº¿t kiá»‡m khÃ´ng gian hÆ¡n OHE.
# - **Frequency Encoding** pháº£n Ã¡nh táº§n suáº¥t â†’ cÃ³ thá»ƒ tÆ°Æ¡ng quan vá»›i nhau náº¿u phÃ¢n phá»‘i tÆ°Æ¡ng tá»±.
# - PhÆ°Æ¡ng phÃ¡p nÃ o cÃ³ **VIF > 10** cáº§n cÃ¢n nháº¯c loáº¡i bá» hoáº·c thay tháº¿ Ä‘á»ƒ trÃ¡nh báº¥t á»•n trong há»“i quy.

# %% [markdown]
# ### BÆ°á»›c 2: Ãp dá»¥ng encoding lÃªn toÃ n bá»™ train / test
# #
# Chiáº¿n lÆ°á»£c: OHE cho cá»™t low-card, Binary cho cá»™t high-card, thÃªm Target-CV + Frequency + Ordinal
# cho táº¥t cáº£ cá»™t phÃ¢n loáº¡i.

# %%
print("=== Ãp dá»¥ng encoding lÃªn toÃ n bá»™ train/test ===\n")

# 1. One-Hot Encoding (cá»™t low-cardinality â‰¤20)
low_card_exist = [
    c for c in low_card if c in train.columns and train[c].dtype == object]
if low_card_exist:
    ohe = ce.OneHotEncoder(cols=low_card_exist,
                           use_cat_names=True, handle_missing='value')
    ohe_train = ohe.fit_transform(train[low_card_exist])
    ohe_test = ohe.transform(test[low_card_exist])
    train = pd.concat([train.drop(columns=low_card_exist), ohe_train], axis=1)
    test = pd.concat([test.drop(columns=low_card_exist),  ohe_test],  axis=1)
    print(
        f"[1/5] OHE: {len(low_card_exist)} low-card cols â†’ {ohe_train.shape[1]} binary cols | train: {train.shape}")
else:
    print("[1/5] OHE: khÃ´ng cÃ³ cá»™t low-cardinality.")

# Cáº­p nháº­t danh sÃ¡ch cá»™t phÃ¢n loáº¡i cÃ²n láº¡i (sau OHE)
cat_cols_remain = [c for c in train.select_dtypes(include=['object']).columns
                   if c in test.columns]

# 2. Ordinal Encoding (thÃªm _ord cho táº¥t cáº£ cá»™t phÃ¢n loáº¡i cÃ²n láº¡i)
ordinal_encoders = {}
for col in cat_cols_remain:
    le = LabelEncoder()
    combined = list(train[col].astype(str)) + list(test[col].astype(str))
    le.fit(combined)
    train[f'{col}_ord'] = le.transform(train[col].astype(str))
    test[f'{col}_ord'] = le.transform(test[col].astype(str))
    ordinal_encoders[col] = le
print(f"[2/5] Ordinal Encoding: thÃªm {len(cat_cols_remain)} cá»™t _ord")

# 3. Target Encoding (5-fold CV, trÃ¡nh leakage)
target_means_store = {}
print(f"[3/5] Target Encoding (5-fold CV) â€“ {len(cat_cols_remain)} cá»™t...")
for col in cat_cols_remain:
    train[f'{col}_te'] = target_encode_cv(train, col, 'isFraud', seed=SEED)
    means_map = train.groupby(col)['isFraud'].mean()
    target_means_store[col] = means_map
    test[f'{col}_te'] = test[col].map(
        means_map).fillna(train['isFraud'].mean())
print(f"    ÄÃ£ thÃªm {len(cat_cols_remain)} cá»™t _te")

# 4. Binary Encoding (cá»™t high-cardinality >20)
high_card_remain = [c for c in cat_cols_remain if c in train.columns
                    and train[c].dtype == object and cardinality.get(c, 0) > 20]
if high_card_remain:
    binary_enc = ce.BinaryEncoder(
        cols=high_card_remain, handle_missing='value')
    be_train = binary_enc.fit_transform(train[high_card_remain])
    be_test = binary_enc.transform(test[high_card_remain])
    train = pd.concat([train.drop(columns=high_card_remain), be_train], axis=1)
    test = pd.concat([test.drop(columns=high_card_remain),  be_test],  axis=1)
    print(
        f"[4/5] Binary Encoding: {len(high_card_remain)} high-card cols â†’ {be_train.shape[1]} binary cols | train: {train.shape}")
else:
    print("[4/5] Binary Encoding: khÃ´ng cÃ³ cá»™t high-card nÃ o cÃ²n láº¡i sau OHE.")

# 5. Frequency Encoding (thÃªm _freq cho táº¥t cáº£ cá»™t phÃ¢n loáº¡i cÃ²n láº¡i cÃ²n lÃ  object)
freq_cols_all = [
    c for c in cat_cols_remain if c in train.columns and train[c].dtype == object]
for col in freq_cols_all:
    freq_map = train[col].value_counts(normalize=True)
    train[f'{col}_freq'] = train[col].map(freq_map).fillna(0)
    test[f'{col}_freq'] = test[col].map(freq_map).fillna(0)
print(f"[5/5] Frequency Encoding: thÃªm {len(freq_cols_all)} cá»™t _freq")

# XÃ³a cá»™t object gá»‘c cÃ²n sÃ³t láº¡i
residual_obj = [c for c in train.select_dtypes(
    include=['object']).columns if c != 'isFraud']
if residual_obj:
    train = train.drop(columns=residual_obj)
    test = test.drop(columns=[c for c in residual_obj if c in test.columns])
    print(f"\nXÃ³a {len(residual_obj)} cá»™t object gá»‘c cÃ²n sÃ³t.")
print(f"\nâ†’ train shape sau encoding: {train.shape}")
print(f"â†’ test  shape sau encoding: {test.shape}")

# %%
# VIF trÃªn cÃ¡c cá»™t _te vÃ  _ord Ä‘á»ƒ kiá»ƒm tra Ä‘a cá»™ng tuyáº¿n sau encoding
print("\n=== VIF sau encoding (cá»™t _te vÃ  _ord) ===")
enc_vif_cols = ([c for c in train.columns if c.endswith('_te')][:8] +
                [c for c in train.columns if c.endswith('_ord')][:7])
enc_vif_cols = [c for c in enc_vif_cols if train[c].std() > 0]

if len(enc_vif_cols) >= 2:
    vif_sample = train[enc_vif_cols].dropna().sample(
        min(3000, len(train)), random_state=SEED)
    vif_enc_data = pd.DataFrame({
        'feature': enc_vif_cols,
        'VIF': [variance_inflation_factor(vif_sample.values, i)
                for i in range(len(enc_vif_cols))]
    }).sort_values('VIF', ascending=False)
    print(vif_enc_data.to_string(index=False))
    high_vif = vif_enc_data[vif_enc_data['VIF'] > 10]
    print(
        f"\nâ†’ Äáº·c trÆ°ng cÃ³ VIF > 10 (Ä‘a cá»™ng tuyáº¿n Ä‘Ã¡ng ngáº¡i): {len(high_vif)}")
    if len(high_vif) > 0:
        print("  Cáº§n xem xÃ©t loáº¡i bá»:", high_vif['feature'].tolist())
else:
    print("KhÃ´ng Ä‘á»§ cá»™t encoded Ä‘á»ƒ tÃ­nh VIF.")

# %% [markdown]
# ---
# ## 2.2.3e. Lá»±a chá»n vÃ  giáº£m chiá»u Ä‘áº·c trÆ°ng
# #
# Ba táº§ng lá»±a chá»n Ä‘áº·c trÆ°ng theo yÃªu cáº§u:
# #
# | Táº§ng | PhÆ°Æ¡ng phÃ¡p | NguyÃªn lÃ½ |
# |---|---|---|
# | **Táº§ng 1 â€“ Lá»c thá»‘ng kÃª** | ANOVA F-test, Chi-square, Mutual Information | ÄÃ¡nh giÃ¡ Ä‘á»™c láº­p tá»«ng Ä‘áº·c trÆ°ng vá»›i target |
# | **Táº§ng 2 â€“ Dá»±a trÃªn mÃ´ hÃ¬nh** | RF importance, GB importance, RFE + CV | ÄÃ¡nh giÃ¡ theo Ä‘Ã³ng gÃ³p trong mÃ´ hÃ¬nh cá»¥ thá»ƒ |
# | **Táº§ng 3 â€“ Giáº£m chiá»u** | PCA (95% variance), t-SNE, UMAP | Biáº¿n Ä‘á»•i/trá»±c quan hÃ³a khÃ´ng gian Ä‘áº·c trÆ°ng |
# #
# **Táº§ng 1 â€“ CÃ´ng thá»©c toÃ¡n:**
# #
# **ANOVA F-test** so sÃ¡nh phÆ°Æ¡ng sai giá»¯a cÃ¡c nhÃ³m ($y=0$ vÃ  $y=1$):
# #
# $$F = \frac{\text{between-group variance}}{\text{within-group variance}}
# = \frac{\sum_k n_k (\bar{x}_k - \bar{x})^2 / (K-1)}
# {\sum_k \sum_i (x_{ki} - \bar{x}_k)^2 / (N-K)}$$
# #
# $F$ lá»›n $\Rightarrow$ Ä‘áº·c trÆ°ng phÃ¢n biá»‡t Ä‘Æ°á»£c tá»‘t giá»¯a cÃ¡c nhÃ³m.
# #
# **Chi-square** kiá»ƒm tra Ä‘á»™c láº­p giá»¯a Ä‘áº·c trÆ°ng phÃ¢n loáº¡i vÃ  target:
# #
# $$\chi^2 = \sum_{i,j} \frac{(O_{ij} - E_{ij})^2}{E_{ij}}, \quad
# E_{ij} = \frac{R_i \cdot C_j}{N}$$
# #
# **Mutual Information** Ä‘o thÃ´ng tin cá»§a $X_j$ vá» target $Y$:
# #
# $$I(X_j; Y) = \sum_{x,y} p(x,y) \log \frac{p(x,y)}{p(x)\,p(y)}$$
# #
# KhÃ´ng cÃ³ giáº£ Ä‘á»‹nh tuyáº¿n tÃ­nh â€” pháº©t hiá»‡n cáº£ quan há»‡ phi tuyáº¿n.
# #
# **Táº§ng 2 â€“ RFE (Recursive Feature Elimination)**:
# Láº·p: train model â†’ xÃ¡c Ä‘á»‹nh Ä‘áº·c trÆ°ng Ã­t quan trá»ng nháº¥t â†’ loáº¡i â†’ láº·p láº¡i.
# DÃ¹ng 5-fold CV F1-score Ä‘á»ƒ chá»n $k$ tá»‘t nháº¥t.
# #
# **Táº§ng 3 â€“ PCA giáº£m chiá»u** (giá»¯ $k^*$ components Ä‘áº¡t 95% explained variance):
# #
# $$k^* = \min\left\{k : \sum_{i=1}^k \lambda_i \big/ \sum_{i} \lambda_i \geq 0.95\right\}$$
# #
# **ÄÃ¡nh giÃ¡ cuá»‘i**: vá»›i má»—i phÆ°Æ¡ng phÃ¡p lá»c, huáº¥n luyá»‡n Logistic Regression vÃ  bÃ¡o cÃ¡o **5-fold CV F1-score** theo sá»‘ lÆ°á»£ng Ä‘áº·c trÆ°ng.

# %%

# â”€â”€ Chuáº©n bá»‹ X_fs, y_fs (dÃ¹ng chung cho cáº£ 3 táº§ng) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
fs_num_cols = [c for c in train.select_dtypes(include=[np.number]).columns
               if c not in ('isFraud', 'TransactionID', 'TransactionDT')]
X_fs = train[fs_num_cols].fillna(0)
y_fs = train['isFraud']

FS_N = min(30_000, len(X_fs))
idx_fs = X_fs.sample(FS_N, random_state=SEED).index
X_fs_sample = X_fs.loc[idx_fs]
y_fs_sample = y_fs.loc[idx_fs]

print(
    f"=== Feature Selection: {X_fs_sample.shape[0]:,} dÃ²ng Ã— {X_fs_sample.shape[1]} cá»™t ===")
print(f"  Tá»‰ lá»‡ Fraud trong sample: {y_fs_sample.mean():.3f}")

# %% [markdown]
# ### Táº§ng 1: Lá»c thá»‘ng kÃª (ANOVA F-test Â· Chi-square Â· Mutual Information)

# %%
print("\n" + "="*60)
print("Táº¦NG 1: Lá»ŒC THá»NG KÃŠ")
print("="*60)

# â”€â”€â”€ 1a. ANOVA F-test (Ä‘áº·c trÆ°ng sá»‘) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
f_vals, p_anova = f_classif(X_fs_sample, y_fs_sample)
anova_df = pd.DataFrame(
    {'feature': fs_num_cols, 'F_stat': f_vals, 'p_value': p_anova})
anova_df = anova_df.sort_values(
    'F_stat', ascending=False).reset_index(drop=True)
top_anova_feats = anova_df.head(20)['feature'].tolist()
print(
    f"\n[1a] ANOVA F-test â€“ Top 15 (trÃªn {len(fs_num_cols)} Ä‘áº·c trÆ°ng sá»‘):")
print(anova_df.head(15).to_string(index=False))

# â”€â”€â”€ 1b. Mutual Information â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
mi_scores = mutual_info_classif(X_fs_sample, y_fs_sample, random_state=SEED)
mi_df = pd.DataFrame({'feature': fs_num_cols, 'MI_score': mi_scores})
mi_df = mi_df.sort_values('MI_score', ascending=False).reset_index(drop=True)
top_mi_feats = mi_df.head(20)['feature'].tolist()
print(f"\n[1b] Mutual Information â€“ Top 15:")
print(mi_df.head(15).to_string(index=False))

# â”€â”€â”€ 1c. Chi-square test (cá»™t phÃ¢n loáº¡i Ä‘Ã£ encode, giÃ¡ trá»‹ â‰¥ 0) â”€â”€â”€â”€â”€â”€â”€â”€â”€
ord_cols_chi2 = [c for c in fs_num_cols if c.endswith(
    '_ord') or c.endswith('_freq')]
if not ord_cols_chi2:
    ord_cols_chi2 = [c for c in fs_num_cols if X_fs_sample[c].min() >= 0][:30]

print(
    f"\n[1c] Chi-square test â€“ {len(ord_cols_chi2)} cá»™t phÃ¢n loáº¡i encoded (giÃ¡ trá»‹ â‰¥ 0):")
if ord_cols_chi2:
    X_chi2 = X_fs_sample[ord_cols_chi2].fillna(0)
    X_chi2 = X_chi2 - X_chi2.min()          # Ä‘áº£m báº£o â‰¥ 0
    chi2_scores, chi2_pvals = chi2(X_chi2, y_fs_sample)
    chi2_df = pd.DataFrame({'feature': ord_cols_chi2,
                            'chi2_stat': chi2_scores,
                            'p_value': chi2_pvals}
                           ).sort_values('chi2_stat', ascending=False).reset_index(drop=True)
    print(chi2_df.head(15).to_string(index=False))
    top_chi2_feats = chi2_df[chi2_df['p_value'] < 0.05]['feature'].tolist()
    print(f"  â†’ {len(top_chi2_feats)} cá»™t cÃ³ p < 0.05")
else:
    chi2_df = pd.DataFrame(columns=['feature', 'chi2_stat', 'p_value'])
    top_chi2_feats = []
    print("  KhÃ´ng cÃ³ cá»™t phÃ¹ há»£p â€“ bá» qua.")

# %%
# Biá»ƒu Ä‘á»“ so sÃ¡nh 3 phÆ°Æ¡ng phÃ¡p lá»c thá»‘ng kÃª (3 subplot)
fig, axes = plt.subplots(1, 3, figsize=(20, 6))

anova_df.head(20).plot(x='feature', y='F_stat', kind='barh',
                       ax=axes[0], color='steelblue', legend=False)
axes[0].set_title('ANOVA F-test (Top 20)', fontsize=11)
axes[0].set_xlabel('F-statistic')

mi_df.head(20).plot(x='feature', y='MI_score', kind='barh',
                    ax=axes[1], color='darkorange', legend=False)
axes[1].set_title('Mutual Information (Top 20)', fontsize=11)
axes[1].set_xlabel('MI Score')

if not chi2_df.empty:
    chi2_df.head(20).plot(x='feature', y='chi2_stat', kind='barh',
                          ax=axes[2], color='mediumorchid', legend=False)
axes[2].set_title('Chi-square test (Top 20)', fontsize=11)
axes[2].set_xlabel('Chi2 Statistic')

plt.suptitle(
    'Táº§ng 1 â€“ Lá»c thá»‘ng kÃª: ANOVA Â· Mutual Information Â· Chi-square', fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'fig_08_statistical_filters.png'),
            dpi=100, bbox_inches='tight')
plt.show()

# Union top-20 tá»« 3 phÆ°Æ¡ng phÃ¡p â†’ táº­p Ä‘áº·c trÆ°ng lá»c thá»‘ng kÃª
filter_union_feats = list(set(top_anova_feats) | set(
    top_mi_feats) | set(top_chi2_feats))
print(
    f"\nâ†’ Union top-20 tá»« 3 phÆ°Æ¡ng phÃ¡p lá»c: {len(filter_union_feats)} Ä‘áº·c trÆ°ng")

# %% [markdown]
# ### Táº§ng 2: Lá»c dá»±a trÃªn mÃ´ hÃ¬nh (RF Â· GB Â· RFE vá»›i Cross-Validation)

# %%
print("\n" + "="*60)
print("Táº¦NG 2: Lá»ŒC Dá»°A TRÃŠN MÃ” HÃŒNH")
print("="*60)

# â”€â”€â”€ 2a. Random Forest Feature Importance â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
print("\n[2a] Random Forest Importance (n_estimators=100, max_depth=6)...")
rf = RandomForestClassifier(
    n_estimators=100, max_depth=6, n_jobs=-1, random_state=SEED)
rf.fit(X_fs_sample, y_fs_sample)
rf_importance = pd.DataFrame({'feature': fs_num_cols,
                              'RF_importance': rf.feature_importances_}
                             ).sort_values('RF_importance', ascending=False).reset_index(drop=True)
top_rf_feats = rf_importance.head(20)['feature'].tolist()
print("RF â€“ Top 15 Ä‘áº·c trÆ°ng:")
print(rf_importance.head(15).to_string(index=False))

# â”€â”€â”€ 2b. Gradient Boosting Feature Importance â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
print("\n[2b] Gradient Boosting Importance (n_estimators=100, max_depth=4)...")
gb = GradientBoostingClassifier(
    n_estimators=100, max_depth=4, random_state=SEED)
gb.fit(X_fs_sample, y_fs_sample)
gb_importance = pd.DataFrame({'feature': fs_num_cols,
                              'GB_importance': gb.feature_importances_}
                             ).sort_values('GB_importance', ascending=False).reset_index(drop=True)
top_gb_feats = gb_importance.head(20)['feature'].tolist()
print("GB â€“ Top 15 Ä‘áº·c trÆ°ng:")
print(gb_importance.head(15).to_string(index=False))

# %%
# Biá»ƒu Ä‘á»“ RF vs GB importance (cÃ¹ng thang Ä‘o)
fig, axes = plt.subplots(1, 2, figsize=(18, 8))
rf_importance.head(20).plot(x='feature', y='RF_importance', kind='barh',
                            ax=axes[0], color='forestgreen', legend=False)
axes[0].set_title('Random Forest Feature Importance (Top 20)', fontsize=12)
axes[0].set_xlabel('Importance (Gini)')

gb_importance.head(20).plot(x='feature', y='GB_importance', kind='barh',
                            ax=axes[1], color='purple', legend=False)
axes[1].set_title('Gradient Boosting Feature Importance (Top 20)', fontsize=12)
axes[1].set_xlabel('Importance')

plt.suptitle(
    'Táº§ng 2 â€“ Model-based: RF vs GB Feature Importance', fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'fig_09_model_importance.png'),
            dpi=100, bbox_inches='tight')
plt.show()

# %%
# â”€â”€â”€ 2c. RFE vá»›i Logistic Regression (5-fold CV) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# LÃ½ do subsample: RFE vá»›i LR cÃ³ Ä‘á»™ phá»©c táº¡p O(nÂ·dÂ·kÂ²) trong k vÃ²ng láº·p
# â†’ 30k dÃ²ng Ã— 400+ cá»™t Ã— 5-fold sáº½ máº¥t hÃ ng giá»; 3000 dÃ²ng Ä‘á»§ á»•n Ä‘á»‹nh
print("\n[2c] RFE vá»›i Logistic Regression (5-fold CV, subsample=3000)...")
print("     LÃ½ do subsample: RFE cÃ³ O(nÂ·dÂ·kÂ²) â†’ 3000 dÃ²ng Ä‘á»§ Ä‘á»ƒ Ä‘Ã¡nh giÃ¡ phÃ¢n loáº¡i.")

RFE_N = min(3000, len(X_fs_sample))
idx_rfe = X_fs_sample.sample(RFE_N, random_state=SEED).index
X_rfe = X_fs_sample.loc[idx_rfe]
y_rfe = y_fs_sample.loc[idx_rfe]

cv_5fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
N_FEATS_RFE = [5, 10, 15, 20]
rfe_f1_scores = {}

for n in N_FEATS_RFE:
    rfe_est = RFE(
        estimator=LogisticRegression(max_iter=300, solver='saga', C=0.1,
                                     class_weight='balanced', random_state=SEED),
        n_features_to_select=n
    )
    pipe = Pipeline([
        ('rfe', rfe_est),
        ('clf', LogisticRegression(max_iter=300, solver='saga', C=0.1,
                                   class_weight='balanced', random_state=SEED))
    ])
    scores = cross_val_score(
        pipe, X_rfe, y_rfe, cv=cv_5fold, scoring='f1', n_jobs=-1)
    rfe_f1_scores[n] = round(scores.mean(), 4)
    print(
        f"  n_features={n:3d}: F1={scores.mean():.4f} +/- {scores.std():.4f}")

best_rfe_n = max(rfe_f1_scores, key=rfe_f1_scores.get)
print(
    f"\nâ†’ RFE tá»‘t nháº¥t: n_features={best_rfe_n}, F1={rfe_f1_scores[best_rfe_n]:.4f}")

# %%
# Bar chart: CV F1-score theo sá»‘ lÆ°á»£ng Ä‘áº·c trÆ°ng RFE
fig, ax = plt.subplots(figsize=(7, 4))
ns = list(rfe_f1_scores.keys())
f1s = [rfe_f1_scores[n] for n in ns]
bars = ax.bar([str(n) for n in ns], f1s, color='steelblue', edgecolor='white')
ax.bar([str(best_rfe_n)], [rfe_f1_scores[best_rfe_n]], color='tomato', edgecolor='white',
       label=f'Best: n={best_rfe_n}')
for bar, val in zip(bars, f1s):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.002, f'{val:.4f}',
            ha='center', va='bottom', fontsize=9)
ax.set_xlabel('Sá»‘ Ä‘áº·c trÆ°ng (n_features_to_select)')
ax.set_ylabel('CV F1-score (mean, 5-fold)')
ax.set_title(
    'RFE vá»›i Logistic Regression â€” CV F1 theo sá»‘ Ä‘áº·c trÆ°ng', fontsize=12)
ax.set_ylim(0, max(f1s) * 1.15)
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'fig_09b_rfe_f1.png'),
            dpi=100, bbox_inches='tight')
plt.show()

# %% [markdown]
# ### Táº§ng 3: Giáº£m chiá»u (PCA Â· t-SNE Â· UMAP)

# %%
print("\n" + "="*60)
print("Táº¦NG 3: GIáº¢M CHIá»€U")
print("="*60)

# â”€â”€â”€ 3a. PCA: cumulative explained variance â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
print("\n[3a] PCA â€“ giá»¯ 95% phÆ°Æ¡ng sai...")
pca_full = PCA(n_components=min(100, X_fs_sample.shape[1]), random_state=SEED)
pca_full.fit(X_fs_sample.fillna(0))
cumvar = np.cumsum(pca_full.explained_variance_ratio_)
n_comp_95 = int((cumvar >= 0.95).argmax()) + 1
n_comp_99 = int((cumvar >= 0.99).argmax()) + 1
print(
    f"  Sá»‘ thÃ nh pháº§n cáº§n Ä‘á»ƒ giá»¯ 95% variance: {n_comp_95} / {X_fs_sample.shape[1]}")
print(
    f"  Sá»‘ thÃ nh pháº§n cáº§n Ä‘á»ƒ giá»¯ 99% variance: {n_comp_99} / {X_fs_sample.shape[1]}")

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(range(1, len(cumvar)+1), cumvar, marker='.',
        color='steelblue', linewidth=1.5)
ax.axhline(0.95, color='red', linestyle='--',
           label=f'95% variance â†’ {n_comp_95} thÃ nh pháº§n')
ax.axhline(0.99, color='darkorange', linestyle='--',
           label=f'99% variance â†’ {n_comp_99} thÃ nh pháº§n')
ax.set_xlabel('Sá»‘ thÃ nh pháº§n PCA')
ax.set_ylabel('Cumulative Explained Variance Ratio')
ax.set_title('PCA â€“ PhÆ°Æ¡ng sai giáº£i thÃ­ch tÃ­ch lÅ©y', fontsize=12)
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'fig_10_pca_cumvar.png'),
            dpi=100, bbox_inches='tight')
plt.show()

# %%
# â”€â”€â”€ 3b. t-SNE 2D scatter: Fraud vs Normal â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
print("\n[3b] t-SNE 2D (n=3000)...")
TSNE_N = min(3000, len(X_fs_sample))
idx_tsne = X_fs_sample.sample(TSNE_N, random_state=SEED).index
X_tsne_in = X_fs_sample.loc[idx_tsne].fillna(0)
y_tsne = y_fs_sample.loc[idx_tsne]

tsne = TSNE(n_components=2, perplexity=30, random_state=SEED, max_iter=500)
X_tsne = tsne.fit_transform(X_tsne_in)

fig, ax = plt.subplots(figsize=(10, 7))
ax.scatter(X_tsne[y_tsne == 0, 0], X_tsne[y_tsne == 0, 1],
           c='steelblue', alpha=0.4, s=5, label=f'Normal (n={int((y_tsne == 0).sum())})')
ax.scatter(X_tsne[y_tsne == 1, 0], X_tsne[y_tsne == 1, 1],
           c='tomato', alpha=0.8, s=15, label=f'Fraud  (n={int((y_tsne == 1).sum())})')
ax.set_title(
    f't-SNE 2D (n={TSNE_N}): PhÃ¢n tÃ¡ch Fraud vs Normal', fontsize=12)
ax.legend(markerscale=3)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'fig_11_tsne.png'),
            dpi=100, bbox_inches='tight')
plt.show()
print("  â†’ t-SNE hoÃ n thÃ nh.")

# %%
# â”€â”€â”€ 3c. UMAP (náº¿u Ä‘Æ°á»£c cÃ i Ä‘áº·t) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
try:
    import umap as umap_lib
    print("\n[3c] UMAP 2D Ä‘ang cháº¡y...")
    reducer = umap_lib.UMAP(
        n_components=2, random_state=SEED, n_neighbors=15, min_dist=0.1)
    X_umap = reducer.fit_transform(X_tsne_in)

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.scatter(X_umap[y_tsne == 0, 0], X_umap[y_tsne == 0, 1],
               c='steelblue', alpha=0.4, s=5, label=f'Normal (n={int((y_tsne == 0).sum())})')
    ax.scatter(X_umap[y_tsne == 1, 0], X_umap[y_tsne == 1, 1],
               c='tomato', alpha=0.8, s=15, label=f'Fraud  (n={int((y_tsne == 1).sum())})')
    ax.set_title(
        f'UMAP 2D (n={TSNE_N}): PhÃ¢n tÃ¡ch Fraud vs Normal', fontsize=12)
    ax.legend(markerscale=3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig_12_umap.png'),
                dpi=100, bbox_inches='tight')
    plt.show()
    print("  â†’ UMAP hoÃ n thÃ nh.")
except ImportError:
    print("  umap-learn chÆ°a cÃ i â€” bá» qua UMAP. (pip install umap-learn)")
except Exception as _umap_err:
    print(f"  UMAP lá»—i: {_umap_err}")

# %% [markdown]
# ### Tá»•ng há»£p káº¿t quáº£ Feature Selection â€“ Lá»±a chá»n cuá»‘i
#
# | PhÆ°Æ¡ng phÃ¡p | Sá»‘ Ä‘áº·c trÆ°ng | CÆ¡ sá»Ÿ Ä‘Ã¡nh giÃ¡ |
# |---|---|---|
# | Filter â€“ ANOVA Top20 | 20 | F-statistic so vá»›i target |
# | Filter â€“ MI Top20 | 20 | Mutual Information (phi tuyáº¿n) |
# | Filter â€“ Union(3) | ~40-60 | Há»£p táº¥t cáº£ filter methods |
# | Model â€“ RF Top20 | 20 | Gini importance |
# | Model â€“ GB Top20 | 20 | Gradient Boosting importance |
# | RFE(LR) best-n | best_rfe_n | 5-fold CV F1 tá»‘i Ä‘a |
#
# **Chiáº¿n lÆ°á»£c chá»n cuá»‘i:** Union RF+GB top-20 (káº¿t há»£p 2 model-based biá»‡n há»™ nháº¥t),
# bá»• sung thÃªm Ä‘áº·c trÆ°ng tá»« top-ANOVA Ä‘á»ƒ Ä‘áº£m báº£o coverage.

# %%
print("\n" + "=" * 60)
print("Tá»”NG Há»¢P FEATURE SELECTION â€“ Lá»°A CHá»ŒN CUá»I")
print("=" * 60)

# Tá»•ng há»£p sá»‘ lÆ°á»£ng Ä‘áº·c trÆ°ng tá»« má»—i level
fs_summary_counts = {
    'Filter â€“ ANOVA Top20': len(top_anova_feats),
    'Filter â€“ MI Top20': len(top_mi_feats),
    'Filter â€“ Union(3 methods)': len(filter_union_feats),
    'Model â€“ RF Top20': len(top_rf_feats),
    'Model â€“ GB Top20': len(top_gb_feats),
    f'RFE(LR) n={best_rfe_n}': best_rfe_n,
}
for label, cnt in fs_summary_counts.items():
    print(f"  {label:<35}: {cnt} Ä‘áº·c trÆ°ng")

print(
    f"\nâ†’ RFE tá»‘t nháº¥t: n_features={best_rfe_n}, 5-fold CV F1={rfe_f1_scores[best_rfe_n]:.4f}")

# Lá»±a chá»n cuá»‘i: union RF + GB Ä‘á»ƒ káº¿t há»£p 2 model-based approaches
FINAL_FEATURES = list(set(top_rf_feats) | set(top_gb_feats))
# Loáº¡i cÃ¡c cá»™t khÃ´ng cÃ²n tá»“n táº¡i trong train (sau encoding)
FINAL_FEATURES = [c for c in FINAL_FEATURES if c in train.columns]
winner = 'RF+GB Union (model-based)'
print(
    f"\nâ†’ FINAL FEATURE SET [{winner}]: {len(FINAL_FEATURES)} Ä‘áº·c trÆ°ng")
print(f"  VÃ­ dá»¥: {FINAL_FEATURES[:5]}")

# %%
# RFE F1 progression chart (Â§4e â€” CV F1-score chart theo sá»‘ lÆ°á»£ng Ä‘áº·c trÆ°ng)
fig, ax = plt.subplots(figsize=(8, 4))
ns_rfe = sorted(rfe_f1_scores.keys())
f1s_rfe = [rfe_f1_scores[n] for n in ns_rfe]
ax.plot(ns_rfe, f1s_rfe, marker='o',
        color='steelblue', linewidth=2, markersize=8)
ax.axvline(best_rfe_n, color='red', linestyle='--', alpha=0.7,
           label=f'Best n={best_rfe_n} (F1={rfe_f1_scores[best_rfe_n]:.4f})')
for n, f1 in zip(ns_rfe, f1s_rfe):
    ax.annotate(f'{f1:.3f}', (n, f1), textcoords='offset points', xytext=(0, 8),
                ha='center', fontsize=9)
ax.set_xlabel('Sá»‘ lÆ°á»£ng Ä‘áº·c trÆ°ng (k)')
ax.set_ylabel('5-fold CV F1 score')
ax.set_title(
    'RFE â€“ F1 Score theo Sá»‘ LÆ°á»£ng Äáº·c TrÆ°ng (Logistic Regression)', fontsize=12)
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'fig_13_rfe_f1_curve.png'),
            dpi=100, bbox_inches='tight')
plt.show()

# %% [markdown]
# ---
# ## 2.2.3f. Xá»­ lÃ½ máº¥t cÃ¢n báº±ng lá»›p â€” SMOTE / ADASYN / Random Under-sampling
#
# Dataset IEEE-CIS Fraud Detection cÃ³ tá»· lá»‡ fraud ráº¥t tháº¥p (~3.5%), gÃ¢y ra
# **class imbalance**: mÃ´ hÃ¬nh cÃ³ xu hÆ°á»›ng dá»± Ä‘oÃ¡n táº¥t cáº£ lÃ  Normal vÃ  Ä‘áº¡t accuracy cao
# nhÆ°ng F1(Fraud) tháº¥p.
#
# ### CÃ¡c phÆ°Æ¡ng phÃ¡p xá»­ lÃ½
#
# | PhÆ°Æ¡ng phÃ¡p | CÆ¡ cháº¿ | Sinh dá»¯ liá»‡u má»›i? |
# |---|---|---|
# | **KhÃ´ng xá»­ lÃ½ (baseline)** | Train trá»±c tiáº¿p | KhÃ´ng |
# | **SMOTE** | Táº¡o synthetic minority samples theo K-NN interpolation | CÃ³ |
# | **ADASYN** | NhÆ° SMOTE nhÆ°ng táº­p trung vÃ o vÃ¹ng khÃ³ há»c | CÃ³ |
# | **Random Under-sampling (RUS)** | Giáº£m majority class ngáº«u nhiÃªn | KhÃ´ng |
#
# ### NguyÃªn lÃ½ SMOTE:
# $$x_{new} = x_i + \lambda \cdot (x_{knn} - x_i), \quad \lambda \sim \text{Uniform}(0, 1)$$
#
# ### Quy táº¯c an toÃ n:
# > **Chá»‰ Ã¡p dá»¥ng resampling trÃªn táº­p TRAIN.** Test set pháº£i pháº£n Ã¡nh phÃ¢n phá»‘i thá»±c (imbalanced).
# > Resampling trÃªn test lÃ m káº¿t quáº£ P/R/F1 khÃ´ng Ä‘Ãºng thá»±c táº¿ triá»ƒn khai.
#
# **ÄÃ¡nh giÃ¡:** Precision, Recall, **F1-macro**, **AUC-ROC** trÃªn val set (khÃ´ng resampled).

# %%

# Sá»­ dá»¥ng FINAL_FEATURES Ä‘á»ƒ Ä‘áº£m báº£o Ä‘á»§ Ä‘áº·c trÆ°ng vÃ  khÃ´ng cÃ³ NaN
fs_cols_clean = [
    c for c in FINAL_FEATURES if c in train.columns and train[c].dtype != object]
X_imb = train[fs_cols_clean].fillna(0).values
y_imb = train['isFraud'].values

fraud_count = int(y_imb.sum())
normal_count = int((y_imb == 0).sum())
imbalance_ratio = normal_count / fraud_count

print(f"PhÃ¢n phá»‘i lá»›p trong táº­p train:")
print(
    f"  isFraud=0 (Normal): {normal_count:,}  ({normal_count/(normal_count+fraud_count):.2%})")
print(
    f"  isFraud=1 (Fraud):  {fraud_count:,}  ({fraud_count/(normal_count+fraud_count):.2%})")
print(f"  Imbalance ratio: {imbalance_ratio:.1f}x  (Normal / Fraud)")
print(f"  Sá»‘ Ä‘áº·c trÆ°ng: {X_imb.shape[1]}")

# PhÃ¢n chia train/val theo tá»‰ lá»‡ 80/20 (stratified Ä‘á»ƒ giá»¯ tá»· lá»‡ fraud)
# âš ï¸ QUAN TRá»ŒNG: chá»‰ resampling trÃªn X_tr, y_tr â€“ KHÃ”NG trÃªn X_val, y_val
X_tr, X_val, y_tr, y_val = train_test_split(
    X_imb, y_imb, test_size=0.20, random_state=SEED, stratify=y_imb
)
print(f"\nSplit: train_sub={len(X_tr):,}  |  val={len(X_val):,}")
print(f"Tá»· lá»‡ Fraud trong val (khÃ´ng resampled): {y_val.mean():.4f}")

# %%
# So sÃ¡nh 4 chiáº¿n lÆ°á»£c trÃªn LR baseline
resampling_configs = {
    'No Resampling (baseline)': (X_tr, y_tr),
}

# SMOTE
try:
    sm = SMOTE(random_state=SEED, k_neighbors=min(5, fraud_count - 1))
    X_sm, y_sm = sm.fit_resample(X_tr, y_tr)
    resampling_configs['SMOTE'] = (X_sm, y_sm)
    print(
        f"SMOTE:  {int((y_sm == 1).sum()):,} Fraud / {int((y_sm == 0).sum()):,} Normal")
except Exception as e:
    print(f"SMOTE lá»—i: {e}")

# ADASYN
try:
    ada = ADASYN(random_state=SEED, n_neighbors=min(5, fraud_count - 1))
    X_ada, y_ada = ada.fit_resample(X_tr, y_tr)
    resampling_configs['ADASYN'] = (X_ada, y_ada)
    print(
        f"ADASYN: {int((y_ada == 1).sum()):,} Fraud / {int((y_ada == 0).sum()):,} Normal")
except Exception as e:
    print(f"ADASYN lá»—i: {e}")

# Random Under-sampling
rus = RandomUnderSampler(random_state=SEED)
X_rus, y_rus = rus.fit_resample(X_tr, y_tr)
resampling_configs['Random Under-sampling'] = (X_rus, y_rus)
print(
    f"RUS:    {int((y_rus == 1).sum()):,} Fraud / {int((y_rus == 0).sum()):,} Normal")

# %%
# Train LR vÃ  Ä‘Ã¡nh giÃ¡ trÃªn val (imbalanced)
resampling_results = {}
clf_lr = LogisticRegression(max_iter=300, solver='saga', C=0.1,
                            class_weight=None, random_state=SEED)

print("\n" + "=" * 70)
print(f"{'PhÆ°Æ¡ng phÃ¡p':<30} {'Precision':>10} {'Recall':>8} {'F1-macro':>10} {'AUC-ROC':>9}")
print("-" * 70)

for name, (X_r, y_r) in resampling_configs.items():
    clf_lr.fit(X_r, y_r)
    y_pred = clf_lr.predict(X_val)
    y_prob = clf_lr.predict_proba(X_val)[:, 1]
    p = round(precision_score(y_val, y_pred, zero_division=0), 4)
    r = round(recall_score(y_val, y_pred, zero_division=0), 4)
    f1 = round(f1_score(y_val, y_pred, average='macro', zero_division=0), 4)
    auc = round(roc_auc_score(y_val, y_prob), 4)
    resampling_results[name] = {'Precision': p,
                                'Recall': r, 'F1-macro': f1, 'AUC-ROC': auc}
    print(f"{name:<30} {p:>10.4f} {r:>8.4f} {f1:>10.4f} {auc:>9.4f}")

print("=" * 70)

resamp_df = pd.DataFrame(resampling_results).T
best_resamp = resamp_df['F1-macro'].idxmax()
print(f"\nâ†’ PhÆ°Æ¡ng phÃ¡p tá»‘t nháº¥t (F1-macro): {best_resamp}")
print(f"  â†’ LÃ½ giáº£i: SMOTE/ADASYN giÃºp model há»c boundary Fraud tá»‘t hÆ¡n")
print(f"    (tÄƒng Recall); val set luÃ´n dÃ¹ng phÃ¢n phá»‘i thá»±c (khÃ´ng resampled).")

# %%
# Biá»ƒu Ä‘á»“ so sÃ¡nh
metrics_plot = ['Precision', 'Recall', 'F1-macro', 'AUC-ROC']
fig, axes = plt.subplots(1, len(metrics_plot), figsize=(18, 5))
colors_rs = ['steelblue', 'tomato', 'darkorange', 'forestgreen']

for ax, metric in zip(axes, metrics_plot):
    vals = [resampling_results[m][metric] for m in resampling_results]
    bars = ax.bar(list(resampling_results.keys()), vals,
                  color=colors_rs[:len(resampling_results)], edgecolor='white', alpha=0.85)
    ax.set_title(metric, fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.tick_params(axis='x', rotation=20, labelsize=8)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01, f'{v:.3f}',
                ha='center', fontsize=8)
    ax.grid(axis='y', alpha=0.3)

plt.suptitle('So sÃ¡nh chiáº¿n lÆ°á»£c xá»­ lÃ½ máº¥t cÃ¢n báº±ng lá»›p\n'
             f'(ÄÃ¡nh giÃ¡ trÃªn val set chÆ°a resampled, n={len(X_val):,})', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'fig_14_imbalance_comparison.png'),
            dpi=100, bbox_inches='tight')
plt.show()

# %% [markdown]
# ---
# ## 2.2.3g. LÆ°u káº¿t quáº£ xá»­ lÃ½ (Pipeline Output)
#
# LÆ°u láº¡i:
# - `X_train_processed.npy` / `y_train.npy` â€” Ä‘áº·c trÆ°ng Ä‘Ã£ xá»­ lÃ½ (FINAL_FEATURES)
# - `X_test_processed.npy` â€” táº­p test xá»­ lÃ½ tÆ°Æ¡ng á»©ng
# - `feature_names.npy` â€” tÃªn cÃ¡c Ä‘áº·c trÆ°ng Ä‘Ã£ chá»n
# - `pipeline_choices.json` â€” cÃ¡c quyáº¿t Ä‘á»‹nh cá»§a toÃ n bá»™ pipeline

# %%

# ThÆ° má»¥c processed
PROCESSED_DIR = os.path.join(os.path.dirname(OUTPUT_DIR), 'processed')
os.makedirs(PROCESSED_DIR, exist_ok=True)

# Feature matrix cuá»‘i (chá»‰ FINAL_FEATURES tá»“n táº¡i trong train & test)
final_cols = [
    c for c in FINAL_FEATURES if c in train.columns and c in test.columns]
X_train_final = train[final_cols].fillna(0).values.astype(np.float32)
y_train_final = train['isFraud'].values.astype(np.int8)
X_test_final = test[final_cols].fillna(0).values.astype(np.float32)

np.save(os.path.join(PROCESSED_DIR, 'X_train_processed.npy'), X_train_final)
np.save(os.path.join(PROCESSED_DIR, 'y_train.npy'),           y_train_final)
np.save(os.path.join(PROCESSED_DIR, 'X_test_processed.npy'),  X_test_final)
np.save(os.path.join(PROCESSED_DIR, 'feature_names.npy'),     np.array(final_cols))

pipeline_choices = {
    'imputation': {
        'benchmark_best': str(best_strategy),
        'production_choice': 'Median',
        'reason': 'Memory-safe for 590k rows x 400 cols; RMSE comparable to kNN/MICE'
    },
    'outlier': {
        'ks_best_method': str(best_ks),
        'production_choice': 'IQR_clipping',
        'reason': 'Báº£o toÃ n sá»‘ dÃ²ng; deterministic; O(nÂ·d) scalable'
    },
    'scaling': {
        'benchmark': list(scalers.keys()),
        'production_choice': 'RobustScaler',
        'reason': 'Bá»n vá»¯ng vá»›i outlier; phÃ¹ há»£p 399/400 cá»™t phi chuáº©n'
    },
    'encoding': {
        'low_card': 'OneHotEncoding',
        'high_card': 'BinaryEncoding',
        'all_cols': 'Ordinal + Target(5-fold CV) + Frequency'
    },
    'feature_selection': {
        'winner': winner,
        'n_features': len(final_cols),
        'rfe_best_n': int(best_rfe_n),
        'rfe_best_f1': float(rfe_f1_scores[best_rfe_n])
    },
    'imbalance': {
        'imbalance_ratio': round(float(imbalance_ratio), 2),
        'best_method': best_resamp,
        'eval_metric': 'F1-macro on unsampled val set'
    }
}

with open(os.path.join(PROCESSED_DIR, 'pipeline_choices.json'), 'w', encoding='utf-8') as f:
    json.dump(pipeline_choices, f, ensure_ascii=False, indent=2)

print(f"âœ… ÄÃ£ lÆ°u dá»¯ liá»‡u xá»­ lÃ½ vÃ o: {PROCESSED_DIR}")
print(
    f"   X_train_processed.npy : shape={X_train_final.shape}, dtype={X_train_final.dtype}")
print(
    f"   y_train.npy           : shape={y_train_final.shape}, fraud_rate={y_train_final.mean():.4f}")
print(f"   X_test_processed.npy  : shape={X_test_final.shape}")
print(f"   feature_names.npy     : {len(final_cols)} Ä‘áº·c trÆ°ng")
print(f"   pipeline_choices.json : {list(pipeline_choices.keys())}")
print(f"\n=== HOÃ€N THÃ€NH TOÃ€N Bá»˜ PIPELINE TIá»€N Xá»¬ LÃ TABULAR ===")
