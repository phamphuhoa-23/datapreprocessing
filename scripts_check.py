import re

files = [
    'Source/notebooks/01_EDA_image.py',
    'Source/notebooks/02_preprocessing_image.py',
    'Source/notebooks/03_advanced_image.py',
    'Source/notebooks/04_tabular_preprocessing.py',
    'Source/notebooks/05_text_preprocessing.py',
]

# Patterns that look like AI-style interpretation inside code cells
PATTERNS = [
    # if p < 0.05: print("=> ...")
    r'if\s+\w+_?p\w*\s*[<>]',
    # print("  =>")
    r'print\(["\'].*=>\s',
    # print("=> ...")
    r'print\(f?["\']=>',
    # lớn / nhỏ / trung bình interpretation
    r'print\(.*\blớn\b|\btrung bình\b|\bnhỏ\b',
    # effect size interpretation
    r'print\(.*effect\s+size',
    # "Bác bỏ" / "Chấp nhận" H0 in code
    r'print\(.*[Bb]ác\s+bỏ|[Cc]hấp\s+nhận',
    # "Kết luận" in print
    r'print\(.*[Kk]ết\s+luận',
]

for fpath in files:
    try:
        with open(fpath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        print(f"\n{'='*70}")
        print(f"FILE: {fpath}")
        print('='*70)
        
        in_code = False
        for i, line in enumerate(lines):
            stripped = line.rstrip()
            if stripped.startswith('# %% [markdown]'):
                in_code = False
            elif stripped.startswith('# %%'):
                in_code = True
            
            if in_code:
                for pat in PATTERNS:
                    if re.search(pat, line) and not line.strip().startswith('#'):
                        print(f"  L{i+1}: {line.rstrip()}")
                        break
    except FileNotFoundError:
        print(f"\nFILE NOT FOUND: {fpath}")

print('\nDone.')
