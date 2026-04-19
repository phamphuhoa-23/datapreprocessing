import re

files = [
    'Source/notebooks/01_EDA_image.py',
    'Source/notebooks/02_preprocessing_image.py',
    'Source/notebooks/03_advanced_image.py',
    'Source/notebooks/04_tabular_preprocessing.py',
    'Source/notebooks/05_text_preprocessing.py',
]

PATTERNS = [
    r'print\([f]?["\'].*=>\s',
    r'print\([f]?["\']=>',
    r'if\s+\w*[pP]\w*\s*[<>].*:\s*$',
    r'print\(.*[Bb]ác bỏ|print\(.*[Cc]hấp nhận',
    r'print\(.*effect size',
    r'print\(.*\blớn\b.*\btrung bình\b|\btrung bình\b.*\bnhỏ\b',
    r'lớn.*trung bình.*nhỏ|nhỏ.*trung bình.*lớn',
]

for fpath in files:
    try:
        with open(fpath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        print(f"\n{'='*70}")
        print(f"FILE: {fpath}")
        print('='*70)
        
        in_code = True  # treat all non-markdown as code
        for i, line in enumerate(lines):
            stripped = line.rstrip()
            if stripped.startswith('# %% [markdown]'):
                in_code = False
            elif stripped == '# %%' or re.match(r'^# %% ', stripped) and '[markdown]' not in stripped:
                in_code = True
            
            if not in_code:
                continue
            if line.strip().startswith('#'):
                continue
                
            for pat in PATTERNS:
                if re.search(pat, line, re.IGNORECASE):
                    # show context: line-1, line, line+1, line+2
                    start = max(0, i-1)
                    end = min(len(lines), i+5)
                    print(f"\n  --- L{i+1}: MATCH ---")
                    for j in range(start, end):
                        marker = ">>>" if j == i else "   "
                        print(f"  {marker} L{j+1}: {lines[j].rstrip()}")
                    break
    except FileNotFoundError:
        print(f"\nFILE NOT FOUND: {fpath}")

print('\nDone.')
