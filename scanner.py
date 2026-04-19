import re

files = [
    r'Source/notebooks/01_EDA_image.py',
    r'Source/notebooks/02_preprocessing_image.py', 
    r'Source/notebooks/03_advanced_image.py',
    r'Source/notebooks/04_tabular_preprocessing.py',
    r'Source/notebooks/05_text_preprocessing.py',
]

for fpath in files:
    try:
        with open(fpath, 'r', encoding='utf-8') as f:
            content = f.read()
    except FileNotFoundError:
        print(f'\nFile not found: {fpath}')
        continue
    
    print(f'\n{"="*60}')
    print(f'FILE: {fpath}')
    print('='*60)
    
    lines = content.split('\n')
    in_markdown = False
    cell_lines = []
    cell_start = 0
    
    for i, line in enumerate(lines):
        if line.startswith('# %% [markdown]'):
            in_markdown = True
            cell_lines = []
            cell_start = i + 1
        elif (line.startswith('# %%') or i == len(lines) - 1) and in_markdown:
            if i == len(lines) - 1:
                cell_lines.append(line)
            if cell_lines:
                cell_text = '\n'.join(cell_lines)
                patterns = [
                    r'\=[\d.]+\$',
                    r'\=[\d.]+\$',
                    r'\\eta\^2.*=.*[\d.]+',
                    r'\|\s*[\d.]+\s*\|',
                    r'= 0\.\d{3}',
                ]
                hits = []
                for pat in patterns:
                    if re.search(pat, cell_text):
                        hits.append(pat)
                if hits:
                    print(f'\n  Line {cell_start}: SUSPICIOUS MARKDOWN CELL')
                    for cl in cell_lines[:25]:
                        print(f'    {cl}')
                    if len(cell_lines) > 25:
                        print(f'    ... ({len(cell_lines)-25} more lines)')
                    print(f'  Patterns matched: {hits}')
            in_markdown = False
            cell_lines = []
        elif in_markdown:
            cell_lines.append(line)

print('\nScan complete.')
