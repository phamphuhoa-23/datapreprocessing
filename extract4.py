# -*- coding: utf-8 -*-
import nbformat
nb = nbformat.read('Source/notebooks/05_text_preprocessing_executed.ipynb', as_version=4)

def get_all_text(cell):
    parts = []
    for o in cell.get('outputs', []):
        otype = o.get('output_type', '')
        if otype == 'stream' and o.get('name') == 'stdout':
            parts.append(''.join(o.get('text', '')))
        elif otype in ('execute_result', 'display_data'):
            txt = o.get('data', {}).get('text/plain', '')
            if txt: parts.append(txt if isinstance(txt, str) else ''.join(txt))
    return '\n'.join(parts)

targets = [
    ("TOK", "SO SÁNH CÁC PHƯƠNG PHÁP TOKENIZATION"),
    ("STEM_COLLISION", "Tổng từ unique (input)"),
    ("COSINE", "COSINE SIMILARITY (trên sample"),
    ("SIL_MAIN", "SILHOUETTE SCORE (đánh giá chất lượng"),
    ("KMEANS_SIL", "K-MEANS CLUSTERING (k=2)"),
    ("SVM_SCORES", "LINEAR SVM: TF-IDF vs Sentence"),
    ("ZIPF_ALPHA", "Zipf α:"),
    ("BASIC_STATS", "Thống kê độ dài text (ký tự)"),
]

for label, key in targets:
    for cell in nb.cells:
        if cell.cell_type == 'code' and key in cell.source:
            out = get_all_text(cell)
            if out.strip():
                print(f"\n{'='*70}")
                print(f"[{label}]")
                print(out[:3000])
                break
