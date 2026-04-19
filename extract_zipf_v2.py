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

for cell in nb.cells:
    if cell.cell_type == 'code' and "α" in cell.source or "alpha" in cell.source.lower():
         out = get_all_text(cell)
         if out.strip():
             print(f"--- Cell Source ---\n{cell.source[:100]}\n--- Output ---")
             print(out)
