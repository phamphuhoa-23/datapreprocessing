import nbformat
nb = nbformat.read('Source/notebooks/05_text_preprocessing_executed.ipynb', as_version=4)

def get_stdout(cell):
    parts = []
    for o in cell.get('outputs', []):
        if o.get('output_type') == 'stream' and o.get('name') == 'stdout':
            parts.append(''.join(o.get('text', '')))
    return ''.join(parts)

sections = {
    "MANN_WHITNEY": "MANN-WHITNEY U TEST",
    "ZIPF_MAIN": "PHÂN TÍCH ĐỊNH LUẬT ZIPF",
    "ZIPF_LABEL_PRINT": "_zipf_slopes",
    "PIPELINE_TABLE": "_step_rows",
    "TOKENIZATION": "SO SÁNH CÁC PHƯƠNG PHÁP TOKENIZATION",
    "STOP_VOCAB": "SO SÁNH TRƯỚC VÀ SAU KHI XÓA STOP WORDS",
    "MI_NB": "MI TRUNG BÌNH TRƯỚC/SAU",
    "STOP_NB_SCORE": "HIỆU NĂNG NAIVE BAYES",
    "STEM": "SO SÁNH STEMMING VÀ LEMMATIZATION",
    "STEM_RESULTS": "Kết luận: Phương pháp tốt nhất",
    "COSINE": "COSINE SIMILARITY",
    "SIL_MAIN": "SILHOUETTE SCORE",
    "SIL_BOOT": "Silhouette bootstrap mean",
    "KMEANS": "K-MEANS CLUSTERING",
    "SVM": "LINEAR SVM",
    "LEADERBOARD": "BẢNG TỔNG HỢP HIỆU NĂNG",
}

for label, key in sections.items():
    for cell in nb.cells:
        if cell.cell_type == 'code' and key in cell.source:
            out = get_stdout(cell)
            if out.strip():
                print(f"\n{'='*70}")
                print(f"[{label}]")
                print('='*70)
                print(out[:2500])
                break
