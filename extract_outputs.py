import nbformat, json

nb = nbformat.read('Source/notebooks/05_text_preprocessing_executed.ipynb', as_version=4)

def get_text_output(cell):
    result = []
    for out in cell.get('outputs', []):
        if out.get('output_type') in ('stream', 'execute_result', 'display_data'):
            text = out.get('text', '') or ''.join(out.get('data', {}).get('text/plain', ''))
            if text.strip():
                result.append(text.strip())
    return '\n'.join(result)

# Find cells by patterns and print their text outputs
line_ranges_of_interest = [
    ("BASIC_STATS", 255, 267),        # text length stats
    ("MANN_WHITNEY", 361, 421),        # Mann-Whitney results
    ("TTR", 537, 563),                 # TTR results
    ("ZIPF_MAIN", 578, 602),           # Zipf main analysis
    ("ZIPF_PLOT_PRINT", 659, 735),     # Zipf print with per-label alpha
    ("PIPELINE_STEP_TABLE", 756, 829), # Pipeline per-step table
    ("TOKENIZATION", 925, 969),        # Tokenization comparison
    ("STOP_VOCAB", 972, 1014),         # Stop words vocab/token stats
    ("MI_NB", 1103, 1173),             # MI mean + NB F1
    ("STEM_COLLISION", 1226, 1319),    # Stemming collision + F1
    ("COSINE", 1456, 1492),            # Cosine similarity
    ("SILHOUETTE", 1553, 1634),        # Silhouette + Friedman for vectorization
    ("KMEANS", 1664, 1687),            # K-Means silhouette
    ("SVM_F1", 1690, 1716),            # SVM F1 comparison
    ("LEADERBOARD_TABLE", 1884, 1925), # All results leaderboard
]

for label, start, end in line_ranges_of_interest:
    for cell in nb.cells:
        if cell.cell_type == 'code':
            src = cell.source
            matched = False
            if label == "BASIC_STATS" and "Thống kê độ dài text" in src: matched = True
            elif label == "MANN_WHITNEY" and "MANN-WHITNEY U TEST" in src: matched = True
            elif label == "TTR" and "TYPE-TOKEN RATIO" in src: matched = True
            elif label == "ZIPF_MAIN" and "PHÂN TÍCH ĐỊNH LUẬT ZIPF" in src: matched = True
            elif label == "ZIPF_PLOT_PRINT" and "_zipf_slopes" in src: matched = True
            elif label == "PIPELINE_STEP_TABLE" and "PIPELINE_STEP_TABLE" in src and "_step_rows" in src: matched = True
            elif label == "TOKENIZATION" and "tokenization_results" in src and "SO SÁNH" in src: matched = True
            elif label == "STOP_VOCAB" and "SO SÁNH TRƯỚC VÀ SAU KHI XÓA STOP WORDS" in src: matched = True
            elif label == "MI_NB" and "MI TRUNG BÌNH" in src: matched = True
            elif label == "STEM_COLLISION" and "SO SÁNH STEMMING" in src: matched = True
            elif label == "COSINE" and "COSINE SIMILARITY" in src: matched = True
            elif label == "SILHOUETTE" and "Bootstrap silhouette" in src: matched = True
            elif label == "KMEANS" and "K-MEANS CLUSTERING" in src: matched = True
            elif label == "SVM_F1" and "LINEAR SVM" in src: matched = True
            elif label == "LEADERBOARD_TABLE" and "BẢNG TỔNG HỢP" in src: matched = True
            
            if matched:
                out = get_text_output(cell)
                if out:
                    print(f"\n{'='*60}")
                    print(f"SECTION: {label}")
                    print('='*60)
                    print(out[:3000])
                    break
