import nbformat
nb = nbformat.read("Source/notebooks/05_text_preprocessing_executed.ipynb", as_version=4)

def get_output(cell):
    output_parts = []
    for o in cell.get("outputs", []):
        if o.get("output_type") == "stream":
            output_parts.append("".join(o.get("text", "")))
        elif o.get("output_type") == "execute_result":
            data = o.get("data", {})
            if "text/plain" in data:
                output_parts.append(data["text/plain"])
    return "".join(output_parts)

targets = [
    ("MANN_WHITNEY_FULL", "MANN-WHITNEY U TEST + EFFECT SIZE"),
    ("PIPELINE_TABLE_FULL", "BẢNG THỐNG KÊ PER-STEP PIPELINE"),
    ("TOKENIZATION_FULL", "SO SÁNH CÁC PHƯƠNG PHÁP TOKENIZATION"),
    ("STOP_WORDS_VOCAB_FULL", "SO SÁNH TRƯỚC VÀ SAU KHI XÓA STOP WORDS"),
    ("MI_NB_FULL", "MI TRUNG BÌNH TRƯỚC/SAU XÓA STOP WORDS"),
    ("STEM_FULL", "SO SÁNH STEMMING VÀ LEMMATIZATION"),
    ("STEM_SCORES", "None (baseline)"),
    ("COSINE_FULL", "COSINE SIMILARITY (trên sample"),
    ("SIL_FULL", "SILHOUETTE SCORE (đánh giá"),
    ("SIL_BOOT_FULL", "Silhouette bootstrap mean"),
    ("KMEANS_FULL", "K-MEANS CLUSTERING (k=2)"),
    ("SVM_FULL", "LINEAR SVM: TF-IDF vs Sentence Transformer"),
    ("LEADERBOARD_FULL", "BẢNG TỔNG HỢP HIỆU NĂNG PHÂN LOẠI"),
]

for label, key in targets:
    for cell in nb.cells:
        if cell.cell_type == "code" and key in cell.source:
            out = get_output(cell)
            if out.strip():
                print(f"\n{'='*70}")
                print(f"[{label}]")
                print(out)
                break
