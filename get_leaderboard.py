import nbformat
nb = nbformat.read("Source/notebooks/05_text_preprocessing_executed.ipynb", as_version=4)

def get_stdout(cell):
    parts = []
    for o in cell.get("outputs", []):
        if o.get("output_type") == "stream" and o.get("name") == "stdout":
            parts.append("".join(o.get("text", "")))
    return "".join(parts)

# The leaderboard was truncated in the previous run. Let's find the cell containing the leaderboard table.
for cell in nb.cells:
    if cell.cell_type == "code" and "BẢNG TỔNG HỢP HIỆU NĂNG PHÂN LOẠI" in cell.source:
        print(get_stdout(cell))
