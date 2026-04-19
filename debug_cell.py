import nbformat
nb = nbformat.read("Source/notebooks/05_text_preprocessing_executed.ipynb", as_version=4)

for cell in nb.cells:
    if cell.cell_type == "code" and "BẢNG TỔNG HỢP HIỆU NĂNG PHÂN LOẠI" in cell.source:
        print("--- SOURCE ---")
        print(cell.source)
        print("--- OUTPUT ---")
        for o in cell.get("outputs", []):
             if o.get("output_type") == "stream":
                 print(o.get("text", ""))
