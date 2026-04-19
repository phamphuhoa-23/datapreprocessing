import nbformat
import pandas as pd # Just in case it's needed for repr
nb = nbformat.read("Source/notebooks/05_text_preprocessing_executed.ipynb", as_version=4)

for cell in nb.cells:
    if cell.cell_type == "code" and "BẢNG TỔNG HỢP HIỆU NĂNG PHÂN LOẠI" in cell.source:
        print("--- OUTPUT ---")
        for o in cell.get("outputs", []):
             if o.get("output_type") == "stream":
                 print(o.get("text", ""))
             elif o.get("output_type") == "execute_result":
                 data = o.get("data", {})
                 if "text/plain" in data:
                     print(data["text/plain"])
                 elif "text/html" in data:
                     print(data["text/html"])
