"""Extract highlights and notes from FINALFEEDBACK PDF."""
import fitz  # pymupdf
from pathlib import Path

PDF_PATH = Path(__file__).parent / "CSC14004 - Data Mining FINALFEEDBACK.pdf"
OUT_PATH = Path(__file__).parent / "FINALFEEDBACK_ANNOTATIONS.md"

doc = fitz.open(str(PDF_PATH))

lines = [
    "# FINALFEEDBACK — Highlights & Notes trích xuất từ PDF",
    "",
    f"> File: `{PDF_PATH.name}`  ",
    f"> Tổng trang: {doc.page_count}",
    "",
]

for page_num, page in enumerate(doc, start=1):
    annots = list(page.annots())
    if not annots:
        continue

    lines.append(f"---")
    lines.append(f"## Trang {page_num}")
    lines.append("")

    for annot in annots:
        # e.g. 'Highlight', 'Text', 'FreeText', 'Ink', 'Underline'
        atype = annot.type[1]

        # Highlighted text
        if atype in ("Highlight", "Underline", "StrikeOut", "Squiggly"):
            # Get the highlighted text
            quads = annot.vertices
            highlighted_text = ""
            if quads:
                highlighted_text = page.get_textbox(annot.rect).strip()

            color = annot.colors.get("stroke") or annot.colors.get("fill")
            color_name = "🟡 YELLOW" if color and color[1] > 0.8 and color[0] > 0.8 else "🟢 GREEN"

            if highlighted_text:
                lines.append(f"**[{atype} {color_name}]**")
                lines.append(f"> {highlighted_text}")

            # Note attached to this highlight
            note = annot.info.get("content", "").strip()
            if note:
                lines.append(f"  💬 **NOTE:** {note}")
            lines.append("")

        elif atype in ("Text", "FreeText"):
            note = annot.info.get("content", "").strip()
            if note:
                lines.append(f"**[COMMENT]**")
                lines.append(f"> {note}")
                lines.append("")

        elif atype == "Ink":
            note = annot.info.get("content", "").strip()
            if note:
                lines.append(f"**[INK NOTE]**")
                lines.append(f"> {note}")
                lines.append("")

doc.close()

out_text = "\n".join(lines)
with open(str(OUT_PATH), "w", encoding="utf-8") as f:
    f.write(out_text)

print(f"Saved to: {OUT_PATH}")
print(f"Total lines: {len(lines)}")
