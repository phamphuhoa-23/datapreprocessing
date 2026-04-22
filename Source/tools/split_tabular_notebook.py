from __future__ import annotations

from pathlib import Path


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8", newline="\n")


def main() -> int:
    """
    Split `notebooks/04_tabular_preprocessing.py` into:
    - `notebooks/03_EDA_tabular.py` (load + EDA)
    - `notebooks/04_preprocessing_tabular.py` (load + preprocessing + save outputs)

    We keep "load & merge" in BOTH notebooks so each runs standalone.
    """
    source_root = Path(__file__).resolve().parent.parent
    src = source_root / "notebooks" / "04_tabular_preprocessing.py"
    if not src.exists():
        raise FileNotFoundError(f"Missing source notebook: {src}")

    text = _read_text(src)
    lines = text.splitlines(True)  # keep newlines

    def find_line_index(startswith: str) -> int:
        for i, line in enumerate(lines):
            if line.startswith(startswith):
                return i
        raise RuntimeError(f"Marker not found: {startswith!r}")

    # Markers (verified by grep)
    idx_eda = find_line_index("# ## Phân tích thống kê khám phá (EDA)")
    idx_pre = find_line_index("# ## 2.2.3a. Xử lý giá trị thiếu có kiểm soát – So sánh 5 chiến lược")

    # EDA notebook: everything up to (but excluding) preprocessing section
    eda_lines = lines[:idx_pre]

    # Preprocessing notebook: keep header + imports + load section (everything up to EDA),
    # then append preprocessing section onward.
    pre_lines = lines[:idx_eda] + lines[idx_pre:]

    out_eda = source_root / "notebooks" / "03_EDA_tabular.py"
    out_pre = source_root / "notebooks" / "04_preprocessing_tabular.py"

    _write_text(out_eda, "".join(eda_lines))
    _write_text(out_pre, "".join(pre_lines))

    print(f"[OK] Wrote: {out_eda}")
    print(f"[OK] Wrote: {out_pre}")
    print("[NOTE] Original file kept as reference:", src)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

