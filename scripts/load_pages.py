"""Load cleaned documents into page records with source information."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = ROOT / "data" / "cleaned"

PAGE_MARKER = re.compile(r"^\{(\d+)\}-{3,}[ \t]*$", re.MULTILINE)


def load_pages() -> list[dict]:
    """Read nonempty pages, preserving the PDF filename and physical page number."""
    source_files = sorted(INPUT_DIR.glob("*.txt"))
    if not source_files:
        raise FileNotFoundError("No cleaned TXT files found in data/cleaned/.")

    pages = []
    for path in source_files:
        text = path.read_text(encoding="utf-8")
        parts = PAGE_MARKER.split(text)

        if len(parts) == 1 or parts[0].strip():
            raise ValueError(f"Unexpected page-marker format: {path.name}")

        page_ids = [int(parts[i]) for i in range(1, len(parts), 2)]
        if page_ids != list(range(len(page_ids))):
            raise ValueError(f"Missing or unordered page markers: {path.name}")

        for i in range(1, len(parts), 2):
            content = parts[i + 1].strip()
            if not content:
                continue

            pages.append({
                "source": path.with_suffix(".pdf").name,
                "page": int(parts[i]) + 1,
                "text": content,
            })

    return pages


if __name__ == "__main__":
    pages = load_pages()
    print(f"Loaded {len(pages)} nonempty pages.")
