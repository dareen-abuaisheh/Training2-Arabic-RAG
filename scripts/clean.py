"""Conservative cleaning for Chandra text, with optional reviewed repairs."""

import argparse
import hashlib
import re
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "data" / "extracted"
OUTPUT_DIR = ROOT / "data" / "cleaned"

PAGE_MARKER = re.compile(r"^\{(\d+)\}-{3,}[ \t\r]*$", re.MULTILINE)
# Preserve tabs and line/page boundaries. Do not remove Arabic directional marks
# or joining characters: they can affect how text is displayed.
INVALID_CONTROLS = re.compile(r"[\x00-\x08\x0e-\x1f\x7f]")
FENCE_START = re.compile(r"^ {0,3}(`{3,}|~{3,})(.*)$")


def normalize_blank_lines(text: str) -> str:
    """Reduce blank runs outside code; preserve meaningful Markdown spacing."""
    output = []
    pending_blanks = []
    fence = None
    previous_indented = False
    lines = text.split("\n")
    if text.endswith("\n"):
        lines.pop()  # The final newline is added once when joining below.

    for line in lines:
        if fence is not None:
            output.append(line)
            character, length = fence
            if re.fullmatch(rf" {{0,3}}{character}{{{length},}}[ \t]*", line):
                fence = None
            continue

        if not line.strip(" \t"):
            pending_blanks.append(line)
            continue

        indented = line.startswith(("    ", "\t"))
        if pending_blanks and (output or indented):
            # Indentation can mark code or nested lists: preserve nearby blanks.
            if previous_indented or indented:
                output.extend(pending_blanks)
            else:
                output.append("")
        pending_blanks = []
        output.append(line)
        previous_indented = indented

        match = FENCE_START.match(line)
        if match and not (match[1][0] == "`" and "`" in match[2]):
            fence = (match[1][0], len(match[1]))

    if previous_indented:
        output.extend(pending_blanks)
    return "\n".join(output) + "\n" if output else ""


def clean_text(text: str) -> str:
    """Normalize representation and spacing without deleting document content."""
    text = text.lstrip("\ufeff").replace("\r\n", "\n").replace("\r", "\n")
    # Preserve boundaries rather than joining words across page/control breaks.
    text = text.replace("\x0c", "\n\n").replace("\x0b", "\n")
    text = INVALID_CONTROLS.sub("", text)
    # NFC combines equivalent Unicode sequences. It does not strip Arabic
    # diacritics, stem words, remove stopwords, or standardize letter variants.
    text = unicodedata.normalize("NFC", text)
    cleaned = normalize_blank_lines(text)

    # General cleaning must preserve all non-whitespace content.
    if re.sub(r"\s", "", cleaned) != re.sub(r"\s", "", text):
        raise ValueError("Document content changed during general cleaning.")
    return cleaned


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reviewed-fixes",
        action="store_true",
        help="Also apply separate, reviewed repairs to exact matching extractions.",
    )
    args = parser.parse_args()
    if args.reviewed_fixes:
        from reviewed_fixes import apply_reviewed_fixes

    source_files = sorted(SOURCE_DIR.glob("*.txt"))
    if not source_files:
        raise SystemExit("No extracted TXT files found in data/extracted/.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for source in source_files:
        raw_bytes = source.read_bytes()
        original = raw_bytes.decode("utf-8-sig").replace("\r\n", "\n").replace("\r", "\n")
        working = original
        details = "general cleaning only"
        if args.reviewed_fixes:
            working, counts = apply_reviewed_fixes(
                working, hashlib.sha256(raw_bytes).hexdigest()
            )
            if counts:
                details = (
                    f"reviewed repairs: {counts['images']} image links, "
                    f"{counts['captions']} captions, {counts['diagrams']} diagrams, "
                    f"{counts['titles']} titles"
                )
            else:
                details = "general cleaning; no reviewed repairs for this source"

        cleaned = clean_text(working)
        if not cleaned.strip():
            raise SystemExit(f"Empty text: {source.name}. Nothing saved for this file.")
        if PAGE_MARKER.findall(cleaned) != PAGE_MARKER.findall(original):
            raise ValueError(f"Page markers changed: {source.name}. Output not saved.")

        output = OUTPUT_DIR / source.name
        output.write_text(cleaned, encoding="utf-8")
        print(f"Saved: data/cleaned/{source.name} | {details}")


if __name__ == "__main__":
    main()
