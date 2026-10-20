"""Split cleaned pages using the selected embedding model's tokenizer."""

import json
import re
from pathlib import Path

from transformers import AutoTokenizer

from load_pages import load_pages

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = ROOT / "data" / "chunks.jsonl"
MODEL_NAME = "intfloat/multilingual-e5-small"
MODEL_REVISION = "614241f622f53c4eeff9890bdc4f31cfecc418b3"
MODEL_LIMIT = 512
PASSAGE_PREFIX = "passage: "
MAX_TOKENS = 384
OVERLAP_TOKENS = 64

PARAGRAPH_BREAK = re.compile(r"\n[ \t]*\n")
SENTENCE_BREAK = re.compile(r"(?<=[.!?؟؛])\s+")
WORD_BREAK = re.compile(r"\s+")
HEADING = re.compile(r"#{1,6}[ \t]+[^\n]+")


def load_tokenizer():
    """Download/cache tokenizer files only; no embedding model weights yet."""
    cache_dir = ROOT / ".cache"
    try:
        return AutoTokenizer.from_pretrained(
            MODEL_NAME, revision=MODEL_REVISION, cache_dir=cache_dir, local_files_only=True
        )
    except OSError:
        return AutoTokenizer.from_pretrained(
            MODEL_NAME, revision=MODEL_REVISION, cache_dir=cache_dir
        )


def count_tokens(text: str, tokenizer, passage: bool = True) -> int:
    """Count the exact input we will embed, including its prefix and special tokens."""
    content = text.strip()
    if passage:
        content = PASSAGE_PREFIX + content
    encoded = tokenizer(
        content,
        add_special_tokens=passage,
        truncation=False,
        verbose=False,
    )
    return len(encoded["input_ids"])


def fitting_end(text: str, start: int, tokenizer, max_tokens: int) -> int:
    """Find a measured, safe character endpoint without decoding token slices."""
    if count_tokens(text[start:], tokenizer) <= max_tokens:
        return len(text)

    low, high = start + 1, len(text) - 1
    best = start
    while low <= high:
        middle = (low + high) // 2
        if count_tokens(text[start:middle], tokenizer) <= max_tokens:
            best = middle
            low = middle + 1
        else:
            high = middle - 1
    if best == start:
        raise ValueError("Token budget cannot fit the next source character.")
    return best


def choose_end(
    text: str, start: int, minimum: int, limit: int, tokenizer, max_tokens: int
) -> int:
    """Prefer paragraph, sentence, then word boundaries that fit the token budget."""
    for pattern in (PARAGRAPH_BREAK, SENTENCE_BREAK, WORD_BREAK):
        for match in reversed(list(pattern.finditer(text, minimum, limit))):
            if pattern is PARAGRAPH_BREAK:
                previous = PARAGRAPH_BREAK.split(text[:match.start()])[-1].strip()
                if HEADING.fullmatch(previous):
                    continue  # Keep a short heading with its following content.
            # Cutting at an earlier boundary can change tokenization: recheck it.
            if count_tokens(text[start:match.end()], tokenizer) <= max_tokens:
                return match.end()

    # A long unbroken string may require a cut inside a word. Original characters
    # are preserved; the endpoint was already measured by fitting_end().
    return limit


def overlap_start(text: str, start: int, end: int, tokenizer, overlap: int) -> int:
    """Keep a tail of whole words, within the overlap budget, with forward progress."""
    best = end
    if overlap == 0:
        return best
    for match in reversed(list(WORD_BREAK.finditer(text, start, end))):
        candidate = match.end()
        if candidate <= start:
            continue
        if count_tokens(text[candidate:end], tokenizer, passage=False) > overlap:
            break
        best = candidate
    return best


def chunk_spans(
    text: str, tokenizer, max_tokens: int = MAX_TOKENS, overlap: int = OVERLAP_TOKENS
) -> list[tuple[int, int]]:
    """Return source-text ranges, retaining all content and some shared context."""
    if not 0 <= overlap < max_tokens <= MODEL_LIMIT:
        raise ValueError("Require 0 <= overlap < max_tokens <= 512.")
    if max_tokens <= count_tokens("", tokenizer):
        raise ValueError("Token budget must have room for the passage prefix and text.")

    spans = []
    start = 0
    while start < len(text):
        limit = fitting_end(text, start, tokenizer, max_tokens)
        previous_end = spans[-1][1] if spans else start
        if limit <= previous_end:
            start = previous_end  # Drop overlap if it leaves no room for new text.
            continue
        if limit == len(text):
            end = limit
        else:
            minimum = max(start + (limit - start) // 2, previous_end + 1)
            end = choose_end(text, start, minimum, limit, tokenizer, max_tokens)

        spans.append((start, end))
        if end == len(text):
            break

        start = overlap_start(text, start, end, tokenizer, overlap)

    return spans


def build_chunks(pages: list[dict], tokenizer) -> list[dict]:
    """Keep every passage connected to its original PDF and physical page."""
    chunks = []
    for page in pages:
        index = 0
        for start, end in chunk_spans(page["text"], tokenizer):
            content = page["text"][start:end].strip()
            if not content:
                continue
            token_count = count_tokens(content, tokenizer)
            if token_count > min(MAX_TOKENS, MODEL_LIMIT, tokenizer.model_max_length):
                raise ValueError("Chunk exceeds the token budget. No output was saved.")
            index += 1
            chunks.append({
                "id": f"{Path(page['source']).stem}:p{page['page']}:c{index}",
                "source": page["source"],
                "page": page["page"],
                "text": content,
                "token_count": token_count,
            })
    return chunks


def main() -> None:
    pages = load_pages()
    tokenizer = load_tokenizer()
    chunks = build_chunks(pages, tokenizer)
    if not chunks:
        raise SystemExit("No text available for chunking. No output was saved.")

    # JSONL stores one JSON object per line; ensure_ascii=False keeps Arabic readable.
    with OUTPUT_PATH.open("w", encoding="utf-8", newline="\n") as output:
        for chunk in chunks:
            output.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    print(f"Created {len(chunks)} chunks from {len(pages)} nonempty pages.")
    print(f"Largest input: {max(c['token_count'] for c in chunks)} tokens, including passage prefix and special tokens.")
    print("Saved: data/chunks.jsonl")


if __name__ == "__main__":
    main()
