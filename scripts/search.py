"""Retrieve relevant Arabic passages, keeping their PDF and page references."""

import argparse
import os
import sys

import numpy as np
import torch

from build_index import load_artifacts, load_index
from chunk import MODEL_LIMIT, load_tokenizer
from embed import encode_texts, load_model

QUERY_PREFIX = "query: "


class Retriever:
    def __init__(self):
        chunks, vectors, self.ids = load_artifacts()
        self.chunks_by_id = {chunk["id"]: chunk for chunk in chunks}
        self.index = load_index(vectors)
        self.tokenizer = load_tokenizer()
        self.model = load_model()
        if self.model.config.hidden_size != self.index.d:
            raise ValueError("Query model dimensions do not match the index.")

    def search(self, question: str, top_k: int = 3, include_neighbors: bool = False) -> list[dict]:
        """Retrieve nearest passages, optionally appending context around the best hit."""
        if not question.strip():
            raise ValueError("Question cannot be empty.")
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
            raise ValueError("top_k must be a positive integer.")
        query = QUERY_PREFIX + question.strip()
        token_count = len(self.tokenizer(query, truncation=False, verbose=False)["input_ids"])
        if token_count > min(MODEL_LIMIT, self.tokenizer.model_max_length):
            raise ValueError("Question exceeds the model's token limit. Please shorten it.")

        # Reuse the passage model, pooling, and normalization; only the prefix changes.
        vector = encode_texts([query], self.tokenizer, self.model)
        scores, rows = self.index.search(vector, min(top_k, self.index.ntotal))
        matches = [(int(row), float(score)) for row, score in zip(rows[0], scores[0])]
        if include_neighbors:
            # Chunks are stored in document order. A neighboring chunk may finish
            # a section that crosses a page or chunk boundary. Never cross PDFs.
            best_row = matches[0][0]
            source = self.chunks_by_id[self.ids[best_row]]["source"]
            selected_rows = {row for row, _ in matches}
            for row in (best_row - 1, best_row + 1):
                if (
                    0 <= row < len(self.ids)
                    and row not in selected_rows
                    and self.chunks_by_id[self.ids[row]]["source"] == source
                ):
                    score = float(np.dot(vector[0], self.index.reconstruct(row)))
                    matches.append((row, score))
        results = []
        for row, score in matches:
            # FAISS row -> saved embedding ID -> original chunk with source metadata.
            chunk = self.chunks_by_id[self.ids[int(row)]]
            results.append({
                "id": chunk["id"],
                "source": chunk["source"],
                "page": chunk["page"],
                "score": float(score),
                "text": chunk["text"],
            })
        return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", nargs="?", help="Arabic question; prompted when omitted.")
    parser.add_argument("--top-k", type=int, default=3, help="Number of passages to retrieve (default: 3).")
    args = parser.parse_args()
    if args.top_k < 1:
        parser.error("--top-k must be positive.")
    sys.stdout.reconfigure(encoding="utf-8")
    question = args.question if args.question is not None else input("Question: ")
    if not question.strip():
        parser.error("Question cannot be empty.")
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    retriever = Retriever()
    for rank, result in enumerate(retriever.search(question, args.top_k), start=1):
        print(f"\n{rank}. {result['source']} | page {result['page']} | cosine {result['score']:.4f}")
        print(f"Chunk: {result['id']}")
        print(result["text"])


if __name__ == "__main__":
    main()
