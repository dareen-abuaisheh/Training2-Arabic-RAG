"""Create local E5 embeddings and keep each vector linked to its chunk ID."""

import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel

from chunk import (
    MODEL_LIMIT,
    MODEL_NAME,
    MODEL_REVISION,
    PASSAGE_PREFIX,
    count_tokens,
    load_tokenizer,
)

ROOT = Path(__file__).resolve().parents[1]
CHUNKS_PATH = ROOT / "data" / "chunks.jsonl"
OUTPUT_PATH = ROOT / "data" / "embeddings.npz"
BATCH_SIZE = 4


def load_chunks() -> tuple[list[dict], str]:
    """Read chunks once and fingerprint the exact input file for later search."""
    raw_bytes = CHUNKS_PATH.read_bytes()
    chunks = [json.loads(line) for line in raw_bytes.decode("utf-8").splitlines() if line.strip()]
    if not chunks:
        raise ValueError("No chunks found. Run scripts/chunk.py first.")
    return chunks, hashlib.sha256(raw_bytes).hexdigest()


def prepare_inputs(chunks: list[dict], tokenizer) -> list[str]:
    """Validate IDs and token lengths before downloading or running the model."""
    inputs = []
    seen_ids = set()
    for chunk in chunks:
        if not isinstance(chunk, dict):
            raise ValueError("Each chunk must be a JSON object.")
        chunk_id, text = chunk.get("id"), chunk.get("text")
        if not isinstance(chunk_id, str) or not chunk_id.strip() or chunk_id in seen_ids:
            raise ValueError("Chunk IDs must be nonempty, unique strings.")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"Chunk {chunk_id} has no text.")
        seen_ids.add(chunk_id)
        actual_count = count_tokens(text, tokenizer)
        if actual_count > min(MODEL_LIMIT, tokenizer.model_max_length):
            raise ValueError(f"Chunk {chunk_id} exceeds the model's token limit.")
        if chunk.get("token_count") != actual_count:
            raise ValueError(f"Token count changed for {chunk_id}. Run scripts/chunk.py again.")
        inputs.append(PASSAGE_PREFIX + text.strip())
    return inputs


def load_model():
    """Reuse cached safe-format weights, downloading them on the first run."""
    options = {
        "revision": MODEL_REVISION,
        "cache_dir": ROOT / ".cache",
        "use_safetensors": True,
    }
    try:
        model = AutoModel.from_pretrained(MODEL_NAME, local_files_only=True, **options)
    except OSError:
        print("Downloading embedding model weights for the first run...", flush=True)
        model = AutoModel.from_pretrained(MODEL_NAME, **options)
    return model.to("cpu").eval()


def average_pool(hidden_states, attention_mask):
    """Combine real token vectors; padding must contribute nothing to the mean."""
    mask = attention_mask.unsqueeze(-1).bool()
    hidden_states = hidden_states.masked_fill(~mask, 0.0)
    return hidden_states.sum(dim=1) / mask.sum(dim=1).clamp(min=1)


def encode_texts(inputs: list[str], tokenizer, model, show_progress: bool = False) -> np.ndarray:
    """Embed already-prefixed inputs in small CPU batches without truncation."""
    if not inputs:
        raise ValueError("No input texts to embed.")
    batches = []
    with torch.inference_mode():
        for start in range(0, len(inputs), BATCH_SIZE):
            batch = tokenizer(
                inputs[start:start + BATCH_SIZE],
                padding=True,
                truncation=False,
                verbose=False,
                return_tensors="pt",
            )
            if batch["input_ids"].shape[1] > min(MODEL_LIMIT, tokenizer.model_max_length):
                raise ValueError("Embedding input exceeds the token limit; nothing was truncated.")
            output = model(**batch)
            pooled = average_pool(output.last_hidden_state, batch["attention_mask"])
            vectors = F.normalize(pooled, p=2, dim=1)
            batches.append(vectors.cpu().numpy())
            if show_progress:
                print(f"Embedded {min(start + BATCH_SIZE, len(inputs))}/{len(inputs)} texts.", flush=True)

    matrix = np.concatenate(batches).astype(np.float32, copy=False)
    if not np.isfinite(matrix).all() or not np.allclose(np.linalg.norm(matrix, axis=1), 1.0, atol=1e-5):
        raise ValueError("Invalid embedding vectors. No output was saved.")
    return matrix


def save_embeddings(vectors: np.ndarray, chunks: list[dict], chunks_hash: str) -> None:
    """Save one archive, replacing it only after the complete archive is written."""
    if hashlib.sha256(CHUNKS_PATH.read_bytes()).hexdigest() != chunks_hash:
        raise RuntimeError("Chunks changed during embedding. Rerun scripts/embed.py.")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=OUTPUT_PATH.parent, suffix=".npz", delete=False) as output:
            temporary = Path(output.name)
            np.savez_compressed(
                output,
                vectors=vectors,
                ids=np.array([chunk["id"] for chunk in chunks], dtype=str),
                model_name=np.array(MODEL_NAME),
                model_revision=np.array(MODEL_REVISION),
                chunks_sha256=np.array(chunks_hash),
            )
        temporary.replace(OUTPUT_PATH)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main() -> None:
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    chunks, chunks_hash = load_chunks()
    tokenizer = load_tokenizer()
    inputs = prepare_inputs(chunks, tokenizer)
    model = load_model()
    vectors = encode_texts(inputs, tokenizer, model, show_progress=True)
    if vectors.shape != (len(chunks), model.config.hidden_size):
        raise ValueError("Embedding dimensions do not match the model and chunks.")
    save_embeddings(vectors, chunks, chunks_hash)
    print(f"Saved: data/embeddings.npz | {vectors.shape[0]} vectors, {vectors.shape[1]} dimensions each.")


if __name__ == "__main__":
    main()
