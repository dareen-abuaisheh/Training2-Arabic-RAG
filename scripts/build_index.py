"""Build an exact cosine-similarity index from the existing E5 vectors."""

import tempfile
from pathlib import Path

import faiss
import numpy as np

from chunk import MODEL_NAME, MODEL_REVISION
from embed import OUTPUT_PATH as EMBEDDINGS_PATH, ROOT, load_chunks

INDEX_PATH = ROOT / "data" / "index.faiss"


def load_artifacts() -> tuple[list[dict], np.ndarray, list[str]]:
    """Reject stale embeddings before building an index or returning passages."""
    chunks, chunks_hash = load_chunks()
    with np.load(EMBEDDINGS_PATH, allow_pickle=False) as saved:
        if (
            str(saved["model_name"].item()) != MODEL_NAME
            or str(saved["model_revision"].item()) != MODEL_REVISION
            or str(saved["chunks_sha256"].item()) != chunks_hash
        ):
            raise ValueError("Chunks or model changed. Rerun embed.py, then build_index.py.")
        vectors = saved["vectors"]
        ids = saved["ids"].tolist()

    if ids != [chunk["id"] for chunk in chunks] or len(set(ids)) != len(ids):
        raise ValueError("Embedding IDs do not match the chunks. Rerun embed.py.")
    if (
        vectors.dtype != np.float32
        or vectors.ndim != 2
        or vectors.shape[0] != len(ids)
        or vectors.shape[1] == 0
        or not np.isfinite(vectors).all()
        or not np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5)
    ):
        raise ValueError("Expected one finite, normalized float32 vector per chunk.")
    return chunks, np.ascontiguousarray(vectors), ids


def load_index(vectors: np.ndarray):
    """Check that the stored FAISS rows still match our vector-to-ID mapping."""
    # Python handles Unicode paths, including Arabic folder names on Windows.
    index = faiss.deserialize_index(np.frombuffer(INDEX_PATH.read_bytes(), dtype=np.uint8))
    if (
        not isinstance(index, faiss.IndexFlatIP)
        or index.ntotal != vectors.shape[0]
        or index.d != vectors.shape[1]
        or not np.array_equal(index.reconstruct_n(0, index.ntotal), vectors)
    ):
        raise ValueError("Index does not match the embeddings. Rerun build_index.py.")
    return index


def main() -> None:
    _, vectors, _ = load_artifacts()
    # Inner product equals cosine similarity because all vectors have length 1.
    # Flat search compares every vector exactly; it needs no training step.
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=INDEX_PATH.parent, suffix=".faiss", delete=False) as output:
            temporary = Path(output.name)
            output.write(faiss.serialize_index(index).tobytes())
        temporary.replace(INDEX_PATH)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    print(f"Saved: data/index.faiss | {index.ntotal} vectors, {index.d} dimensions each.")


if __name__ == "__main__":
    main()
