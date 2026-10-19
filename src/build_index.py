
from pathlib import Path

import faiss
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = ROOT / "artifacts"

EMBEDDINGS_FILE = ARTIFACTS_DIR / "frame_embeddings.npy"
METADATA_FILE = ARTIFACTS_DIR / "frame_metadata.csv"
INDEX_FILE = ARTIFACTS_DIR / "frames.faiss"


def main() -> None:
    if not EMBEDDINGS_FILE.exists():
        raise FileNotFoundError(
            f"Embeddings not found: {EMBEDDINGS_FILE}"
        )

    if not METADATA_FILE.exists():
        raise FileNotFoundError(
            f"Metadata not found: {METADATA_FILE}"
        )

    embeddings = np.load(EMBEDDINGS_FILE).astype(np.float32)
    metadata = pd.read_csv(METADATA_FILE)

    if embeddings.ndim != 2:
        raise ValueError(
            f"Expected a 2D embedding matrix, got {embeddings.shape}"
        )

    if len(embeddings) != len(metadata):
        raise ValueError(
            f"Embedding count ({len(embeddings)}) does not match "
            f"metadata count ({len(metadata)})"
        )

    if not np.isfinite(embeddings).all():
        raise ValueError("Embeddings contain NaN or infinite values.")

    norms = np.linalg.norm(embeddings, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-3):
        raise ValueError(
            "Embeddings are not L2-normalized. "
            "Check src/embed_frames.py."
        )

    dimension = embeddings.shape[1]

    # Exact cosine-similarity search on normalized vectors.
    index = faiss.IndexFlatIP(dimension)
    index.add(np.ascontiguousarray(embeddings))

    faiss.write_index(index, str(INDEX_FILE))

    print("\n--- FAISS index summary ---")
    print(f"Embedding vectors: {index.ntotal}")
    print(f"Dimensions:        {dimension}")
    print("Metric:            Inner product (cosine for normalized vectors)")
    print(f"Index type:        {type(index).__name__}")
    print(f"Index saved:       {INDEX_FILE}")


if __name__ == "__main__":
    main()
