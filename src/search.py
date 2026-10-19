
from pathlib import Path

import faiss
import numpy as np
import open_clip
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = ROOT / "artifacts"

INDEX_FILE = ARTIFACTS_DIR / "frames.faiss"
METADATA_FILE = ARTIFACTS_DIR / "frame_metadata.csv"

MODEL_NAME = "ViT-B-32"
PRETRAINED = "laion2b_s34b_b79k"


def load_model():
    """Load the same CLIP model used to embed the frames."""

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, _, _ = open_clip.create_model_and_transforms(
        MODEL_NAME,
        pretrained=PRETRAINED,
        device=device,
    )
    tokenizer = open_clip.get_tokenizer(MODEL_NAME)
    model.eval()

    print(f"Using device: {device}")
    return model, tokenizer, device


def encode_text(query, model, tokenizer, device):
    """Encode a text query into a normalized CLIP vector."""

    tokens = tokenizer([query]).to(device)

    with torch.inference_mode():
        features = model.encode_text(tokens)
        features = features / features.norm(
            dim=-1, keepdim=True
        ).clamp_min(1e-12)

    return features.cpu().numpy().astype(np.float32)


def search(query, top_k=5):
    """Return the best matching videos and individual frames."""

    if not INDEX_FILE.exists():
        raise FileNotFoundError(
            f"FAISS index not found: {INDEX_FILE}"
        )

    if not METADATA_FILE.exists():
        raise FileNotFoundError(
            f"Frame metadata not found: {METADATA_FILE}"
        )

    index = faiss.read_index(str(INDEX_FILE))
    metadata = pd.read_csv(METADATA_FILE)

    if index.ntotal != len(metadata):
        raise ValueError(
            "FAISS index size does not match frame metadata."
        )

    model, tokenizer, device = load_model()
    query_vector = encode_text(query, model, tokenizer, device)

    # Search all indexed frames so video aggregation considers
    # every frame, not just the top few frame matches.
    scores, indices = index.search(
        query_vector,
        index.ntotal,
    )

    results = metadata.iloc[indices[0]].copy()
    results["similarity"] = scores[0]

    # Rank videos by their strongest matching frame.
    # This is a baseline aggregation strategy.
    video_results = (
        results.groupby(["video_id", "video"], as_index=False)
        .agg(
            similarity=("similarity", "max"),
            best_timestamp=("timestamp_seconds", "first"),
            matching_frames=("similarity", "size"),
        )
        .sort_values("similarity", ascending=False)
        .head(top_k)
        .reset_index(drop=True)
    )

    # Find the actual best frame for each video, ensuring its
    # timestamp and score correspond to that frame.
    best_frames = (
        results.sort_values("similarity", ascending=False)
        .drop_duplicates(subset=["video_id"])
        [["video_id", "timestamp_seconds", "frame_path", "similarity"]]
    )

    video_results = video_results.drop(
        columns=["best_timestamp", "similarity"]
    ).merge(
        best_frames.rename(
            columns={
                "timestamp_seconds": "best_timestamp",
                "frame_path": "best_frame_path",
                "similarity": "similarity",
            }
        ),
        on="video_id",
        how="left",
    ).sort_values("similarity", ascending=False)

    return video_results, results.head(10)


def main():
    query = input("Enter a video search query: ").strip()

    if not query:
        print("Please enter a non-empty query.")
        return

    video_results, frame_results = search(query, top_k=5)

    print("\n--- Top matching videos ---")
    print(video_results.to_string(index=False))

    print("\n--- Top matching frames ---")
    print(
        frame_results[
            ["video_id", "timestamp_seconds", "similarity", "frame_path"]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
