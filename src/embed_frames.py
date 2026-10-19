
from pathlib import Path

import numpy as np
import open_clip
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm


ROOT = Path(__file__).resolve().parent.parent
FRAME_MANIFEST = ROOT / "evaluation" / "frames_manifest.csv"
OUTPUT_DIR = ROOT / "artifacts"
EMBEDDINGS_FILE = OUTPUT_DIR / "frame_embeddings.npy"
METADATA_FILE = OUTPUT_DIR / "frame_metadata.csv"

MODEL_NAME = "ViT-B-32"
PRETRAINED = "laion2b_s34b_b79k"
BATCH_SIZE = 32


def main() -> None:
    if not FRAME_MANIFEST.exists():
        raise FileNotFoundError(
            f"Missing frame manifest: {FRAME_MANIFEST}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    model, _, preprocess = open_clip.create_model_and_transforms(
        MODEL_NAME,
        pretrained=PRETRAINED,
        device=device,
    )
    model.eval()

    manifest = pd.read_csv(FRAME_MANIFEST)
    if manifest.empty:
        raise ValueError("The frame manifest contains no frames.")

    all_embeddings = []
    valid_metadata = []
    failed_frames = []

    for start in tqdm(
        range(0, len(manifest), BATCH_SIZE),
        desc="Embedding frames",
    ):
        batch_rows = manifest.iloc[start : start + BATCH_SIZE]
        images = []
        batch_metadata = []

        for _, row in batch_rows.iterrows():
            frame_path = ROOT / row["frame_path"]

            try:
                with Image.open(frame_path) as image:
                    images.append(preprocess(image.convert("RGB")))
                batch_metadata.append(row.to_dict())
            except Exception as exc:
                failed_frames.append(
                    {
                        "frame_path": str(row["frame_path"]),
                        "error": str(exc),
                    }
                )

        if not images:
            continue

        image_tensor = torch.stack(images).to(device)

        with torch.inference_mode():
            features = model.encode_image(image_tensor)
            features = features / features.norm(
                dim=-1, keepdim=True
            ).clamp_min(1e-12)

        embeddings = features.cpu().numpy().astype(np.float32)

        all_embeddings.append(embeddings)
        valid_metadata.extend(batch_metadata)

    if not all_embeddings:
        raise RuntimeError("No frame embeddings were generated.")

    embedding_matrix = np.concatenate(all_embeddings, axis=0)
    metadata = pd.DataFrame(valid_metadata)

    if len(embedding_matrix) != len(metadata):
        raise RuntimeError(
            "Embedding and metadata counts do not match."
        )

    np.save(EMBEDDINGS_FILE, embedding_matrix)
    metadata.to_csv(METADATA_FILE, index=False)

    print("\n--- CLIP embedding summary ---")
    print(f"Model:                 {MODEL_NAME}")
    print(f"Pretrained weights:    {PRETRAINED}")
    print(f"Device:                {device}")
    print(f"Frames in manifest:    {len(manifest)}")
    print(f"Successful embeddings: {len(embedding_matrix)}")
    print(f"Failed frames:         {len(failed_frames)}")
    print(f"Embedding shape:       {embedding_matrix.shape}")
    print(f"Embedding dtype:       {embedding_matrix.dtype}")
    print(f"Embeddings saved:      {EMBEDDINGS_FILE}")
    print(f"Metadata saved:        {METADATA_FILE}")

    if failed_frames:
        failed_path = OUTPUT_DIR / "failed_frames.csv"
        pd.DataFrame(failed_frames).to_csv(failed_path, index=False)
        print(f"Failure details saved: {failed_path}")

    if device == "cuda":
        print(
            f"Peak allocated GPU memory: "
            f"{torch.cuda.max_memory_allocated() / 1024**2:.1f} MB"
        )


if __name__ == "__main__":
    main()
