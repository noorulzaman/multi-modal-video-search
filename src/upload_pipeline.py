
from pathlib import Path

import cv2
import faiss
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm


ROOT = Path(__file__).resolve().parent.parent


def extract_frames(video_path, output_dir, interval_seconds=1):
    """Extract one frame at each sampling interval."""

    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(video_path))

    if not capture.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS))
    reported_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 0:
        capture.release()
        raise ValueError("The video has an invalid or unavailable frame rate.")

    duration = reported_frames / fps if reported_frames > 0 else None
    records = []
    frame_index = 0
    next_sample_time = 0.0

    try:
        with tqdm(
            total=reported_frames if reported_frames > 0 else None,
            desc="Extracting frames",
            unit="frame",
        ) as progress:
            while True:
                success, frame = capture.read()

                if not success:
                    break

                timestamp = frame_index / fps

                if timestamp + 1e-6 >= next_sample_time:
                    # Exclude fully black frames.
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                    if int(gray.max()) > 0:
                        height, width = frame.shape[:2]
                        max_width = 512

                        if width > max_width:
                            scale = max_width / width
                            frame = cv2.resize(
                                frame,
                                (
                                    max_width,
                                    max(1, round(height * scale)),
                                ),
                                interpolation=cv2.INTER_AREA,
                            )

                        frame_name = (
                            f"frame_{timestamp:09.2f}s.jpg"
                        )
                        frame_path = output_dir / frame_name

                        saved = cv2.imwrite(
                            str(frame_path),
                            frame,
                            [cv2.IMWRITE_JPEG_QUALITY, 90],
                        )

                        if saved:
                            records.append(
                                {
                                    "video_id": video_path.stem,
                                    "video": video_path.name,
                                    "timestamp_seconds": round(
                                        timestamp, 3
                                    ),
                                    "frame_path": str(
                                        frame_path.relative_to(ROOT)
                                    ),
                                }
                            )

                    # Move to the next sampling time.
                    while next_sample_time <= timestamp + 1e-6:
                        next_sample_time += interval_seconds

                frame_index += 1
                progress.update(1)
    finally:
        capture.release()

    if not records:
        raise ValueError(
            "No usable frames were extracted from this video."
        )

    manifest = pd.DataFrame(records)
    manifest_path = output_dir.parent / "frames_manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    return manifest, fps, duration, manifest_path


def embed_frames(
    manifest,
    model,
    preprocess,
    device,
    batch_size=32,
):
    """Generate normalized CLIP embeddings for extracted frames."""

    embedding_batches = []
    metadata_records = []
    frame_records = manifest.to_dict(orient="records")

    for start in tqdm(
        range(0, len(frame_records), batch_size),
        desc="Embedding frames",
        unit="batch",
    ):
        batch_records = frame_records[start : start + batch_size]
        tensors = []
        valid_records = []

        for record in batch_records:
            path = ROOT / record["frame_path"]

            try:
                with Image.open(path) as image:
                    tensor = preprocess(image.convert("RGB"))

                tensors.append(tensor)
                valid_records.append(record)
            except (OSError, ValueError) as exc:
                print(f"[SKIP] Could not read {path}: {exc}")

        if not tensors:
            continue

        batch = torch.stack(tensors).to(device)

        with torch.inference_mode():
            features = model.encode_image(batch)
            features = features / features.norm(
                dim=-1, keepdim=True
            ).clamp_min(1e-12)

        vectors = features.float().cpu().numpy()

        # Defensive validation before adding vectors to the index.
        valid_mask = np.isfinite(vectors).all(axis=1)

        for vector, record, valid in zip(
            vectors, valid_records, valid_mask
        ):
            if valid:
                embedding_batches.append(vector)
                metadata_records.append(record)
            else:
                print(
                    f"[SKIP] Non-finite embedding: "
                    f"{record['frame_path']}"
                )

    if not embedding_batches:
        raise ValueError("No valid CLIP embeddings were generated.")

    embeddings = np.asarray(
        embedding_batches, dtype=np.float32
    )

    metadata = pd.DataFrame(metadata_records)

    if len(embeddings) != len(metadata):
        raise ValueError(
            "Embedding count does not match metadata row count."
        )

    return embeddings, metadata


def build_faiss_index(embeddings):
    """Build an exact cosine-similarity index."""

    if embeddings.ndim != 2 or embeddings.shape[1] != 512:
        raise ValueError(
            f"Expected embeddings with shape (N, 512), "
            f"received {embeddings.shape}."
        )

    if not np.isfinite(embeddings).all():
        raise ValueError("Embeddings contain NaN or infinity.")

    # OpenCLIP vectors are normalized before indexing.
    norms = np.linalg.norm(embeddings, axis=1)

    if not np.allclose(norms, 1.0, atol=1e-3):
        raise ValueError("Embeddings are not L2-normalized.")

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(np.ascontiguousarray(embeddings))

    return index


def process_uploaded_video(
    video_path,
    model,
    preprocess,
    device,
    interval_seconds=1,
    batch_size=32,
):
    """Extract, embed, and index a single uploaded video."""

    video_path = Path(video_path).resolve()

    if not video_path.is_file():
        raise FileNotFoundError(
            f"Uploaded video not found: {video_path}"
        )

    if interval_seconds < 1:
        raise ValueError("Sampling interval must be at least 1 second.")

    # Use the file hash in its saved name to keep each upload isolated.
    upload_id = video_path.stem.rsplit("_", 1)[-1]
    processing_dir = ROOT / "data" / "processed" / upload_id
    frame_dir = processing_dir / "frames"
    artifact_dir = ROOT / "artifacts" / "uploaded" / upload_id

    frame_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    manifest, fps, duration, manifest_path = extract_frames(
        video_path,
        frame_dir,
        interval_seconds=interval_seconds,
    )

    embeddings, metadata = embed_frames(
        manifest,
        model,
        preprocess,
        device,
        batch_size=batch_size,
    )

    index = build_faiss_index(embeddings)

    embeddings_path = artifact_dir / "frame_embeddings.npy"
    metadata_path = artifact_dir / "frame_metadata.csv"
    index_path = artifact_dir / "frames.faiss"

    np.save(embeddings_path, embeddings)
    metadata.to_csv(metadata_path, index=False)
    faiss.write_index(index, str(index_path))

    return {
        "video_name": video_path.name,
        "duration_seconds": duration,
        "fps": fps,
        "frames_extracted": len(manifest),
        "frames_embedded": len(metadata),
        "embedding_shape": embeddings.shape,
        "device": device,
        "manifest_path": manifest_path,
        "embeddings_path": embeddings_path,
        "metadata_path": metadata_path,
        "index_path": index_path,
    }
