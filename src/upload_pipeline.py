from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import faiss
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent


def report_progress(callback, stage, current, total, message):
    """Report progress to the UI, if a callback was provided."""
    if callback is not None:
        callback(stage, current, total, message)


def extract_frames(
    video_path,
    output_dir,
    interval_seconds=1,
    progress_callback=None,
):
    """Seek to sample timestamps and save usable frames."""
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Remove stale frame images from an earlier processing run.
    for old_frame in output_dir.glob("frame_*.jpg"):
        old_frame.unlink()

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        reported_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

        if fps <= 0 or reported_frames <= 0:
            raise ValueError("The video has invalid frame-rate or duration metadata.")

        duration = reported_frames / fps
        sample_times = np.arange(
            0, duration, interval_seconds, dtype=np.float64
        )
        records = []

        for i, timestamp in enumerate(
            tqdm(sample_times, desc="Extracting frames", unit="frame")
        ):
            target_frame = min(
                int(round(timestamp * fps)), reported_frames - 1
            )
            capture.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
            success, frame = capture.read()

            if success:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # Skip completely black frames.
                if int(gray.max()) > 0:
                    height, width = frame.shape[:2]
                    max_width = 512

                    if width > max_width:
                        scale = max_width / width
                        frame = cv2.resize(
                            frame,
                            (max_width, max(1, round(height * scale))),
                            interpolation=cv2.INTER_AREA,
                        )

                    frame_name = f"frame_{timestamp:09.2f}s.jpg"
                    frame_path = output_dir / frame_name

                    if cv2.imwrite(
                        str(frame_path),
                        frame,
                        [cv2.IMWRITE_JPEG_QUALITY, 90],
                    ):
                        records.append({
                            "video_id": video_path.stem,
                            "video": video_path.name,
                            "timestamp_seconds": round(float(timestamp), 3),
                            "frame_path": str(frame_path.relative_to(ROOT)),
                        })

            report_progress(
                progress_callback,
                "extract",
                i + 1,
                len(sample_times),
                f"Extracting frames: {i + 1}/{len(sample_times)}",
            )

        if not records:
            raise ValueError("No usable frames could be extracted.")

        manifest = pd.DataFrame(records)
        manifest_path = output_dir.parent / "frames_manifest.csv"
        manifest.to_csv(manifest_path, index=False)

        return manifest, fps, duration, manifest_path
    finally:
        capture.release()


def _load_and_preprocess(record, preprocess):
    """Load one image and prepare its CLIP input tensor."""
    path = ROOT / record["frame_path"]
    try:
        with Image.open(path) as image:
            tensor = preprocess(image.convert("RGB"))
        return record, tensor, None
    except (OSError, ValueError) as exc:
        return record, None, str(exc)



def embed_frames(
    manifest,
    model,
    preprocess,
    device,
    batch_size=32,
    progress_callback=None,
    max_workers=4,
):
    """Preprocess images concurrently and encode them in GPU batches."""
    records = manifest.to_dict(orient="records")

    if not records:
        raise ValueError("The frame manifest is empty.")

    prepared = [None] * len(records)
    total_records = len(records)
    completed = 0

    # Workers only load and preprocess images.
    # Progress updates happen on the main thread.
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_load_and_preprocess, record, preprocess): i
            for i, record in enumerate(records)
        }

        for future in as_completed(futures):
            position = futures[future]
            prepared[position] = future.result()
            completed += 1

            report_progress(
                progress_callback,
                "preprocess",
                completed,
                total_records,
                f"Preparing images: {completed}/{total_records}",
            )

    valid_items = []

    for record, tensor, error in prepared:
        if tensor is None:
            print(f"[SKIP] {record['frame_path']}: {error}")
        else:
            valid_items.append((record, tensor))

    if not valid_items:
        raise ValueError("No readable frames were available for embedding.")

    embedding_batches = []
    metadata_records = []
    total_batches = (len(valid_items) + batch_size - 1) // batch_size

    for batch_number, start in enumerate(
        range(0, len(valid_items), batch_size),
        start=1,
    ):
        batch_items = valid_items[start:start + batch_size]
        batch = torch.stack(
            [tensor for _, tensor in batch_items]
        ).to(device)

        with torch.inference_mode():
            features = model.encode_image(batch)
            features = features / features.norm(
                dim=-1,
                keepdim=True,
            ).clamp_min(1e-12)

        vectors = features.float().cpu().numpy()

        for vector, (record, _) in zip(vectors, batch_items):
            if np.isfinite(vector).all():
                embedding_batches.append(vector)
                metadata_records.append(record)
            else:
                print(
                    f"[SKIP] Non-finite embedding: "
                    f"{record['frame_path']}"
                )

        report_progress(
            progress_callback,
            "embed",
            batch_number,
            total_batches,
            f"Analyzing frames: {batch_number}/{total_batches}",
        )

    if not embedding_batches:
        raise ValueError("No valid CLIP embeddings were generated.")

    embeddings = np.asarray(embedding_batches, dtype=np.float32)
    metadata = pd.DataFrame(metadata_records)

    if len(embeddings) != len(metadata):
        raise ValueError("Embedding count does not match metadata row count.")

    return embeddings, metadata



def build_faiss_index(embeddings, expected_dim=None):
    """Build an exact cosine-similarity index from normalized embeddings."""
    if embeddings.ndim != 2 or embeddings.shape[1] == 0:
        raise ValueError(
            f"Expected a non-empty 2-D embedding matrix, received {embeddings.shape}."
        )

    if expected_dim is not None and embeddings.shape[1] != expected_dim:
        raise ValueError(
            f"Expected embeddings with shape (N, {expected_dim}), "
            f"received {embeddings.shape}."
        )

    if not np.isfinite(embeddings).all():
        raise ValueError("Embeddings contain NaN or infinity.")

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
    embedding_dim=None,
    interval_seconds=1,
    batch_size=32,
    progress_callback=None,
):
    """Extract, embed, and index one uploaded video."""
    video_path = Path(video_path).resolve()

    if not video_path.is_file():
        raise FileNotFoundError(f"Uploaded video not found: {video_path}")

    if interval_seconds < 1:
        raise ValueError("Sampling interval must be at least 1 second.")

    # The saved filename ends with a unique content hash.
    upload_id = video_path.stem.rsplit("_", 1)[-1]
    processing_dir = ROOT / "data" / "processed" / upload_id
    frame_dir = processing_dir / "frames"
    artifact_dir = ROOT / "artifacts" / "uploaded" / upload_id

    frame_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    report_progress(progress_callback, "extract", 0, 1, "Extracting video frames")

    manifest, fps, duration, manifest_path = extract_frames(
        video_path,
        frame_dir,
        interval_seconds=interval_seconds,
        progress_callback=progress_callback,
    )

    embeddings, metadata = embed_frames(
        manifest,
        model,
        preprocess,
        device,
        batch_size=batch_size,
        progress_callback=progress_callback,
    )

    report_progress(progress_callback, "index", 0, 1, "Building visual index")
    index = build_faiss_index(embeddings, expected_dim=embedding_dim)

    embeddings_path = artifact_dir / "frame_embeddings.npy"
    metadata_path = artifact_dir / "frame_metadata.csv"
    index_path = artifact_dir / "frames.faiss"

    np.save(embeddings_path, embeddings)
    metadata.to_csv(metadata_path, index=False)
    faiss.write_index(index, str(index_path))

    report_progress(progress_callback, "done", 1, 1, "Processing complete")

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
