"""Turn an uploaded video into searchable visual and speech indexes.

Visual: the video is split into shots (PySceneDetect), long shots are cut
into short segments, a few keyframes per segment are embedded with a
CLIP-family model, and the embeddings are stored in a FAISS index.

Speech: the audio track is transcribed with Whisper, grouped into short
time windows, embedded with a multilingual text model, and indexed.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re

import cv2
import faiss
import numpy as np
import pandas as pd
from PIL import Image

from src import config


def report_progress(callback, stage, current, total, message):
    """Report progress to the UI, if a callback was provided."""
    if callback is not None:
        callback(stage, current, total, message)


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


def read_json(path):
    path = Path(path)
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


# --------------------------------------------------
# Video probing and segmentation
# --------------------------------------------------

def probe_video(video_path):
    """Return (fps, frame_count, duration) or raise for unreadable videos."""
    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

        if fps <= 0 or frame_count <= 0:
            raise ValueError("The video has invalid frame-rate or duration metadata.")

        return fps, frame_count, frame_count / fps
    finally:
        capture.release()


def detect_scene_bounds(video_path, threshold=config.SCENE_DETECT_THRESHOLD):
    """Return (start, end) seconds for each shot, or [] if none were found."""
    from scenedetect import ContentDetector, SceneManager, open_video

    video = open_video(str(video_path))
    manager = SceneManager()
    manager.add_detector(ContentDetector(threshold=threshold))

    # Analysing ~12 frames per second finds the same cuts much faster
    # than every frame of 60 FPS footage.
    frame_skip = max(0, round(video.frame_rate / config.SCENE_DETECT_FPS) - 1)
    manager.detect_scenes(video, frame_skip=frame_skip)

    return [(start.seconds, end.seconds) for start, end in manager.get_scene_list()]


def build_segments(scene_bounds, duration, max_segment_seconds=config.MAX_SEGMENT_SECONDS):
    """
    Split shots into searchable segments no longer than max_segment_seconds.

    With no detected cuts the whole video is treated as one shot.
    """
    if not scene_bounds:
        scene_bounds = [(0.0, duration)]

    segments = []
    for scene_id, (start, end) in enumerate(scene_bounds):
        end = min(end, duration)
        length = end - start
        if length <= 0:
            continue

        pieces = max(1, math.ceil(length / max_segment_seconds))
        piece_length = length / pieces

        for piece in range(pieces):
            segments.append({
                "segment_id": len(segments),
                "scene_id": scene_id,
                "start_seconds": round(start + piece * piece_length, 3),
                "end_seconds": round(start + (piece + 1) * piece_length, 3),
            })

    return pd.DataFrame(
        segments,
        columns=["segment_id", "scene_id", "start_seconds", "end_seconds"],
    )


def keyframe_times(segments, interval_seconds):
    """Evenly spaced sample times inside each segment, at least one each."""
    times = []
    for row in segments.itertuples(index=False):
        length = row.end_seconds - row.start_seconds
        count = max(1, round(length / interval_seconds))
        step = length / count
        for k in range(count):
            times.append((row.start_seconds + (k + 0.5) * step, row.segment_id))
    return times


def is_dark_frame(frame):
    """True for black screens and fades, which carry no searchable content."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return gray.mean() < config.DARK_FRAME_MEAN and gray.std() < config.DARK_FRAME_STD


def _resize_for_storage(frame, max_width=config.FRAME_MAX_WIDTH):
    height, width = frame.shape[:2]
    if width <= max_width:
        return frame
    scale = max_width / width
    return cv2.resize(
        frame,
        (max_width, max(1, round(height * scale))),
        interpolation=cv2.INTER_AREA,
    )


# --------------------------------------------------
# Frame extraction and embedding
# --------------------------------------------------

def extract_frames(
    video_path,
    output_dir,
    segments,
    interval_seconds=2,
    progress_callback=None,
):
    """
    Walk the video forward and save the keyframes of each segment.

    Nearby targets are reached by grabbing frames in order; distant ones
    by seeking, which skips decoding the frames in between.
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Remove stale frame images from an earlier processing run.
    for old_frame in output_dir.glob("frame_*.jpg"):
        old_frame.unlink()

    fps, frame_count, _ = probe_video(video_path)

    # Map target frame numbers to (timestamp, segment); one frame may
    # serve two samples when segments are very short.
    targets = {}
    for timestamp, segment_id in keyframe_times(segments, interval_seconds):
        frame_number = min(int(timestamp * fps), frame_count - 1)
        targets.setdefault(frame_number, (timestamp, segment_id))

    pending = sorted(targets)
    total = len(pending)
    records = []

    # Seeking only pays off when it skips more than about two seconds.
    seek_gap = max(30, int(fps * 2))

    capture = cv2.VideoCapture(str(video_path))
    try:
        frame_number = 0
        next_target = 0

        while next_target < total:
            target = pending[next_target]

            if target - frame_number > seek_gap:
                capture.set(cv2.CAP_PROP_POS_FRAMES, target)
                frame_number = target

            if not capture.grab():
                break

            if frame_number == target:
                timestamp, segment_id = targets[frame_number]
                success, frame = capture.retrieve()

                if success and not is_dark_frame(frame):
                    frame_path = output_dir / f"frame_{timestamp:09.2f}s.jpg"
                    if cv2.imwrite(
                        str(frame_path),
                        _resize_for_storage(frame),
                        [cv2.IMWRITE_JPEG_QUALITY, 90],
                    ):
                        records.append({
                            "video_id": video_path.stem,
                            "segment_id": int(segment_id),
                            "timestamp_seconds": round(float(timestamp), 3),
                            "frame_path": str(frame_path.relative_to(config.ROOT)),
                        })

                next_target += 1
                report_progress(
                    progress_callback,
                    "extract",
                    next_target,
                    total,
                    f"Extracting frames: {next_target}/{total}",
                )

            frame_number += 1
    finally:
        capture.release()

    if not records:
        raise ValueError("No usable frames could be extracted.")

    return pd.DataFrame(records)


def _load_and_preprocess(record, preprocess):
    """Load one image and prepare its model input tensor."""
    path = config.ROOT / record["frame_path"]
    try:
        with Image.open(path) as image:
            tensor = preprocess(image.convert("RGB"))
        return record, tensor, None
    except (OSError, ValueError) as exc:
        return record, None, str(exc)


def embed_frames(
    manifest,
    encoder,
    batch_size=32,
    progress_callback=None,
    max_workers=4,
):
    """Preprocess images concurrently and encode them in GPU batches."""
    import torch

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
            executor.submit(_load_and_preprocess, record, encoder.preprocess): i
            for i, record in enumerate(records)
        }

        for future in as_completed(futures):
            prepared[futures[future]] = future.result()
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

    vectors_out = []
    metadata_records = []
    total_batches = (len(valid_items) + batch_size - 1) // batch_size

    for batch_number, start in enumerate(range(0, len(valid_items), batch_size), start=1):
        batch_items = valid_items[start:start + batch_size]
        vectors = encoder.encode_images(torch.stack([t for _, t in batch_items]))

        for vector, (record, _) in zip(vectors, batch_items):
            if np.isfinite(vector).all():
                vectors_out.append(vector)
                metadata_records.append(record)
            else:
                print(f"[SKIP] Non-finite embedding: {record['frame_path']}")

        report_progress(
            progress_callback,
            "embed",
            batch_number,
            total_batches,
            f"Analyzing frames: {batch_number}/{total_batches}",
        )

    if not vectors_out:
        raise ValueError("No valid embeddings were generated.")

    return np.asarray(vectors_out, dtype=np.float32), pd.DataFrame(metadata_records)


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


# --------------------------------------------------
# Speech transcription
# --------------------------------------------------

def load_audio(video_path, sample_rate=config.AUDIO_SAMPLE_RATE):
    """Decode the first audio track to mono float32, or None if there is none."""
    import av

    with av.open(str(video_path)) as container:
        stream = next((s for s in container.streams if s.type == "audio"), None)
        if stream is None:
            return None

        resampler = av.AudioResampler(format="flt", layout="mono", rate=sample_rate)
        pieces = []

        for frame in container.decode(stream):
            for resampled in resampler.resample(frame):
                pieces.append(resampled.to_ndarray().reshape(-1))

        for resampled in resampler.resample(None):
            pieces.append(resampled.to_ndarray().reshape(-1))

    if not pieces:
        return None

    return np.concatenate(pieces).astype(np.float32)


def collapse_repeats(text):
    """Collapse phrases Whisper repeated three or more times in a row."""
    return re.sub(
        r"\b((?:\S+\s+){0,5}?\S+?)(?:[\s,.!?]+\1\b){2,}",
        r"\1",
        text,
        flags=re.IGNORECASE,
    )


def is_silence_hallucination(start, end, text):
    """
    Whisper fills silent or music-only 30-second windows with stock
    phrases ("Thank you.", "I'm sorry."): a long span with very few words.
    """
    return end - start >= config.HALLUCINATION_MIN_SECONDS and len(text.split()) <= 4


def transcribe_audio(audio, asr, sample_rate=config.AUDIO_SAMPLE_RATE):
    """Run Whisper and return [{start, end, text}] chunks."""
    # Sequential long-form decoding with temperature fallback gives accurate
    # timestamps and avoids the repetition loops of chunked decoding.
    output = asr(
        {"raw": audio, "sampling_rate": sample_rate},
        return_timestamps=True,
        generate_kwargs={
            "task": "transcribe",
            "condition_on_prev_tokens": False,
            "compression_ratio_threshold": 1.35,
            "temperature": (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
            "logprob_threshold": -1.0,
            "no_speech_threshold": 0.6,
        },
    )

    duration = len(audio) / sample_rate
    chunks = []
    for chunk in output.get("chunks", []):
        start, end = chunk["timestamp"]
        text = collapse_repeats(chunk["text"].strip())
        if not text or start is None:
            continue

        end = duration if end is None else max(float(end), float(start))
        if is_silence_hallucination(start, end, text):
            continue

        chunks.append({
            "start_seconds": float(start),
            "end_seconds": float(end),
            "text": text,
        })
    return chunks


def build_transcript_windows(chunks, window_seconds=config.TRANSCRIPT_WINDOW_SECONDS):
    """Merge consecutive Whisper chunks into windows of about window_seconds."""
    windows = []
    current = []

    for chunk in chunks:
        if current and chunk["end_seconds"] - current[0]["start_seconds"] > window_seconds:
            windows.append(current)
            current = []
        current.append(chunk)

    if current:
        windows.append(current)

    return pd.DataFrame(
        [
            {
                "window_id": i,
                "start_seconds": round(window[0]["start_seconds"], 3),
                "end_seconds": round(window[-1]["end_seconds"], 3),
                "text": " ".join(chunk["text"] for chunk in window),
            }
            for i, window in enumerate(windows)
        ],
        columns=["window_id", "start_seconds", "end_seconds", "text"],
    )


def index_transcript(video_path, text_embedder, progress_callback=None, force=False):
    """
    Transcribe, window, embed, and index the speech in a video.

    The result does not depend on the visual model, so it is reused
    across visual re-processing unless force is set.
    """
    from src import models

    upload_id = config.upload_id_for(video_path)
    out_dir = config.transcript_index_dir(upload_id)
    manifest_path = out_dir / "manifest.json"

    existing = read_json(manifest_path)
    if (
        not force
        and existing is not None
        and existing.get("asr_model") == config.ASR_MODEL
        and existing.get("text_model") == config.TEXT_EMBED_MODEL
    ):
        return existing

    out_dir.mkdir(parents=True, exist_ok=True)
    report_progress(progress_callback, "transcribe", 0, 1, "Transcribing speech")

    audio = load_audio(video_path)
    windows = build_transcript_windows([])

    if audio is not None and len(audio) > 0:
        # Free the re-ranker first so Whisper fits on small GPUs.
        models.unload_reranker()
        asr = models.load_asr()
        try:
            chunks = transcribe_audio(audio, asr)
        finally:
            del asr
            models.free_gpu_memory()
        windows = build_transcript_windows(chunks)

    report_progress(progress_callback, "transcript_index", 0, 1, "Indexing speech")

    windows.to_csv(out_dir / "transcript.csv", index=False)
    index_path = out_dir / "transcript.faiss"

    if len(windows):
        embeddings = models.embed_texts(text_embedder, windows["text"].tolist())
        faiss.write_index(build_faiss_index(embeddings), str(index_path))
    elif index_path.exists():
        index_path.unlink()

    manifest = {
        "index_version": config.INDEX_VERSION,
        "asr_model": config.ASR_MODEL,
        "text_model": config.TEXT_EMBED_MODEL,
        "has_audio": audio is not None,
        "windows": len(windows),
        "created": datetime.now(timezone.utc).isoformat(),
    }
    write_json(manifest_path, manifest)
    return manifest


# --------------------------------------------------
# End-to-end processing
# --------------------------------------------------

def process_uploaded_video(
    video_path,
    encoder,
    interval_seconds=2,
    batch_size=32,
    transcribe=True,
    text_embedder=None,
    progress_callback=None,
):
    """Build the visual (and optionally speech) index for one uploaded video."""
    video_path = Path(video_path).resolve()

    if not video_path.is_file():
        raise FileNotFoundError(f"Uploaded video not found: {video_path}")

    if interval_seconds < 1:
        raise ValueError("Sampling interval must be at least 1 second.")

    upload_id = config.upload_id_for(video_path)
    slug = config.model_slug(encoder.name, encoder.pretrained)
    frame_dir = config.PROCESSED_DIR / upload_id / slug / "frames"
    visual_dir = config.visual_index_dir(upload_id, encoder.name, encoder.pretrained)
    visual_dir.mkdir(parents=True, exist_ok=True)

    fps, _, duration = probe_video(video_path)

    report_progress(progress_callback, "scenes", 0, 1, "Detecting scenes")
    segments = build_segments(detect_scene_bounds(video_path), duration)

    manifest = extract_frames(
        video_path,
        frame_dir,
        segments,
        interval_seconds=interval_seconds,
        progress_callback=progress_callback,
    )

    embeddings, metadata = embed_frames(
        manifest,
        encoder,
        batch_size=batch_size,
        progress_callback=progress_callback,
    )

    report_progress(progress_callback, "index", 0, 1, "Building visual index")
    index = build_faiss_index(embeddings, expected_dim=encoder.dim)

    np.save(visual_dir / "frame_embeddings.npy", embeddings)
    metadata.to_csv(visual_dir / "frame_metadata.csv", index=False)
    segments.to_csv(visual_dir / "segments.csv", index=False)
    faiss.write_index(index, str(visual_dir / "frames.faiss"))

    summary = {
        "index_version": config.INDEX_VERSION,
        "video_name": video_path.name,
        "visual_model": encoder.name,
        "visual_pretrained": encoder.pretrained,
        "embedding_dim": encoder.dim,
        "interval_seconds": interval_seconds,
        "duration_seconds": duration,
        "fps": fps,
        "scenes": int(segments["scene_id"].nunique()),
        "segments": len(segments),
        "frames_embedded": len(metadata),
        "created": datetime.now(timezone.utc).isoformat(),
    }
    write_json(visual_dir / "manifest.json", summary)

    if transcribe and text_embedder is not None:
        summary["transcript"] = index_transcript(
            video_path, text_embedder, progress_callback=progress_callback
        )

    report_progress(progress_callback, "done", 1, 1, "Processing complete")
    return summary
