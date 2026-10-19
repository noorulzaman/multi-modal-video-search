
import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path

import cv2
import open_clip
import streamlit as st
import torch

from src.upload_pipeline import process_uploaded_video


# --------------------------------------------------
# Project configuration
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
UPLOAD_DIR = ROOT / "data" / "uploads"

MODEL_NAME = "ViT-B-32"
PRETRAINED = "laion2b_s34b_b79k"

MAX_UPLOAD_BYTES = 200 * 1024 * 1024
ALLOWED_EXTENSIONS = {"mp4", "mov", "avi", "mkv", "webm"}


# --------------------------------------------------
# Streamlit configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Multimodal Video Search",
    page_icon="🎬",
    layout="wide",
)

st.title("🎬 Multimodal Video Search")

st.write(
    "Upload a video to prepare it for visual search. "
    "The app extracts representative frames and creates a visual index."
)


# --------------------------------------------------
# Video upload helpers
# --------------------------------------------------

def save_uploaded_video(uploaded_file):
    """Validate and save an upload using a unique content-based filename."""

    original_name = Path(uploaded_file.name).name
    extension = Path(original_name).suffix.lower().lstrip(".")

    if extension not in ALLOWED_EXTENSIONS:
        raise ValueError(
            f"Unsupported extension: .{extension}. "
            f"Supported formats: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
        )

    file_bytes = uploaded_file.getvalue()

    if not file_bytes:
        raise ValueError("The uploaded file is empty.")

    if len(file_bytes) > MAX_UPLOAD_BYTES:
        raise ValueError("The file exceeds the 200 MiB upload limit.")

    file_hash = hashlib.sha256(file_bytes).hexdigest()[:16]

    original_stem = Path(original_name).stem
    safe_stem = re.sub(r"[^A-Za-z0-9_-]+", "_", original_stem)
    safe_stem = safe_stem.strip("_")[:60] or "uploaded_video"

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    saved_path = UPLOAD_DIR / f"{safe_stem}_{file_hash}.{extension}"

    if not saved_path.exists():
        saved_path.write_bytes(file_bytes)

    return saved_path


def inspect_video(video_path):
    """Inspect video metadata and verify that at least one frame decodes."""

    capture = cv2.VideoCapture(str(video_path))

    try:
        if not capture.isOpened():
            raise ValueError(
                "OpenCV could not open this video. "
                "The file may be corrupt or use an unsupported codec."
            )

        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

        duration = (
            frame_count / fps
            if fps > 0 and frame_count > 0
            else None
        )

        readable, _ = capture.read()

        if not readable:
            raise ValueError(
                "The video opened, but its first frame could not be decoded."
            )

        return {
            "width": width,
            "height": height,
            "fps": fps,
            "frame_count": frame_count,
            "duration": duration,
        }

    finally:
        capture.release()


def inspect_audio_stream(video_path):
    """Inspect audio tracks using ffprobe when available."""

    ffprobe = shutil.which("ffprobe")

    if ffprobe is None:
        return None

    command = [
        ffprobe,
        "-v",
        "error",
        "-show_entries",
        "stream=codec_type,codec_name,sample_rate,channels",
        "-of",
        "json",
        str(video_path),
    ]

    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=15,
            check=True,
        )

        information = json.loads(completed.stdout)

        return [
            stream
            for stream in information.get("streams", [])
            if stream.get("codec_type") == "audio"
        ]

    except (
        subprocess.SubprocessError,
        OSError,
        json.JSONDecodeError,
    ):
        return None


# --------------------------------------------------
# CLIP resources
# --------------------------------------------------

@st.cache_resource
def load_clip_resources():
    """Load CLIP once and reuse it across Streamlit reruns."""

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, _, image_preprocess = open_clip.create_model_and_transforms(
        MODEL_NAME,
        pretrained=PRETRAINED,
        device=device,
    )

    model.eval()

    return model, image_preprocess, device


# --------------------------------------------------
# Progress reporting
# --------------------------------------------------

def update_processing_progress(progress_bar, status, stage, current, total, message):
    """Update a single progress bar from pipeline stage updates."""

    fraction = min(max(current / max(total, 1), 0.0), 1.0)

    stage_ranges = {
        "extract": (0.0, 0.45),
        "preprocess": (0.45, 0.60),
        "embed": (0.60, 0.95),
        "index": (0.95, 0.99),
        "done": (1.0, 1.0),
    }

    start, end = stage_ranges.get(stage, (0.0, 0.0))
    progress = start + (end - start) * fraction

    progress_bar.progress(
        min(progress, 1.0),
        text="Preparing your video for visual search...",
    )

    # Keep the stage message short; do not show internal paths or filenames.
    friendly_messages = {
        "extract": "Extracting representative frames...",
        "preprocess": "Preparing video frames...",
        "embed": "Analyzing frames with AI...",
        "index": "Finishing up...",
        "done": "Complete",
    }

    status.caption(friendly_messages.get(stage, "Processing video..."))


# --------------------------------------------------
# Section 1: Upload and inspect
# --------------------------------------------------

st.header("1. Upload Your Video")

uploaded_file = st.file_uploader(
    "Choose a video file",
    type=sorted(ALLOWED_EXTENSIONS),
    help="Maximum file size: 200 MiB.",
    key="video_upload",
)

video_path = None
video_info = None

if uploaded_file is None:
    st.info("Upload a video to get started.")

else:
    try:
        video_path = save_uploaded_video(uploaded_file)
        video_info = inspect_video(video_path)

        st.success("Video uploaded successfully.")

        st.subheader("Video Preview")
        st.video(str(video_path))

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Resolution",
                f"{video_info['width']} × {video_info['height']}",
            )

        with col2:
            duration = video_info["duration"]

            st.metric(
                "Duration",
                f"{duration:.1f} seconds"
                if duration is not None
                else "Unavailable",
            )

        with col3:
            st.metric(
                "Frame Rate",
                f"{video_info['fps']:.2f} FPS"
                if video_info["fps"] > 0
                else "Unavailable",
            )

        audio_streams = inspect_audio_stream(video_path)

        if audio_streams is None:
            st.caption("Audio-track information is unavailable.")

        elif not audio_streams:
            st.caption("No audio track detected.")

        else:
            st.caption("Audio track detected.")

    except (ValueError, OSError) as exc:
        video_path = None
        video_info = None
        st.error(f"Could not process the uploaded video: {exc}")


st.divider()


# --------------------------------------------------
# Section 2: Process video
# --------------------------------------------------

st.header("2. Prepare for Visual Search")

if video_path is not None and Path(video_path).is_file():

    interval_seconds = st.select_slider(
        "Frame sampling interval",
        options=[1, 2, 3, 5, 10],
        value=2,
        format_func=lambda value: f"Every {value} second"
        if value == 1
        else f"Every {value} seconds",
        help=(
            "Smaller intervals capture more detail but take longer. "
            "Larger intervals are faster but may miss brief events."
        ),
    )

    if st.button("Process Video", type="primary", use_container_width=True):

        progress_bar = st.progress(
            0,
            text="Starting video processing...",
        )
        status = st.empty()

        def progress_callback(stage, current, total, message):
            update_processing_progress(
                progress_bar,
                status,
                stage,
                current,
                total,
                message,
            )

        try:
            with st.spinner("Loading AI model..."):
                model, image_preprocess, device = load_clip_resources()

            process_uploaded_video(
                video_path=video_path,
                model=model,
                preprocess=image_preprocess,
                device=device,
                interval_seconds=int(interval_seconds),
                batch_size=32,
                progress_callback=progress_callback,
            )

            progress_bar.progress(
                1.0,
                text="Video processing complete.",
            )
            status.empty()

            st.success("Your video is ready for visual search.")

        except Exception as exc:
            progress_bar.empty()
            status.empty()
            st.error(f"Video processing failed: {exc}")

else:
    st.info("Upload a valid video to continue.")


# --------------------------------------------------
# Footer
# --------------------------------------------------

st.divider()

st.caption("Powered by OpenCLIP")
