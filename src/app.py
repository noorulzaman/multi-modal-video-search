
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
    "Upload your own video to inspect it, extract representative frames, "
    "generate CLIP visual embeddings, and build a dedicated visual index. "
    "Question answering will be added in a later stage."
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
    """Load the local CLIP model once and reuse it across Streamlit reruns."""

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, _, image_preprocess = open_clip.create_model_and_transforms(
        MODEL_NAME,
        pretrained=PRETRAINED,
        device=device,
    )

    model.eval()

    return model, image_preprocess, device


# --------------------------------------------------
# Section 1: Upload and inspect
# --------------------------------------------------

st.header("1. Upload and Inspect Your Video")

uploaded_file = st.file_uploader(
    "Choose a video file",
    type=sorted(ALLOWED_EXTENSIONS),
    help="Maximum file size: 200 MiB.",
    key="video_upload",
)

video_path = None
video_info = None

if uploaded_file is None:
    st.info("Upload a video to begin.")

else:
    try:
        # Save the file before inspecting or previewing it.
        video_path = save_uploaded_video(uploaded_file)

        video_info = inspect_video(video_path)

        st.success("Video uploaded and validated successfully.")

        st.subheader("Video Preview")
        st.video(str(video_path))

        st.caption(f"Saved file: {video_path.name}")

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
                f"{duration:.2f} seconds"
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

        st.write(
            f"**Decoded frame count:** {video_info['frame_count']}"
        )

        audio_streams = inspect_audio_stream(video_path)

        if audio_streams is None:
            st.info(
                "Audio-track status could not be verified. "
                "Install FFmpeg to enable ffprobe audio inspection."
            )

        elif not audio_streams:
            st.warning("No audio stream was detected in this video.")

        else:
            st.success(
                f"Audio stream detected: {len(audio_streams)} track(s)."
            )

            for track_number, track in enumerate(
                audio_streams,
                start=1,
            ):
                st.caption(
                    f"Track {track_number}: "
                    f"{track.get('codec_name', 'unknown codec')}, "
                    f"{track.get('sample_rate', 'unknown')} Hz, "
                    f"{track.get('channels', 'unknown')} channel(s)"
                )

    except (ValueError, OSError) as exc:
        video_path = None
        video_info = None
        st.error(f"Could not process the uploaded video: {exc}")


st.divider()


# --------------------------------------------------
# Section 2: Extract frames and build a visual index
# --------------------------------------------------

st.header("2. Build Your Video's Visual Index")

st.write(
    "Sample frames from your uploaded video, encode them with CLIP, "
    "and save a dedicated FAISS index. These visual embeddings will "
    "support frame retrieval in a later question-answering stage."
)

if video_path is not None and Path(video_path).is_file():

    interval_seconds = st.select_slider(
        "Frame sampling interval (seconds)",
        options=[1, 2, 3, 5, 10],
        value=2,
        help=(
            "A 1-second interval captures more visual detail but takes "
            "longer. Larger intervals reduce processing time but can "
            "miss brief events."
        ),
    )

    if video_info and video_info["duration"] is not None:
        estimated_frames = max(
            1,
            int(video_info["duration"] / interval_seconds),
        )

        st.caption(
            f"Estimated sampling count: approximately "
            f"{estimated_frames:,} frames before black-frame filtering."
        )

    if st.button(
        "Extract Frames and Build Video Index",
        type="primary",
    ):
        try:
            with st.spinner(
                "Loading CLIP and processing your video. "
                "Long videos can take several minutes..."
            ):
                model, image_preprocess, device = load_clip_resources()

                processing_result = process_uploaded_video(
                    video_path=video_path,
                    model=model,
                    preprocess=image_preprocess,
                    device=device,
                    interval_seconds=int(interval_seconds),
                    batch_size=32,
                )

            st.success("Video processing completed successfully.")

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Frames Extracted",
                f"{processing_result['frames_extracted']:,}",
            )

            col2.metric(
                "Frames Embedded",
                f"{processing_result['frames_embedded']:,}",
            )

            col3.metric(
                "Embedding Dimensions",
                processing_result["embedding_shape"][1],
            )

            st.subheader("Processing Details")

            st.write(
                f"**Video:** {processing_result['video_name']}"
            )

            st.write(
                f"**Processing device:** {processing_result['device']}"
            )

            st.write(
                f"**Sampling interval:** {interval_seconds} second(s)"
            )

            if processing_result["duration_seconds"] is not None:
                st.write(
                    f"**Duration:** "
                    f"{processing_result['duration_seconds']:.2f} seconds"
                )

            st.write(
                "**Frame manifest:** "
                f"`{processing_result['manifest_path'].relative_to(ROOT)}`"
            )

            st.write(
                "**Embeddings:** "
                f"`{processing_result['embeddings_path'].relative_to(ROOT)}`"
            )

            st.write(
                "**Frame metadata:** "
                f"`{processing_result['metadata_path'].relative_to(ROOT)}`"
            )

            st.write(
                "**FAISS index:** "
                f"`{processing_result['index_path'].relative_to(ROOT)}`"
            )

            st.info(
                "The uploaded video's visual index is ready. "
                "Natural-language frame retrieval and local video "
                "question answering are not implemented yet."
            )

        except Exception as exc:
            st.error(f"Video processing failed: {exc}")

else:
    st.info(
        "Upload and validate a video in Section 1 to enable processing."
    )


# --------------------------------------------------
# Footer
# --------------------------------------------------

st.divider()

st.caption(
    f"Visual encoder: OpenCLIP {MODEL_NAME} | "
    f"Pretrained weights: {PRETRAINED}"
)
