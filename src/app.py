
import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path

import cv2
import faiss
import open_clip
import pandas as pd
import streamlit as st
import torch

from src.upload_pipeline import process_uploaded_video


# --------------------------------------------------
# Project configuration
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = ROOT / "artifacts"
VIDEO_DIR = ROOT / "data" / "videos"
UPLOAD_DIR = ROOT / "data" / "uploads"

INDEX_FILE = ARTIFACTS_DIR / "frames.faiss"
METADATA_FILE = ARTIFACTS_DIR / "frame_metadata.csv"

MODEL_NAME = "ViT-B-32"
PRETRAINED = "laion2b_s34b_b79k"

MAX_UPLOAD_BYTES = 200 * 1024 * 1024
ALLOWED_EXTENSIONS = {"mp4", "mov", "avi", "mkv", "webm"}


# --------------------------------------------------
# Streamlit page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Multimodal Video Search",
    page_icon="🎬",
    layout="wide",
)

st.title("🎬 Multimodal Video Search")
st.write(
    "Search your indexed dataset or upload a video to extract frames, "
    "generate visual embeddings, and prepare it for AI-powered "
    "question answering."
)


# --------------------------------------------------
# Video upload helpers
# --------------------------------------------------

def save_uploaded_video(uploaded_file):
    """Validate and save an uploaded video without overwriting others."""

    original_name = Path(uploaded_file.name).name
    extension = Path(original_name).suffix.lower().lstrip(".")

    if extension not in ALLOWED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file extension: .{extension}. "
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

    # Do not rewrite an identical upload.
    if not saved_path.exists():
        saved_path.write_bytes(file_bytes)

    return saved_path


def inspect_video(video_path):
    """Read basic video properties and verify that a frame is decodable."""

    capture = cv2.VideoCapture(str(video_path))

    try:
        if not capture.isOpened():
            raise ValueError(
                "OpenCV could not open this video. "
                "It may be corrupt or use an unsupported codec."
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
                "The video opened, but OpenCV could not decode its first frame."
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
    """Return audio-track details if ffprobe is installed."""

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
        streams = information.get("streams", [])

        return [
            stream
            for stream in streams
            if stream.get("codec_type") == "audio"
        ]

    except (
        subprocess.SubprocessError,
        OSError,
        json.JSONDecodeError,
    ):
        return None


# --------------------------------------------------
# Section 1: Upload and inspect a video
# --------------------------------------------------

st.header("1. Upload and Inspect a Video")

st.write(
    "Upload a video to preview it and inspect its technical properties. "
    "You can then extract frames and build a separate visual index "
    "for that video."
)

uploaded_file = st.file_uploader(
    "Choose a video",
    type=sorted(ALLOWED_EXTENSIONS),
    help="Maximum upload size: 200 MiB.",
    key="video_upload",
)

# Always initialize the path. It remains None until saving succeeds.
video_path = None
video_info = None

if uploaded_file is not None:
    try:
        # Save first. This fixes the previous None-path problem.
        video_path = save_uploaded_video(uploaded_file)

        # Inspect the actual saved file.
        video_info = inspect_video(video_path)

        st.success("Video uploaded and validated successfully.")

        st.subheader("Video preview")
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
                "Frame rate",
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
                "Install FFmpeg/ffprobe to inspect audio streams."
            )

        elif not audio_streams:
            st.warning("No audio stream was detected in this video.")

        else:
            st.success(
                f"Audio stream detected: {len(audio_streams)} track(s)."
            )

            for track_number, track in enumerate(
                audio_streams, start=1
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
# Section 2: Load resources for dataset search
# --------------------------------------------------

st.header("2. Search the Existing Dataset")


@st.cache_resource
def load_resources():
    """Load the CLIP model and dataset FAISS index once."""

    if not INDEX_FILE.exists():
        raise FileNotFoundError(
            f"FAISS index not found: {INDEX_FILE}"
        )

    if not METADATA_FILE.exists():
        raise FileNotFoundError(
            f"Frame metadata not found: {METADATA_FILE}"
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, _, image_preprocess = open_clip.create_model_and_transforms(
        MODEL_NAME,
        pretrained=PRETRAINED,
        device=device,
    )

    tokenizer = open_clip.get_tokenizer(MODEL_NAME)
    model.eval()

    index = faiss.read_index(str(INDEX_FILE))
    metadata = pd.read_csv(METADATA_FILE)

    if index.ntotal != len(metadata):
        raise ValueError(
            "FAISS index size does not match frame metadata."
        )

    if index.d != 512:
        raise ValueError(
            f"Expected 512-dimensional CLIP embeddings; got {index.d}."
        )

    return model, tokenizer, image_preprocess, device, index, metadata


def encode_query(query, model, tokenizer, device):
    """Encode and normalize a text query with CLIP."""

    tokens = tokenizer([query]).to(device)

    with torch.inference_mode():
        features = model.encode_text(tokens)
        features = features / features.norm(
            dim=-1, keepdim=True
        ).clamp_min(1e-12)

    return features.float().cpu().numpy().astype("float32")


def search_videos(
    query,
    model,
    tokenizer,
    device,
    index,
    metadata,
    top_k,
):
    """Return the strongest matching frame for each dataset video."""

    query_vector = encode_query(
        query,
        model,
        tokenizer,
        device,
    )

    scores, indices = index.search(query_vector, index.ntotal)

    # FAISS may return -1 indices if fewer results are available.
    valid = indices[0] >= 0

    if not valid.any():
        return pd.DataFrame()

    results = metadata.iloc[indices[0][valid]].copy()
    results["similarity"] = scores[0][valid]

    best_frames = (
        results.sort_values("similarity", ascending=False)
        .drop_duplicates(subset="video_id")
        .head(top_k)
        .copy()
    )

    return best_frames.reset_index(drop=True)


# Load the model once for both dataset search and uploaded-video processing.
try:
    (
        model,
        tokenizer,
        image_preprocess,
        device,
        index,
        metadata,
    ) = load_resources()

except Exception as exc:
    st.error(f"Could not load search resources: {exc}")
    st.stop()


st.caption(
    f"Model: {MODEL_NAME} | Device: {device} | "
    f"Indexed dataset frames: {index.ntotal}"
)


# --------------------------------------------------
# Section 3: Process the uploaded video
# --------------------------------------------------

st.divider()
st.header("3. Process Uploaded Video")

if video_path is not None and Path(video_path).is_file():

    interval_seconds = st.select_slider(
        "Frame sampling interval",
        options=[1, 2, 3, 5, 10],
        value=1,
        help=(
            "One second samples more frames and takes longer. "
            "Larger intervals are faster but may miss brief events."
        ),
    )

    if st.button(
        "Extract Frames and Build Video Index",
        type="primary",
    ):
        try:
            with st.spinner(
                "Extracting frames, generating CLIP embeddings, "
                "and building the FAISS index..."
            ):
                processing_result = process_uploaded_video(
                    video_path=video_path,
                    model=model,
                    preprocess=image_preprocess,
                    device=device,
                    interval_seconds=int(interval_seconds),
                    batch_size=32,
                )

            st.success("Uploaded video processed successfully.")

            metric1, metric2, metric3 = st.columns(3)

            metric1.metric(
                "Frames extracted",
                processing_result["frames_extracted"],
            )

            metric2.metric(
                "Frames embedded",
                processing_result["frames_embedded"],
            )

            metric3.metric(
                "Embedding dimensions",
                processing_result["embedding_shape"][1],
            )

            st.write(
                "**Video:**",
                processing_result["video_name"],
            )

            st.write(
                "**Processing device:**",
                processing_result["device"],
            )

            if processing_result["duration_seconds"] is not None:
                st.write(
                    "**Duration:**",
                    f"{processing_result['duration_seconds']:.2f} seconds",
                )

            st.write(
                "**Frame manifest:**",
                str(
                    processing_result["manifest_path"].relative_to(ROOT)
                ),
            )

            st.write(
                "**Frame embeddings:**",
                str(
                    processing_result["embeddings_path"].relative_to(ROOT)
                ),
            )

            st.write(
                "**Frame metadata:**",
                str(
                    processing_result["metadata_path"].relative_to(ROOT)
                ),
            )

            st.write(
                "**FAISS index:**",
                str(
                    processing_result["index_path"].relative_to(ROOT)
                ),
            )

            st.info(
                "Processing is complete. The uploaded video's visual "
                "index is ready for the next stage: retrieving relevant "
                "frames to answer questions."
            )

        except Exception as exc:
            st.error(f"Video processing failed: {exc}")

else:
    st.info(
        "Upload and validate a video in Section 1 to enable processing."
    )


# --------------------------------------------------
# Section 4: Search the existing MSR-VTT dataset
# --------------------------------------------------

st.divider()
st.header("4. Search the Existing Dataset")

with st.form("search_form"):
    query = st.text_input(
        "Describe the scene you want to find",
        placeholder="e.g. a person playing guitar",
    )

    top_k = st.slider(
        "Number of video results",
        min_value=1,
        max_value=10,
        value=5,
    )

    submitted = st.form_submit_button("Search videos")


if submitted:
    if not query.strip():
        st.warning("Enter a description before searching.")

    else:
        with st.spinner("Searching video frames..."):
            results = search_videos(
                query.strip(),
                model,
                tokenizer,
                device,
                index,
                metadata,
                top_k,
            )

        st.subheader(f"Results for: {query.strip()}")

        if results.empty:
            st.info("No results were found.")

        else:
            for rank, (_, row) in enumerate(
                results.iterrows(),
                start=1,
            ):
                frame_path = ROOT / str(row["frame_path"])

                video_path_result = VIDEO_DIR / Path(
                    str(row["video"])
                ).name

                timestamp = max(
                    0.0,
                    float(row["timestamp_seconds"]),
                )

                with st.container(border=True):
                    left, right = st.columns([1, 2])

                    with left:
                        st.markdown(f"### Result #{rank}")

                        if frame_path.is_file():
                            st.image(
                                str(frame_path),
                                caption=(
                                    f"Matching frame: {timestamp:.1f}s"
                                ),
                                use_container_width=True,
                            )

                        else:
                            st.warning("Preview frame not found.")

                    with right:
                        st.markdown(
                            f"**Video:** `{row['video']}`"
                        )

                        st.metric(
                            "Cosine similarity",
                            f"{row['similarity']:.4f}",
                        )

                        st.write(
                            f"**Best timestamp:** {timestamp:.1f} seconds"
                        )

                        st.caption(
                            "Similarity is a ranking score, not a probability."
                        )

                    with st.expander("▶ Play matching video"):
                        if video_path_result.is_file():
                            st.video(
                                str(video_path_result),
                                start_time=int(timestamp),
                            )

                        else:
                            st.warning(
                                "Original dataset video not found."
                            )
