
import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path

import cv2
import streamlit as st

from src import config, models
from src.query_corrector import correct_query
from src.search import (
    IndexNotReady,
    build_query_variants,
    index_status,
    load_video_index,
    search_video,
)
from src.upload_pipeline import process_uploaded_video


# --------------------------------------------------
# Project configuration
# --------------------------------------------------

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
    "Upload a video, index what it shows and what is said in it, and "
    "search for moments using natural-language descriptions."
)

with st.sidebar:
    st.header("Search settings")

    use_speech = st.toggle(
        "Search speech",
        value=True,
        help="Also match the query against what is said in the video.",
    )
    use_correction = st.toggle(
        "Spelling correction",
        value=True,
        help=(
            "Search with both your query and a spell-checked version, "
            "keeping whichever matches better."
        ),
    )
    use_rerank = st.toggle(
        "AI double-check (slower)",
        value=False,
        help=(
            f"Re-rank the top candidates with {config.RERANK_MODEL}. "
            "The first use downloads about 4 GB."
        ),
    )

    st.divider()
    st.caption(
        f"**Visual model:** {config.VISUAL_MODEL} ({config.VISUAL_PRETRAINED})  \n"
        f"**Speech model:** {config.ASR_MODEL}  \n"
        f"**Text model:** {config.TEXT_EMBED_MODEL}"
    )


# --------------------------------------------------
# Video upload helpers
# --------------------------------------------------

def save_uploaded_video(uploaded_file):
    """Validate and save an upload using a content-based filename."""

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

    config.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    saved_path = config.UPLOAD_DIR / f"{safe_stem}_{file_hash}.{extension}"

    if not saved_path.exists():
        saved_path.write_bytes(file_bytes)

    return saved_path


def get_saved_upload_path(uploaded_file):
    """
    Save an upload once per browser session.

    Streamlit reruns the script on every interaction, so without this
    each search would re-read and re-hash the whole video.
    """

    saved_uploads = st.session_state.setdefault("saved_uploads", {})
    saved_path = saved_uploads.get(uploaded_file.file_id)

    if saved_path is None or not saved_path.is_file():
        saved_path = save_uploaded_video(uploaded_file)
        saved_uploads[uploaded_file.file_id] = saved_path

    return saved_path


# Saved filenames include a content hash, so the path is a safe cache key.
@st.cache_data(show_spinner=False)
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


@st.cache_data(show_spinner=False)
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


def format_time(seconds):
    seconds = max(0, int(round(seconds)))
    return f"{seconds // 60}:{seconds % 60:02d}"


# --------------------------------------------------
# Progress reporting
# --------------------------------------------------

STAGE_RANGES = {
    "scenes": (0.00, 0.10),
    "extract": (0.10, 0.40),
    "preprocess": (0.40, 0.50),
    "embed": (0.50, 0.75),
    "index": (0.75, 0.78),
    "transcribe": (0.78, 0.95),
    "transcript_index": (0.95, 0.99),
    "done": (1.0, 1.0),
}

FRIENDLY_MESSAGES = {
    "scenes": "Detecting scene changes...",
    "extract": "Extracting keyframes...",
    "preprocess": "Preparing video frames...",
    "embed": "Analyzing frames with AI...",
    "index": "Building the visual index...",
    "transcribe": "Transcribing speech (this can take a minute)...",
    "transcript_index": "Indexing speech...",
    "done": "Complete",
}


def update_processing_progress(
    progress_bar,
    status,
    stage,
    current,
    total,
    message,
):
    """Update the progress bar from pipeline stage updates."""

    fraction = min(max(current / max(total, 1), 0.0), 1.0)
    start, end = STAGE_RANGES.get(stage, (0.0, 0.0))

    progress_bar.progress(
        min(start + (end - start) * fraction, 1.0),
        text="Preparing your video for search...",
    )
    status.caption(FRIENDLY_MESSAGES.get(stage, "Processing video..."))


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
audio_streams = None

if uploaded_file is None:
    st.info("Upload a video to get started.")

else:
    try:
        video_path = get_saved_upload_path(uploaded_file)
        video_info = inspect_video(video_path)

        st.success("Video uploaded successfully.")

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
                format_time(duration) if duration is not None else "Unavailable",
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

st.header("2. Prepare for Search")

upload_id = None
visual_ready = False

if video_path is not None and Path(video_path).is_file():

    upload_id = config.upload_id_for(video_path)
    visual_ready, transcript_manifest = index_status(upload_id)

    if visual_ready:
        visual_manifest = json.loads(
            (config.visual_index_dir(upload_id) / "manifest.json").read_text()
        )
        speech_note = (
            f", {transcript_manifest['windows']} speech passages"
            if transcript_manifest
            else ", speech not indexed"
        )
        st.caption(
            f"Already processed: {visual_manifest['scenes']} scenes, "
            f"{visual_manifest['segments']} segments, "
            f"{visual_manifest['frames_embedded']} keyframes{speech_note}."
        )

    interval_seconds = st.select_slider(
        "Keyframe spacing",
        options=[1, 2, 3, 5],
        value=2,
        format_func=lambda value: (
            f"Every {value} second"
            if value == 1
            else f"Every {value} seconds"
        ),
        help=(
            "Each detected scene gets at least one keyframe. Smaller "
            "spacing captures more detail inside long scenes but takes longer."
        ),
    )

    has_audio = audio_streams is None or bool(audio_streams)

    transcribe = st.checkbox(
        "Transcribe speech",
        value=has_audio,
        disabled=not has_audio,
        help=(
            f"Uses {config.ASR_MODEL}. The first run downloads about 1.6 GB."
        ),
    )

    if st.button(
        "Re-process Video" if visual_ready else "Process Video",
        type="primary",
        width="stretch",
    ):
        progress_bar = st.progress(0, text="Starting video processing...")
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
            with st.spinner("Loading AI models..."):
                encoder = models.load_visual_encoder()
                text_embedder = models.load_text_embedder() if transcribe else None

            summary = process_uploaded_video(
                video_path=video_path,
                encoder=encoder,
                interval_seconds=int(interval_seconds),
                transcribe=transcribe,
                text_embedder=text_embedder,
                progress_callback=progress_callback,
            )

            progress_bar.progress(1.0, text="Video processing complete.")
            status.empty()
            st.session_state.pop("search_results", None)
            visual_ready = True

            st.success(
                f"Your video is ready: {summary['scenes']} scenes and "
                f"{summary['frames_embedded']} keyframes indexed."
            )

        except Exception as exc:
            progress_bar.empty()
            status.empty()
            st.error(f"Video processing failed: {exc}")

else:
    st.info("Upload a valid video to continue.")


# --------------------------------------------------
# Section 3: Search and play matching moments
# --------------------------------------------------

st.divider()
st.header("3. Search Your Video")


def render_moment(moment, rank, selected):
    """One result row: thumbnail, time range, scores, speech, play button."""

    thumb_col, info_col = st.columns([1, 3])

    with thumb_col:
        if moment["frame_path"] is not None:
            st.image(str(config.ROOT / moment["frame_path"]), width="stretch")

    with info_col:
        marker = "▶ " if selected else ""
        st.markdown(
            f"**{marker}#{rank} · {format_time(moment['start_seconds'])} – "
            f"{format_time(moment['end_seconds'])}**"
        )

        signals = []
        if moment["visual_score"] is not None:
            signals.append(f"🖼 visual {moment['visual_score']:.3f}")
        if moment["transcript_score"] is not None:
            signals.append(f"🗣 speech {moment['transcript_score']:.2f}")
        if moment["rerank_score"] is not None:
            signals.append(f"✅ AI check {moment['rerank_score']:.0%}")
        st.caption(" · ".join(signals))

        if moment["rerank_score"] is not None and moment["rerank_score"] < 0.5:
            st.caption("The AI check thinks this is probably not a match.")

        if moment["transcript_text"]:
            st.markdown(f"> _{moment['transcript_text'][:300]}_")

        if st.button("Play this moment", key=f"play_{rank}"):
            st.session_state["selected_moment"] = rank - 1
            st.rerun()


if not visual_ready or upload_id is None:
    st.info(
        "Process this video first. After processing, "
        "you can search and play matching moments."
    )

else:
    with st.form("search_form"):
        query = st.text_input(
            "What do you want to find?",
            placeholder="e.g. a person walking through a room, or something someone says",
        )

        submitted = st.form_submit_button(
            "Search Video",
            type="primary",
            width="stretch",
        )

    if submitted:
        # Clear previous results so a failed search cannot show stale data.
        st.session_state.pop("search_results", None)

        if not query.strip():
            st.warning("Enter a description before searching.")

        else:
            try:
                with st.spinner("Searching your video..."):
                    corrected = correct_query(query) if use_correction else None
                    variants = build_query_variants(query, corrected)

                    encoder = models.load_visual_encoder()
                    text_embedder = models.load_text_embedder() if use_speech else None

                if use_rerank:
                    with st.spinner("Loading the AI double-check model..."):
                        reranker = models.load_reranker()
                else:
                    reranker = None

                with st.spinner("Ranking moments..."):
                    moments = search_video(
                        load_video_index(upload_id),
                        encoder,
                        variants,
                        text_embedder=text_embedder,
                        use_speech=use_speech,
                        reranker=reranker,
                    )

                st.session_state["search_results"] = {
                    "upload_id": upload_id,
                    "query": query.strip(),
                    "corrected": corrected,
                    "moments": moments,
                }
                st.session_state["selected_moment"] = 0

            except IndexNotReady as exc:
                st.warning(str(exc))

            except Exception as exc:
                st.error(f"Search failed: {exc}")

    saved = st.session_state.get("search_results")

    if saved and saved["upload_id"] == upload_id:
        st.subheader(f'Results for "{saved["query"]}"')

        corrected = saved["corrected"]
        if corrected and corrected.strip().lower() != saved["query"].lower():
            st.info(f"**Also searched for:** {corrected}")

        moments = saved["moments"]

        if not moments:
            st.warning(
                "No matching moments were found. Try describing what is "
                "visible or what is said in a different way."
            )

        else:
            selected = min(st.session_state.get("selected_moment", 0), len(moments) - 1)
            current = moments[selected]

            st.video(
                str(video_path),
                start_time=max(0, int(current["start_seconds"])),
            )

            st.caption(
                "Moments are ranked by how well they match; scores are "
                "relative signals, not probabilities."
            )

            for rank, moment in enumerate(moments, start=1):
                render_moment(moment, rank, selected == rank - 1)


# --------------------------------------------------
# Footer
# --------------------------------------------------

st.divider()
st.caption("Powered by OpenCLIP, Whisper, Sentence Transformers and Qwen3-VL")
