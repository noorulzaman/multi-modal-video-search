"""Project-wide settings: paths, model choices, and search tuning."""

import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
UPLOAD_DIR = ROOT / "data" / "uploads"
PROCESSED_DIR = ROOT / "data" / "processed"
ARTIFACT_DIR = ROOT / "artifacts" / "uploaded"
EVALUATION_DIR = ROOT / "evaluation"

# Bump when the on-disk index layout changes so old indexes are rebuilt.
INDEX_VERSION = 2

# --------------------------------------------------
# Models (each can be overridden with an environment variable)
# --------------------------------------------------

VISUAL_MODEL = os.environ.get("MVS_VISUAL_MODEL", "ViT-SO400M-16-SigLIP2-384")
VISUAL_PRETRAINED = os.environ.get("MVS_VISUAL_PRETRAINED", "webli")

# Multilingual, so English queries can match non-English speech.
TEXT_EMBED_MODEL = os.environ.get("MVS_TEXT_EMBED_MODEL", "BAAI/bge-m3")
# CPU keeps GPU memory free for the visual model and the re-ranker.
TEXT_EMBED_DEVICE = os.environ.get("MVS_TEXT_EMBED_DEVICE", "cpu")

ASR_MODEL = os.environ.get("MVS_ASR_MODEL", "openai/whisper-large-v3-turbo")
RERANK_MODEL = os.environ.get("MVS_RERANK_MODEL", "Qwen/Qwen3-VL-2B-Instruct")

# Averaging several phrasings gives a more robust CLIP text embedding.
# On the evaluation set this helped ViT-B-32 but slightly hurt SigLIP 2,
# which was trained on plain captions, so it is applied per model.
PROMPT_TEMPLATES = (
    "{}",
    "a photo of {}",
    "a video frame showing {}",
)


def prompt_templates_for(model_name):
    if "siglip" in model_name.lower():
        return ("{}",)
    return PROMPT_TEMPLATES

# --------------------------------------------------
# Indexing
# --------------------------------------------------

# PySceneDetect ContentDetector threshold (lower = more cuts).
SCENE_DETECT_THRESHOLD = 27.0
SCENE_DETECT_FPS = 12
# Long shots are split so each searchable segment stays short.
MAX_SEGMENT_SECONDS = 10.0
# Frames darker and flatter than this are fades or black screens.
DARK_FRAME_MEAN = 12.0
DARK_FRAME_STD = 6.0
FRAME_MAX_WIDTH = 512

# Consecutive Whisper chunks are merged into windows of about this length.
TRANSCRIPT_WINDOW_SECONDS = 20.0
AUDIO_SAMPLE_RATE = 16_000
# Whisper chunks this long with almost no words are silence hallucinations.
HALLUCINATION_MIN_SECONDS = 20.0

# --------------------------------------------------
# Search
# --------------------------------------------------

TOP_K_RESULTS = 5
# Visual frames retrieved before grouping into segments.
VISUAL_CANDIDATE_FRAMES = 200
TRANSCRIPT_CANDIDATES = 10
# Transcript windows scoring below this are treated as no speech match.
MIN_TRANSCRIPT_SCORE = 0.60
# Reciprocal rank fusion constant.
RRF_K = 60
# Results closer than this to a better result are dropped as duplicates.
MIN_RESULT_GAP_SECONDS = 4.0
RERANK_CANDIDATES = 10


def model_slug(model_name, pretrained):
    """Filesystem-safe folder name for one visual model."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", f"{model_name}__{pretrained}")


def upload_id_for(video_path):
    """Saved upload filenames end with a content hash, used as the ID."""
    return Path(video_path).stem.rsplit("_", 1)[-1]


def artifact_dir_for(upload_id):
    return ARTIFACT_DIR / upload_id


def visual_index_dir(upload_id, model_name=VISUAL_MODEL, pretrained=VISUAL_PRETRAINED):
    return artifact_dir_for(upload_id) / "visual" / model_slug(model_name, pretrained)


def transcript_index_dir(upload_id):
    return artifact_dir_for(upload_id) / "transcript"
