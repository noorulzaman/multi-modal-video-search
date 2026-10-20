"""Search one processed video by visual content and speech.

1. Visual: every query phrasing is embedded (with prompt ensembling for
   CLIP models) and compared with each keyframe; a segment scores as its
   best frame.
2. Speech: the query is compared with transcript windows; a segment takes
   the best score of windows that overlap it.
3. The two rankings are merged with reciprocal rank fusion, nearby
   duplicates are dropped, and a vision-language model can optionally
   re-check the top candidates.
"""

from dataclasses import dataclass
from functools import lru_cache

import faiss
import numpy as np
import pandas as pd

from src import config
from src.upload_pipeline import read_json


class IndexNotReady(Exception):
    """The video has not been processed with the current pipeline/model."""


@dataclass
class VideoIndex:
    frames: faiss.Index
    frame_metadata: pd.DataFrame
    segments: pd.DataFrame
    manifest: dict
    transcript: faiss.Index | None = None
    transcript_windows: pd.DataFrame | None = None


def index_status(upload_id, model_name=config.VISUAL_MODEL, pretrained=config.VISUAL_PRETRAINED):
    """Return (visual_ready, transcript_manifest_or_None)."""
    visual_manifest = read_json(
        config.visual_index_dir(upload_id, model_name, pretrained) / "manifest.json"
    )
    visual_ready = (
        visual_manifest is not None
        and visual_manifest.get("index_version") == config.INDEX_VERSION
    )
    transcript_manifest = read_json(
        config.transcript_index_dir(upload_id) / "manifest.json"
    )
    return visual_ready, transcript_manifest


def load_video_index(upload_id, model_name=config.VISUAL_MODEL, pretrained=config.VISUAL_PRETRAINED):
    visual_dir = config.visual_index_dir(upload_id, model_name, pretrained)
    transcript_dir = config.transcript_index_dir(upload_id)

    # Modification times are part of the cache key, so re-processing a
    # video invalidates the cached copy.
    stamps = tuple(
        path.stat().st_mtime if path.exists() else None
        for path in (visual_dir / "manifest.json", transcript_dir / "manifest.json")
    )
    return _load_video_index(str(visual_dir), str(transcript_dir), stamps)


@lru_cache(maxsize=8)
def _load_video_index(visual_dir, transcript_dir, _stamps):
    from pathlib import Path

    visual_dir = Path(visual_dir)
    transcript_dir = Path(transcript_dir)

    manifest = read_json(visual_dir / "manifest.json")
    if manifest is None or manifest.get("index_version") != config.INDEX_VERSION:
        raise IndexNotReady("Process this video before searching it.")

    frames = faiss.read_index(str(visual_dir / "frames.faiss"))
    frame_metadata = pd.read_csv(visual_dir / "frame_metadata.csv")
    segments = pd.read_csv(visual_dir / "segments.csv")

    if frames.ntotal != len(frame_metadata):
        raise IndexNotReady(
            "The visual index and frame metadata do not match. "
            "Please process the video again."
        )

    transcript = None
    transcript_windows = None
    if (transcript_dir / "transcript.faiss").is_file():
        transcript = faiss.read_index(str(transcript_dir / "transcript.faiss"))
        transcript_windows = pd.read_csv(transcript_dir / "transcript.csv")

    return VideoIndex(frames, frame_metadata, segments, manifest, transcript, transcript_windows)


def build_query_variants(query, corrected=None):
    """The original query plus its correction, without duplicates."""
    variants = []
    for text in (query, corrected):
        text = (text or "").strip()
        if text and text.lower() not in {v.lower() for v in variants}:
            variants.append(text)
    return variants


# --------------------------------------------------
# Scoring
# --------------------------------------------------

def visual_segment_scores(video_index, encoder, query_variants, templates=None):
    """
    Best frame per segment across all query variants.

    Returns a DataFrame indexed by segment_id with visual_score,
    frame_path, frame_timestamp, and matched_query.
    """
    if video_index.frames.d != encoder.dim:
        raise IndexNotReady(
            "This video was indexed with a different model. Please process it again."
        )

    k = min(config.VISUAL_CANDIDATE_FRAMES, video_index.frames.ntotal)
    rows = []

    for variant in query_variants:
        scores, ids = video_index.frames.search(encoder.encode_query(variant, templates), k)
        for score, row_id in zip(scores[0], ids[0]):
            if row_id < 0:
                continue
            frame = video_index.frame_metadata.iloc[int(row_id)]
            rows.append({
                "segment_id": int(frame["segment_id"]),
                "visual_score": float(score),
                "frame_path": frame["frame_path"],
                "frame_timestamp": float(frame["timestamp_seconds"]),
                "matched_query": variant,
            })

    if not rows:
        return pd.DataFrame(
            columns=["visual_score", "frame_path", "frame_timestamp", "matched_query"]
        )

    frame_scores = pd.DataFrame(rows).sort_values("visual_score", ascending=False)
    return frame_scores.drop_duplicates("segment_id").set_index("segment_id")


def transcript_segment_scores(video_index, text_embedder, query_variants):
    """
    Best overlapping transcript window per segment, above the score floor.

    Returns a DataFrame indexed by segment_id with transcript_score,
    transcript_text, and transcript_rank (rank of the window itself).
    """
    from src import models

    empty = pd.DataFrame(columns=["transcript_score", "transcript_text", "transcript_rank"])

    if video_index.transcript is None or text_embedder is None:
        return empty

    windows = video_index.transcript_windows
    k = min(config.TRANSCRIPT_CANDIDATES, video_index.transcript.ntotal)
    best = {}

    for vector in models.embed_texts(text_embedder, query_variants):
        scores, ids = video_index.transcript.search(vector[None, :], k)
        for score, window_id in zip(scores[0], ids[0]):
            if window_id >= 0 and score >= config.MIN_TRANSCRIPT_SCORE:
                best[int(window_id)] = max(float(score), best.get(int(window_id), -1.0))

    if not best:
        return empty

    ranked_windows = sorted(best.items(), key=lambda item: item[1], reverse=True)
    segments = video_index.segments
    rows = {}

    for rank, (window_id, score) in enumerate(ranked_windows, start=1):
        window = windows.iloc[window_id]
        overlapping = segments[
            (segments["start_seconds"] < window["end_seconds"])
            & (segments["end_seconds"] > window["start_seconds"])
        ]
        for segment_id in overlapping["segment_id"]:
            if int(segment_id) not in rows:
                rows[int(segment_id)] = {
                    "transcript_score": score,
                    "transcript_text": window["text"],
                    "transcript_rank": rank,
                }

    return pd.DataFrame.from_dict(rows, orient="index")


def reciprocal_rank_fusion(visual, transcript, k=config.RRF_K):
    """Merge per-segment rankings: score = sum of 1 / (k + rank)."""
    fused = {}

    for rank, segment_id in enumerate(visual.sort_values("visual_score", ascending=False).index, start=1):
        fused[segment_id] = fused.get(segment_id, 0.0) + 1.0 / (k + rank)

    for segment_id, rank in transcript["transcript_rank"].items():
        fused[segment_id] = fused.get(segment_id, 0.0) + 1.0 / (k + int(rank))

    return sorted(fused.items(), key=lambda item: item[1], reverse=True)


def drop_nearby_duplicates(moments, limit, min_gap=config.MIN_RESULT_GAP_SECONDS):
    """Keep the best moment in each neighbourhood so results are distinct."""
    kept = []
    for moment in moments:
        too_close = any(
            moment["start_seconds"] < other["end_seconds"] + min_gap
            and other["start_seconds"] < moment["end_seconds"] + min_gap
            for other in kept
        )
        if not too_close:
            kept.append(moment)
        if len(kept) >= limit:
            break
    return kept


def _first_frame_of(video_index, segment_id):
    frames = video_index.frame_metadata
    match = frames[frames["segment_id"] == segment_id]
    if match.empty:
        return None, None
    return match.iloc[0]["frame_path"], float(match.iloc[0]["timestamp_seconds"])


def search_video(
    video_index,
    encoder,
    query_variants,
    text_embedder=None,
    use_speech=True,
    reranker=None,
    top_k=config.TOP_K_RESULTS,
    templates=None,
):
    """
    Return up to top_k distinct moments, best first.

    templates defaults to the prompt set suited to the encoder's model.
    """
    if not query_variants:
        raise ValueError("Enter a description before searching.")

    visual = visual_segment_scores(video_index, encoder, query_variants, templates)
    transcript = (
        transcript_segment_scores(video_index, text_embedder, query_variants)
        if use_speech
        else transcript_segment_scores(video_index, None, query_variants)
    )

    segments = video_index.segments.set_index("segment_id")
    moments = []

    for segment_id, fused_score in reciprocal_rank_fusion(visual, transcript):
        segment = segments.loc[segment_id]
        moment = {
            "segment_id": int(segment_id),
            "start_seconds": float(segment["start_seconds"]),
            "end_seconds": float(segment["end_seconds"]),
            "fused_score": fused_score,
            "visual_score": None,
            "transcript_score": None,
            "transcript_text": None,
            "frame_path": None,
            "frame_timestamp": None,
            "matched_query": None,
            "rerank_score": None,
        }

        if segment_id in visual.index:
            row = visual.loc[segment_id]
            moment.update(
                visual_score=float(row["visual_score"]),
                frame_path=row["frame_path"],
                frame_timestamp=float(row["frame_timestamp"]),
                matched_query=row["matched_query"],
            )
        else:
            moment["frame_path"], moment["frame_timestamp"] = _first_frame_of(video_index, segment_id)

        if segment_id in transcript.index:
            row = transcript.loc[segment_id]
            moment.update(
                transcript_score=float(row["transcript_score"]),
                transcript_text=row["transcript_text"],
            )

        moments.append(moment)

    limit = config.RERANK_CANDIDATES if reranker is not None else top_k
    moments = drop_nearby_duplicates(moments, max(limit, top_k))

    if reranker is not None:
        query = query_variants[-1]
        for moment in moments:
            if moment["frame_path"] is not None:
                moment["rerank_score"] = reranker.score(
                    config.ROOT / moment["frame_path"],
                    query,
                    moment["transcript_text"],
                )
        # Stable sort keeps fusion order for ties and unscored moments.
        moments.sort(key=lambda m: m["rerank_score"] if m["rerank_score"] is not None else -1, reverse=True)

    return moments[:top_k]
