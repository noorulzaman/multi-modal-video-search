import faiss
import numpy as np
import pandas as pd
import pytest

from src.search import (
    IndexNotReady,
    VideoIndex,
    build_query_variants,
    drop_nearby_duplicates,
    reciprocal_rank_fusion,
    search_video,
)

# Frame colours as the fake encoder embeds them (BGR order → [b, g, r]).
RED = [0.0, 0.0, 1.0]
GREEN = [0.0, 1.0, 0.0]
BLUE = [1.0, 0.0, 0.0]


def flat_index(rows):
    rows = np.asarray(rows, dtype=np.float32)
    rows /= np.linalg.norm(rows, axis=1, keepdims=True)
    index = faiss.IndexFlatIP(rows.shape[1])
    index.add(rows)
    return index


def make_video_index(text_embedder=None, windows=None):
    """
    Four 10-second segments (red, green, blue, red-ish) with 10-second
    gaps, so the duplicate filter keeps them all.
    """
    segments = pd.DataFrame({
        "segment_id": [0, 1, 2, 3],
        "scene_id": [0, 1, 2, 3],
        "start_seconds": [0.0, 20.0, 40.0, 60.0],
        "end_seconds": [10.0, 30.0, 50.0, 70.0],
    })
    frames = pd.DataFrame({
        "segment_id": [0, 1, 2, 3],
        "timestamp_seconds": [5.0, 25.0, 45.0, 65.0],
        "frame_path": [f"frames/f{i}.jpg" for i in range(4)],
    })
    vectors = [RED, GREEN, BLUE, [0.0, 0.3, 1.0]]

    transcript = None
    if windows is not None:
        transcript = flat_index(text_embedder.encode(windows["text"].tolist()))

    return VideoIndex(
        frames=flat_index(vectors),
        frame_metadata=frames,
        segments=segments,
        manifest={},
        transcript=transcript,
        transcript_windows=windows,
    )


def test_build_query_variants_deduplicates():
    assert build_query_variants("a red car", "a red car") == ["a red car"]
    assert build_query_variants("a red cra", "a red car") == ["a red cra", "a red car"]
    assert build_query_variants("  ", None) == []


def test_reciprocal_rank_fusion_rewards_agreement():
    visual = pd.DataFrame({"visual_score": [0.9, 0.8, 0.7]}, index=[1, 2, 3])
    transcript = pd.DataFrame({"transcript_rank": [1]}, index=[2])

    fused = reciprocal_rank_fusion(visual, transcript, k=60)

    assert [segment for segment, _ in fused] == [2, 1, 3]


def test_drop_nearby_duplicates_keeps_best_per_neighbourhood():
    moments = [
        {"start_seconds": 10.0, "end_seconds": 12.0},
        {"start_seconds": 13.0, "end_seconds": 15.0},  # 1 s after the first
        {"start_seconds": 30.0, "end_seconds": 32.0},
    ]

    kept = drop_nearby_duplicates(moments, limit=5, min_gap=4.0)

    assert [m["start_seconds"] for m in kept] == [10.0, 30.0]


def test_visual_search_ranks_matching_colour_first(fake_encoder):
    moments = search_video(make_video_index(), fake_encoder, ["a red car"], top_k=2)

    assert [m["segment_id"] for m in moments] == [0, 3]
    assert moments[0]["visual_score"] == pytest.approx(1.0)
    assert moments[0]["transcript_score"] is None


def test_speech_match_can_outrank_visual(fake_encoder, fake_text_embedder):
    windows = pd.DataFrame({
        "window_id": [0, 1],
        "start_seconds": [22.0, 61.0],
        "end_seconds": [28.0, 68.0],
        "text": ["they talk about the money", "background music"],
    })
    video_index = make_video_index(fake_text_embedder, windows)

    moments = search_video(
        video_index,
        fake_encoder,
        ["money"],
        text_embedder=fake_text_embedder,
        top_k=4,
    )

    assert moments[0]["segment_id"] == 1
    assert moments[0]["transcript_text"] == "they talk about the money"

    visual_only = search_video(
        video_index, fake_encoder, ["money"], text_embedder=fake_text_embedder, use_speech=False, top_k=4
    )
    assert all(m["transcript_score"] is None for m in visual_only)


def test_reranker_reorders_candidates(fake_encoder):
    class PrefersBlue:
        def score(self, image_path, query, transcript_text=None):
            return 0.9 if str(image_path).endswith("f2.jpg") else 0.2

    moments = search_video(make_video_index(), fake_encoder, ["red"], reranker=PrefersBlue(), top_k=3)

    assert moments[0]["segment_id"] == 2
    assert moments[0]["rerank_score"] == pytest.approx(0.9)


def test_dimension_mismatch_is_reported(fake_encoder):
    video_index = make_video_index()
    video_index.frames = flat_index(np.ones((4, 5)))

    with pytest.raises(IndexNotReady):
        search_video(video_index, fake_encoder, ["red"])


def test_empty_query_is_rejected(fake_encoder):
    with pytest.raises(ValueError):
        search_video(make_video_index(), fake_encoder, [])
