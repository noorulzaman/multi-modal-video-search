import numpy as np
import pandas as pd
import pytest
from PIL import Image

from src import config
from src.upload_pipeline import (
    build_faiss_index,
    build_segments,
    build_transcript_windows,
    embed_frames,
    extract_frames,
    is_dark_frame,
    keyframe_times,
    load_audio,
    process_uploaded_video,
    read_json,
    transcribe_audio,
)


def normalized(rows):
    rows = np.asarray(rows, dtype=np.float32)
    return rows / np.linalg.norm(rows, axis=1, keepdims=True)


# --------------------------------------------------
# Segmentation
# --------------------------------------------------

def test_build_segments_splits_long_scenes():
    segments = build_segments([(0.0, 4.0), (4.0, 25.0)], duration=25.0, max_segment_seconds=10.0)

    assert segments["scene_id"].tolist() == [0, 1, 1, 1]
    assert segments["start_seconds"].tolist() == [0.0, 4.0, 11.0, 18.0]
    assert segments["end_seconds"].tolist() == [4.0, 11.0, 18.0, 25.0]
    assert segments["segment_id"].tolist() == [0, 1, 2, 3]


def test_build_segments_without_cuts_covers_whole_video():
    segments = build_segments([], duration=6.0)

    assert len(segments) == 1
    assert segments.iloc[0]["end_seconds"] == 6.0


def test_keyframe_times_gives_every_segment_a_frame():
    segments = build_segments([(0.0, 0.4), (0.4, 6.4)], duration=6.4)

    times = keyframe_times(segments, interval_seconds=2)

    assert [segment for _, segment in times] == [0, 1, 1, 1]
    assert times[0][0] == pytest.approx(0.2)
    assert [round(t, 2) for t, _ in times[1:]] == [1.4, 3.4, 5.4]


def test_is_dark_frame():
    assert is_dark_frame(np.zeros((8, 8, 3), dtype=np.uint8))
    assert not is_dark_frame(np.full((8, 8, 3), 200, dtype=np.uint8))


# --------------------------------------------------
# extract_frames
# --------------------------------------------------

def test_extract_frames_skips_black_frames(project_root, tiny_video):
    segments = build_segments([], duration=5.0)

    manifest = extract_frames(tiny_video, project_root / "frames", segments, interval_seconds=1)

    # The 0.5 s sample falls in the black first second and is dropped.
    assert manifest["timestamp_seconds"].tolist() == [1.5, 2.5, 3.5, 4.5]
    assert set(manifest["segment_id"]) == {0}
    for frame_path in manifest["frame_path"]:
        assert (project_root / frame_path).is_file()


def test_extract_frames_removes_stale_frames(project_root, tiny_video):
    frame_dir = project_root / "frames"
    frame_dir.mkdir()
    stale = frame_dir / "frame_9999.00s.jpg"
    stale.write_bytes(b"old")

    extract_frames(tiny_video, frame_dir, build_segments([], 5.0), interval_seconds=1)

    assert not stale.exists()


def test_extract_frames_rejects_unreadable_video(project_root):
    bogus = project_root / "not_a_video.mp4"
    bogus.write_bytes(b"definitely not a video")

    with pytest.raises(ValueError):
        extract_frames(bogus, project_root / "frames", build_segments([], 5.0))


# --------------------------------------------------
# embed_frames and build_faiss_index
# --------------------------------------------------

def test_embed_frames_normalizes_and_skips_unreadable(project_root, fake_encoder):
    frame_dir = project_root / "frames"
    frame_dir.mkdir()

    records = []
    for i, colour in enumerate([(255, 0, 0), (0, 255, 0)]):
        path = frame_dir / f"frame_{i}.jpg"
        Image.new("RGB", (8, 8), colour).save(path)
        records.append({"frame_path": str(path.relative_to(project_root)), "timestamp_seconds": float(i)})

    broken = frame_dir / "frame_broken.jpg"
    broken.write_bytes(b"not an image")
    records.append({"frame_path": str(broken.relative_to(project_root)), "timestamp_seconds": 2.0})

    embeddings, metadata = embed_frames(pd.DataFrame(records), fake_encoder, batch_size=1)

    assert embeddings.shape == (2, 3)
    np.testing.assert_allclose(np.linalg.norm(embeddings, axis=1), 1.0, atol=1e-5)
    assert metadata["timestamp_seconds"].tolist() == [0.0, 1.0]


def test_embed_frames_rejects_empty_manifest(fake_encoder):
    with pytest.raises(ValueError):
        embed_frames(pd.DataFrame([]), fake_encoder)


@pytest.mark.parametrize("dim", [512, 1152])
def test_build_faiss_index_accepts_any_model_dimension(dim):
    embeddings = normalized(np.random.default_rng(0).normal(size=(5, dim)))

    index = build_faiss_index(embeddings, expected_dim=dim)

    assert index.ntotal == 5
    _, ids = index.search(embeddings[2:3], 1)
    assert ids[0][0] == 2


def test_build_faiss_index_rejects_bad_input():
    with pytest.raises(ValueError, match="512"):
        build_faiss_index(normalized(np.ones((3, 768))), expected_dim=512)
    with pytest.raises(ValueError, match="normalized"):
        build_faiss_index(np.full((2, 4), 3.0, dtype=np.float32))

    with_nan = normalized(np.ones((2, 4)))
    with_nan[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        build_faiss_index(with_nan)


# --------------------------------------------------
# Speech
# --------------------------------------------------

def test_load_audio_returns_none_without_audio_track(tiny_video):
    assert load_audio(tiny_video) is None


def test_transcribe_audio_converts_whisper_chunks():
    def fake_asr(inputs, **kwargs):
        assert kwargs["return_timestamps"] is True
        return {"chunks": [
            {"timestamp": (0.0, 2.0), "text": " Hello there. "},
            {"timestamp": (2.0, None), "text": "Last words"},
            {"timestamp": (None, None), "text": "dropped"},
            {"timestamp": (3.0, 4.0), "text": "   "},
        ]}

    chunks = transcribe_audio(np.zeros(16_000 * 5, dtype=np.float32), fake_asr)

    assert chunks == [
        {"start_seconds": 0.0, "end_seconds": 2.0, "text": "Hello there."},
        {"start_seconds": 2.0, "end_seconds": 5.0, "text": "Last words"},
    ]


def test_build_transcript_windows_groups_by_duration():
    chunks = [
        {"start_seconds": float(s), "end_seconds": float(s + 4), "text": f"part {s}"}
        for s in range(0, 40, 5)
    ]

    windows = build_transcript_windows(chunks, window_seconds=12)

    assert windows["start_seconds"].tolist() == [0.0, 10.0, 20.0, 30.0]
    assert windows.iloc[0]["text"] == "part 0 part 5"


def test_build_transcript_windows_handles_no_speech():
    assert build_transcript_windows([]).empty


# --------------------------------------------------
# End to end
# --------------------------------------------------

def test_process_uploaded_video_writes_visual_index(project_root, tiny_video, fake_encoder, monkeypatch):
    monkeypatch.setattr(
        "src.upload_pipeline.detect_scene_bounds",
        lambda path: [(0.0, 2.0), (2.0, 5.0)],
    )
    video = config.UPLOAD_DIR / "clip_abc123.avi"
    video.parent.mkdir(parents=True)
    video.write_bytes(tiny_video.read_bytes())

    summary = process_uploaded_video(video, fake_encoder, interval_seconds=1, transcribe=False)

    visual_dir = config.visual_index_dir("abc123", fake_encoder.name, fake_encoder.pretrained)
    manifest = read_json(visual_dir / "manifest.json")

    assert summary["scenes"] == 2
    assert manifest["index_version"] == config.INDEX_VERSION
    assert manifest["embedding_dim"] == 3
    for name in ("frames.faiss", "frame_metadata.csv", "segments.csv", "frame_embeddings.npy"):
        assert (visual_dir / name).is_file()
