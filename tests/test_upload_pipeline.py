import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from src.upload_pipeline import build_faiss_index, embed_frames, extract_frames


class FakeClipModel:
    """Stand-in for CLIP that embeds an image as its mean RGB colour."""

    def encode_image(self, batch):
        return batch.mean(dim=(2, 3))


def fake_preprocess(image):
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1)


def normalized(rows):
    rows = np.asarray(rows, dtype=np.float32)
    return rows / np.linalg.norm(rows, axis=1, keepdims=True)


# --------------------------------------------------
# extract_frames
# --------------------------------------------------

def test_extract_frames_skips_black_frames(project_root, tiny_video):
    frame_dir = project_root / "frames"

    manifest, fps, duration, manifest_path = extract_frames(
        tiny_video, frame_dir, interval_seconds=1
    )

    assert fps == pytest.approx(10)
    assert duration == pytest.approx(5)
    # Second 0 is black and should be dropped.
    assert manifest["timestamp_seconds"].tolist() == [1.0, 2.0, 3.0, 4.0]
    assert manifest_path.is_file()

    for frame_path in manifest["frame_path"]:
        assert (project_root / frame_path).is_file()


def test_extract_frames_respects_interval(project_root, tiny_video):
    manifest, *_ = extract_frames(
        tiny_video, project_root / "frames", interval_seconds=2
    )

    assert manifest["timestamp_seconds"].tolist() == [2.0, 4.0]


def test_extract_frames_removes_stale_frames(project_root, tiny_video):
    frame_dir = project_root / "frames"
    frame_dir.mkdir()
    stale = frame_dir / "frame_9999.00s.jpg"
    stale.write_bytes(b"old")

    extract_frames(tiny_video, frame_dir, interval_seconds=1)

    assert not stale.exists()


def test_extract_frames_rejects_unreadable_video(project_root):
    bogus = project_root / "not_a_video.mp4"
    bogus.write_bytes(b"definitely not a video")

    with pytest.raises(ValueError):
        extract_frames(bogus, project_root / "frames")


# --------------------------------------------------
# embed_frames
# --------------------------------------------------

def test_embed_frames_normalizes_and_skips_unreadable(project_root):
    frame_dir = project_root / "frames"
    frame_dir.mkdir()

    records = []
    for i, colour in enumerate([(255, 0, 0), (0, 255, 0)]):
        path = frame_dir / f"frame_{i}.jpg"
        Image.new("RGB", (8, 8), colour).save(path)
        records.append({"frame_path": str(path.relative_to(project_root)),
                        "timestamp_seconds": float(i)})

    broken = frame_dir / "frame_broken.jpg"
    broken.write_bytes(b"not an image")
    records.append({"frame_path": str(broken.relative_to(project_root)),
                    "timestamp_seconds": 2.0})

    embeddings, metadata = embed_frames(
        pd.DataFrame(records),
        FakeClipModel(),
        fake_preprocess,
        device="cpu",
        batch_size=1,
    )

    assert embeddings.shape == (2, 3)
    assert embeddings.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(embeddings, axis=1), 1.0, atol=1e-5)
    assert metadata["timestamp_seconds"].tolist() == [0.0, 1.0]


def test_embed_frames_rejects_empty_manifest():
    with pytest.raises(ValueError):
        embed_frames(pd.DataFrame([]), FakeClipModel(), fake_preprocess, "cpu")


# --------------------------------------------------
# build_faiss_index
# --------------------------------------------------

@pytest.mark.parametrize("dim", [512, 768])
def test_build_faiss_index_accepts_any_model_dimension(dim):
    rng = np.random.default_rng(0)
    embeddings = normalized(rng.normal(size=(5, dim)))

    index = build_faiss_index(embeddings, expected_dim=dim)

    assert index.ntotal == 5
    assert index.d == dim
    _, ids = index.search(embeddings[2:3], 1)
    assert ids[0][0] == 2


def test_build_faiss_index_rejects_dimension_mismatch():
    embeddings = normalized(np.ones((3, 768)))

    with pytest.raises(ValueError, match="512"):
        build_faiss_index(embeddings, expected_dim=512)


def test_build_faiss_index_rejects_unnormalized():
    with pytest.raises(ValueError, match="normalized"):
        build_faiss_index(np.full((2, 4), 3.0, dtype=np.float32))


def test_build_faiss_index_rejects_non_finite():
    embeddings = normalized(np.ones((2, 4)))
    embeddings[0, 0] = np.nan

    with pytest.raises(ValueError, match="NaN"):
        build_faiss_index(embeddings)
