import cv2
import numpy as np
import pytest
import torch

from src import config


@pytest.fixture
def project_root(tmp_path, monkeypatch):
    """Point every project path at a temporary directory."""
    monkeypatch.setattr(config, "ROOT", tmp_path)
    monkeypatch.setattr(config, "UPLOAD_DIR", tmp_path / "data" / "uploads")
    monkeypatch.setattr(config, "PROCESSED_DIR", tmp_path / "data" / "processed")
    monkeypatch.setattr(config, "ARTIFACT_DIR", tmp_path / "artifacts" / "uploaded")
    return tmp_path


@pytest.fixture
def tiny_video(tmp_path):
    """
    Write a 5-second, 10 FPS clip with no audio.

    The first second is black, so the extractor should skip it.
    Each later second has its own solid colour.
    """
    path = tmp_path / "clip_abc123.avi"
    fps = 10
    width, height = 64, 48
    colours = [(0, 0, 0), (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]

    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        fps,
        (width, height),
    )
    assert writer.isOpened(), "OpenCV could not create the test video."

    try:
        for colour in colours:
            frame = np.full((height, width, 3), colour, dtype=np.uint8)
            for _ in range(fps):
                writer.write(frame)
    finally:
        writer.release()

    return path


class FakeEncoder:
    """
    Stand-in for a CLIP-family encoder.

    Images embed as their mean RGB colour; text embeds by colour words,
    so "red" matches red frames.
    """

    name = "fake-model"
    pretrained = "test"
    dim = 3
    device = "cpu"

    COLOURS = {"red": [0, 0, 1], "green": [0, 1, 0], "blue": [1, 0, 0]}

    @staticmethod
    def preprocess(image):
        array = np.asarray(image, dtype=np.float32) / 255.0
        return torch.from_numpy(array).permute(2, 0, 1)

    def encode_images(self, batch):
        features = batch.mean(dim=(2, 3))
        return (features / features.norm(dim=-1, keepdim=True)).numpy()

    def encode_query(self, query, templates=("{}",)):
        vector = np.array([0.1, 0.1, 0.1], dtype=np.float32)
        for word, colour in self.COLOURS.items():
            if word in query.lower():
                vector = np.array(colour, dtype=np.float32)
        return (vector / np.linalg.norm(vector))[None, :]


class FakeTextEmbedder:
    """Embeds text by keyword so tests control which window matches."""

    KEYWORDS = ["money", "train", "music"]

    def encode(self, texts, **_):
        rows = []
        for text in texts:
            vector = np.array(
                [1.0 if word in text.lower() else 0.0 for word in self.KEYWORDS] + [0.01],
                dtype=np.float32,
            )
            rows.append(vector / np.linalg.norm(vector))
        return np.stack(rows)


@pytest.fixture
def fake_encoder():
    return FakeEncoder()


@pytest.fixture
def fake_text_embedder():
    return FakeTextEmbedder()
