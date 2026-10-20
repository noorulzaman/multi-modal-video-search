import cv2
import numpy as np
import pytest

from src import upload_pipeline


@pytest.fixture
def project_root(tmp_path, monkeypatch):
    """Point the pipeline's ROOT at a temporary directory."""
    monkeypatch.setattr(upload_pipeline, "ROOT", tmp_path)
    return tmp_path


@pytest.fixture
def tiny_video(tmp_path):
    """
    Write a 5-second, 10 FPS clip.

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
