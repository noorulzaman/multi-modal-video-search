
from pathlib import Path

import cv2
import pandas as pd


# Project paths
ROOT = Path(__file__).resolve().parent.parent
VIDEO_DIR = ROOT / "data" / "videos"
OUTPUT_DIR = ROOT / "evaluation"
OUTPUT_FILE = OUTPUT_DIR / "video_manifest.csv"

SUPPORTED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}


def inspect_video(video_path: Path) -> dict:
    """Read basic metadata from one video file."""

    cap = cv2.VideoCapture(str(video_path))

    try:
        if not cap.isOpened():
            return {
                "video": video_path.name,
                "path": str(video_path.relative_to(ROOT)),
                "fps": None,
                "frame_count": None,
                "duration_seconds": None,
                "width": None,
                "height": None,
                "readable": False,
                "error": "OpenCV could not open the video",
            }

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        duration = frame_count / fps if fps and fps > 0 else None

        return {
            "video": video_path.name,
            "path": str(video_path.relative_to(ROOT)),
            "fps": round(fps, 3) if fps and fps > 0 else None,
            "frame_count": frame_count,
            "duration_seconds": (
                round(duration, 3) if duration is not None else None
            ),
            "width": width,
            "height": height,
            "readable": frame_count > 0 and width > 0 and height > 0,
            "error": None,
        }

    finally:
        cap.release()


def main() -> None:
    """Scan extracted videos and save their metadata."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not VIDEO_DIR.exists():
        raise FileNotFoundError(
            f"Video directory does not exist: {VIDEO_DIR}\n"
            "Extract your selected MSR-VTT videos first."
        )

    video_paths = sorted(
        path
        for path in VIDEO_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    if not video_paths:
        raise FileNotFoundError(
            f"No supported video files found in: {VIDEO_DIR}"
        )

    print(f"Project directory: {ROOT}")
    print(f"Video directory:   {VIDEO_DIR}")
    print(f"Videos found:      {len(video_paths)}")
    print("\nInspecting videos...")

    records = []

    for index, video_path in enumerate(video_paths, start=1):
        metadata = inspect_video(video_path)
        records.append(metadata)

        status = "OK" if metadata["readable"] else "FAILED"
        duration = metadata["duration_seconds"]

        print(
            f"[{index}/{len(video_paths)}] {status} "
            f"{video_path.name} | "
            f"duration={duration}s | "
            f"resolution={metadata['width']}x{metadata['height']}"
        )

    manifest = pd.DataFrame(records)
    manifest.to_csv(OUTPUT_FILE, index=False)

    readable_count = int(manifest["readable"].sum())
    failed_count = len(manifest) - readable_count

    print("\n--- Dataset inspection summary ---")
    print(f"Total videos:    {len(manifest)}")
    print(f"Readable videos: {readable_count}")
    print(f"Failed videos:   {failed_count}")
    print(f"Manifest saved:  {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
