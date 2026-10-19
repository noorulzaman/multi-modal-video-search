
from pathlib import Path

import cv2
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
VIDEO_DIR = ROOT / "data" / "videos"
FRAME_DIR = ROOT / "data" / "frames"
VIDEO_MANIFEST = ROOT / "evaluation" / "video_manifest.csv"
OUTPUT_MANIFEST = ROOT / "evaluation" / "frames_manifest.csv"

# Baseline sampling rate: one frame per second.
SAMPLE_INTERVAL_SECONDS = 1.0

# Resize frames to reduce storage and preprocessing cost.
FRAME_WIDTH = 512
JPEG_QUALITY = 90


def extract_video_frames(video_path: Path, output_dir: Path) -> list[dict]:
    """Extract one frame per sampling interval from a video."""

    cap = cv2.VideoCapture(str(video_path))
    records = []

    if not cap.isOpened():
        print(f"[FAILED] Cannot open {video_path.name}")
        return records

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if not fps or fps <= 0 or frame_count <= 0:
        print(f"[FAILED] Invalid video metadata: {video_path.name}")
        cap.release()
        return records

    duration = frame_count / fps
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        timestamp = 0.0

        while timestamp < duration:
            cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
            success, frame = cap.read()

            if not success:
                print(
                    f"[WARNING] Could not read {timestamp:.2f}s "
                    f"from {video_path.name}"
                )
                timestamp += SAMPLE_INTERVAL_SECONDS
                continue

            height, width = frame.shape[:2]

            if width > FRAME_WIDTH:
                new_height = round(height * FRAME_WIDTH / width)
                frame = cv2.resize(
                    frame,
                    (FRAME_WIDTH, new_height),
                    interpolation=cv2.INTER_AREA,
                )

            frame_name = f"{video_path.stem}_{timestamp:08.2f}s.jpg"
            frame_path = output_dir / frame_name

            saved = cv2.imwrite(
                str(frame_path),
                frame,
                [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY],
            )

            if saved:
                records.append(
                    {
                        "video_id": video_path.stem,
                        "video": video_path.name,
                        "timestamp_seconds": round(timestamp, 3),
                        "frame": frame_name,
                        "frame_path": str(
                            frame_path.relative_to(ROOT)
                        ),
                    }
                )

            timestamp += SAMPLE_INTERVAL_SECONDS

    finally:
        cap.release()

    print(f"[OK] {video_path.name}: {len(records)} frames")
    return records


def main() -> None:
    if not VIDEO_MANIFEST.exists():
        raise FileNotFoundError(
            f"Missing video manifest: {VIDEO_MANIFEST}\n"
            "Run python src/prepare_dataset.py first."
        )

    FRAME_DIR.mkdir(parents=True, exist_ok=True)

    videos = pd.read_csv(VIDEO_MANIFEST)
    videos = videos[videos["readable"].astype(bool)]

    all_records = []
    failed_videos = 0

    print(f"Videos to process: {len(videos)}")
    print(f"Sampling interval: {SAMPLE_INTERVAL_SECONDS} second(s)")
    print("Extracting frames...\n")

    for _, row in videos.iterrows():
        video_path = ROOT / row["path"]
        output_dir = FRAME_DIR / row["video"].rsplit(".", 1)[0]

        if not video_path.exists():
            print(f"[FAILED] Missing video: {video_path}")
            failed_videos += 1
            continue

        records = extract_video_frames(video_path, output_dir)

        if not records:
            failed_videos += 1

        all_records.extend(records)

    manifest = pd.DataFrame(
        all_records,
        columns=[
            "video_id",
            "video",
            "timestamp_seconds",
            "frame",
            "frame_path",
        ],
    )
    manifest.to_csv(OUTPUT_MANIFEST, index=False)

    print("\n--- Frame extraction summary ---")
    print(f"Videos processed: {len(videos)}")
    print(f"Videos failed:    {failed_videos}")
    print(f"Frames extracted: {len(manifest)}")
    print(f"Manifest saved:   {OUTPUT_MANIFEST}")
    print(f"Frames directory: {FRAME_DIR}")


if __name__ == "__main__":
    main()
