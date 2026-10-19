
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "evaluation" / "frames_manifest.csv"
OUTPUT = ROOT / "evaluation" / "frame_quality_report.csv"


def assess_frame(path: Path) -> dict:
    image = cv2.imread(str(path))

    if image is None:
        return {
            "readable": False,
            "mean_brightness": None,
            "std_brightness": None,
            "laplacian_variance": None,
            "is_black": False,
            "is_low_detail": False,
        }

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    mean_brightness = float(np.mean(gray))
    std_brightness = float(np.std(gray))
    laplacian_variance = float(cv2.Laplacian(
        gray, cv2.CV_64F
    ).var())

    return {
        "readable": True,
        "mean_brightness": round(mean_brightness, 2),
        "std_brightness": round(std_brightness, 2),
        "laplacian_variance": round(laplacian_variance, 2),
        "is_black": mean_brightness < 15 and std_brightness < 10,
        "is_low_detail": laplacian_variance < 5,
    }


def main():
    frames = pd.read_csv(MANIFEST)
    reports = []

    for _, row in frames.iterrows():
        path = ROOT / str(row["frame_path"])
        quality = assess_frame(path)

        reports.append({
            **row.to_dict(),
            **quality,
        })

    report = pd.DataFrame(reports)
    report.to_csv(OUTPUT, index=False)

    print("\n--- Frame quality summary ---")
    print(f"Total frames:       {len(report)}")
    print(f"Unreadable:         {(~report['readable']).sum()}")
    print(f"Nearly black:       {report['is_black'].sum()}")
    print(f"Low detail:         {report['is_low_detail'].sum()}")
    print(f"Report saved:       {OUTPUT}")

    print("\nFirst 20 suspected blank frames:")
    suspected = report[
        (~report["readable"])
        | report["is_black"]
        | report["is_low_detail"]
    ]
    print(
        suspected[
            [
                "video_id",
                "timestamp_seconds",
                "frame_path",
                "mean_brightness",
                "std_brightness",
                "laplacian_variance",
                "is_black",
                "is_low_detail",
            ]
        ].head(20).to_string(index=False)
    )


if __name__ == "__main__":
    main()
