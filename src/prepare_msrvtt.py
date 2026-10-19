
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "evaluation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATASET_PATH = (
    "hf://datasets/friedrichor/MSR-VTT/msrvtt_train_9k.json"
)

print("Loading MSR-VTT annotations...")
df = pd.read_json(DATASET_PATH)

print("\nDataset shape:", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nFirst 5 rows:")
print(df.head().to_string(index=False))

# Confirm that the expected columns exist.
required = {"video_id", "video", "caption"}
missing = required - set(df.columns)

if missing:
    raise ValueError(f"Missing expected columns: {sorted(missing)}")

# Select up to 50 distinct videos, preserving all rows
# associated with those selected video IDs.
video_ids = (
    df["video_id"]
    .dropna()
    .drop_duplicates()
    .sort_values()
    .head(50)
)

subset = df[df["video_id"].isin(video_ids)].copy()

subset.to_csv(
    OUTPUT_DIR / "msrvtt_50_annotations.csv",
    index=False,
)

# A one-row-per-video inventory for later indexing.
inventory = (
    subset.drop_duplicates(subset=["video_id"])
    [["video_id", "video"]]
    .copy()
)

inventory.to_csv(
    OUTPUT_DIR / "msrvtt_50_videos.csv",
    index=False,
)

print("\nDistinct videos selected:", len(inventory))
print("Annotation rows preserved:", len(subset))
print("Saved:", OUTPUT_DIR / "msrvtt_50_annotations.csv")
print("Saved:", OUTPUT_DIR / "msrvtt_50_videos.csv")
