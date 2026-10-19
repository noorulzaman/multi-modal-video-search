
from pathlib import Path
import zipfile

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
ZIP_PATH = ROOT / "MSRVTT_Videos.zip"
CSV_PATH = ROOT / "evaluation" / "msrvtt_50_videos.csv"
OUTPUT_DIR = ROOT / "data" / "videos"

def main():
    if not ZIP_PATH.exists():
        raise FileNotFoundError(f"ZIP archive not found: {ZIP_PATH}")

    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Video inventory not found: {CSV_PATH}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    inventory = pd.read_csv(CSV_PATH)
    wanted = set(inventory["video"].dropna().astype(str))

    extracted = []
    missing = []

    with zipfile.ZipFile(ZIP_PATH, "r") as archive:
        members = [
            name for name in archive.namelist()
            if not name.endswith("/")
            and Path(name).name in wanted
        ]

        found_names = {Path(name).name for name in members}
        missing = sorted(wanted - found_names)

        for member in members:
            filename = Path(member).name
            destination = OUTPUT_DIR / filename

            # Extract only the expected basename; do not trust archive paths.
            with archive.open(member) as source, destination.open("wb") as target:
                while chunk := source.read(1024 * 1024):
                    target.write(chunk)

            extracted.append(filename)
            print(f"[OK] {filename}")

    print(f"\nRequested: {len(wanted)}")
    print(f"Extracted: {len(set(extracted))}")
    print(f"Missing: {len(missing)}")

    if missing:
        print("Missing filenames:")
        print("\n".join(missing))

if __name__ == "__main__":
    main()
