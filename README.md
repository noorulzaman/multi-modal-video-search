# Multimodal Video Search

Upload a video, index its frames with [OpenCLIP](https://github.com/mlfoundations/open_clip), and find moments by describing them in plain English ("a person riding a bicycle"). The app jumps playback to the best-matching timestamp.

## How it works

1. **Upload.** The video is saved under `data/uploads/` with a content hash in its filename, so re-uploading the same file reuses earlier work.
2. **Process.** One frame is sampled every N seconds (you choose N), downscaled to 512 px wide, and embedded with CLIP ViT-B-32 (`laion2b_s34b_b79k`). The embeddings go into an exact cosine-similarity FAISS index under `artifacts/uploaded/<hash>/`.
3. **Search.** Your query is spell- and grammar-corrected, embedded with the CLIP text encoder, and compared against the 20 closest frames. A heuristic in `src/search.py` accepts the top frame only if it clears a similarity threshold and stands out from frames at least 5 seconds away.

## Requirements

- Python 3.11
- An NVIDIA GPU with CUDA 13.0 drivers is recommended. CPU works, but processing is slower.
- Optional: **Java 17+**, used by LanguageTool for grammar correction. Without it, only spelling is corrected.
- Optional: **ffprobe** (part of FFmpeg) on your `PATH`, used to detect audio tracks.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux
pip install -r requirements-dev.txt
```

`requirements.txt` pulls CUDA 13.0 builds of PyTorch from the PyTorch package index. On a CPU-only machine, change the `--extra-index-url` line at the top to `https://download.pytorch.org/whl/cpu` and remove the `+cu130` suffix from `torch` and `torchvision`.

The first run downloads the CLIP weights (about 600 MB) from Hugging Face. If Java is installed, LanguageTool also downloads about 200 MB the first time it corrects a query.

## Running the app

Run from the repository root so that `src` is importable:

```bash
python -m streamlit run src/app.py
```

Uploads are limited to 200 MiB and to `.mp4`, `.mov`, `.avi`, `.mkv` and `.webm` files.

## Tests

```bash
python -m pytest
```

The suite generates a tiny video and uses a fake CLIP model, so it runs offline in a few seconds. To also load the real CLIP model:

```bash
RUN_CLIP_TESTS=1 python -m pytest
```

(In PowerShell: `$env:RUN_CLIP_TESTS = "1"; python -m pytest`.)

## Project layout

| Path | Purpose |
|---|---|
| `src/app.py` | Streamlit UI: upload, process, and search |
| `src/upload_pipeline.py` | Frame extraction, CLIP image embedding, FAISS index building |
| `src/search.py` | Match-acceptance heuristic and its thresholds |
| `src/query_corrector.py` | Spelling and grammar correction for search queries |
| `tests/` | pytest suite |
| `data/uploads/` | Uploaded videos (git-ignored) |
| `data/processed/<hash>/` | Extracted frames per video (git-ignored) |
| `artifacts/uploaded/<hash>/` | Embeddings, frame metadata and FAISS index per video (git-ignored) |

## Limitations

- Search is visual only. Speech and on-screen text are not indexed yet.
- The match thresholds in `src/search.py` were tuned by hand for ViT-B-32 and are not calibrated probabilities.
- Only the single best moment is shown.
- Uploaded videos and generated files are never cleaned up automatically.
