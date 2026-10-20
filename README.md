# Multimodal Video Search

Upload a video and find moments in it by describing what you see ("a girl riding a bicycle through a park") or what is said ("asking a stranger what time it is"). The app returns the five best-matching moments as time ranges with thumbnails, and plays the video from any of them.

## How it works

**Processing a video**

1. **Scenes.** [PySceneDetect](https://www.scenedetect.com/) finds shot changes. Long shots are split so no searchable segment is longer than 10 seconds.
2. **Keyframes.** Each segment gets at least one keyframe, plus one every N seconds (you choose N) inside longer segments. Black screens and fades are skipped.
3. **Visual index.** Keyframes are embedded with [SigLIP 2](https://huggingface.co/timm/ViT-SO400M-16-SigLIP2-384) (`ViT-SO400M-16-SigLIP2-384`) through OpenCLIP and stored in a FAISS index.
4. **Speech index.** The audio is transcribed with [Whisper large-v3-turbo](https://huggingface.co/openai/whisper-large-v3-turbo), grouped into ~20-second passages, and embedded with the multilingual [BGE-M3](https://huggingface.co/BAAI/bge-m3) model, so English queries can match speech in other languages. Whisper's typical silence hallucinations ("Thank you.") and repetition loops are filtered out.

**Searching**

1. The query is searched as typed and, when spelling correction is on, as a spell-checked version too; whichever matches better counts.
2. Each segment gets a visual score (its best keyframe) and a speech score (the best overlapping transcript passage, if it clears a confidence floor).
3. The two rankings are merged with reciprocal rank fusion, and results within 4 seconds of a better one are dropped so the five results are distinct moments.
4. Optionally, **AI double-check** asks [Qwen3-VL-2B](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct) whether each of the top 10 candidates really matches (looking at the keyframe and any nearby speech) and re-orders them.

## Requirements

- Python 3.11
- An NVIDIA GPU with CUDA 13.0 drivers and at least 8 GB of memory is recommended. Everything also runs on CPU, much more slowly.
- **FFmpeg** (`ffprobe` on your `PATH`) to detect audio tracks. Audio decoding itself uses PyAV, which bundles its own FFmpeg.
- Optional: **Java 17+** for grammar correction with LanguageTool. Without it, only spelling is corrected.
- About **10 GB of disk** in the Hugging Face cache for the models, downloaded on first use:

  | Model | Used for | Download |
  |---|---|---|
  | SigLIP 2 SO400M | Visual search | ~4.5 GB |
  | Whisper large-v3-turbo | Transcription | ~1.6 GB |
  | BGE-M3 | Speech search | ~2.3 GB |
  | Qwen3-VL-2B-Instruct | AI double-check (optional) | ~4.3 GB |

GPU memory: SigLIP 2 uses about 2.3 GB, Qwen3-VL about 4.3 GB, and Whisper about 1.6 GB while transcribing (the double-check model is unloaded first). BGE-M3 runs on the CPU by default to leave room on 8 GB cards.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux
pip install -r requirements-dev.txt
```

`requirements.txt` pulls CUDA 13.0 builds of PyTorch from the PyTorch package index. On a CPU-only machine, change the `--extra-index-url` line at the top to `https://download.pytorch.org/whl/cpu` and remove the `+cu130` suffix from `torch` and `torchvision`.

## Running the app

Run from the repository root so that `src` is importable:

```bash
python -m streamlit run src/app.py
```

Uploads are limited to 200 MiB and to `.mp4`, `.mov`, `.avi`, `.mkv` and `.webm` files. The sidebar has toggles for speech search, spelling correction and the AI double-check.

## Configuration

Models and search settings live in [`src/config.py`](src/config.py). Models can also be swapped with environment variables, without code changes:

| Variable | Default |
|---|---|
| `MVS_VISUAL_MODEL` / `MVS_VISUAL_PRETRAINED` | `ViT-SO400M-16-SigLIP2-384` / `webli` |
| `MVS_ASR_MODEL` | `openai/whisper-large-v3-turbo` |
| `MVS_TEXT_EMBED_MODEL` / `MVS_TEXT_EMBED_DEVICE` | `BAAI/bge-m3` / `cpu` |
| `MVS_RERANK_MODEL` | `Qwen/Qwen3-VL-2B-Instruct` |

Any OpenCLIP model works for visual search (`python -c "import open_clip; print(open_clip.list_pretrained())"`). Each visual model keeps its own index, so switching models only requires processing the video again. The speech index is shared across visual models.

## Measuring search quality

[`evaluation/queries.json`](evaluation/queries.json) holds 37 labelled queries for the two sample videos: 28 visual and 9 about speech, each with the time ranges where the answer appears. [`src/evaluate.py`](src/evaluate.py) reports Recall@1, Recall@5 and MRR (mean reciprocal rank) per search configuration:

```bash
python -m src.evaluate --details
python -m src.evaluate --visual-model ViT-B-32 --pretrained laion2b_s34b_b79k --process
```

`--process` indexes any listed video that has not been processed with the chosen model. The query file names the uploaded videos by their saved filenames, so upload those videos through the app first, or point the file at your own videos and labels.

Results on the current set:

| Configuration | R@1 | R@5 | MRR |
|---|---|---|---|
| ViT-B-32, visual only (previous model) | 0.68 | 0.78 | 0.72 |
| SigLIP 2, visual only | 0.73 | 0.81 | 0.76 |
| **SigLIP 2 + speech (default)** | **0.76** | **0.95** | **0.83** |
| ViT-B-32 + speech | 0.76 | 0.95 | 0.83 |
| SigLIP 2 + speech + AI double-check | 0.84 | 0.92 | 0.86 |

Notes on these numbers:

- Adding speech is the largest gain: speech queries in the top 5 went from 3 of 9 to 8 of 9.
- The AI double-check puts the right moment first more often (R@1 0.76 → 0.84, mostly speech queries), but it rejected one correct visual match, dropping it out of the top 5. With both SigLIP 2 and Qwen3-VL loaded, peak GPU memory was 6.7 GB. It is off by default.
- Both visual models already place ~96% of visual queries in the top 5, so this small set cannot separate them once speech is fused in. A larger, harder query set is needed for that comparison.
- The speech confidence floor (`MIN_TRANSCRIPT_SCORE = 0.60`) and the choice to use prompt templates only for non-SigLIP models were tuned on this same set, so expect somewhat lower numbers on new videos.
- Sung lyrics are transcribed poorly by Whisper, so lyric queries rarely work.

## Tests

```bash
python -m pytest
```

The suite generates a tiny video and uses fake encoders, so it runs offline without downloading any model. To also load the real SigLIP 2 model:

```bash
RUN_CLIP_TESTS=1 python -m pytest
```

(In PowerShell: `$env:RUN_CLIP_TESTS = "1"; python -m pytest`.)

## Project layout

| Path | Purpose |
|---|---|
| `src/app.py` | Streamlit UI: upload, process, search, and play moments |
| `src/config.py` | Paths, model names, and search settings |
| `src/models.py` | Cached loaders for the visual, speech, text, and re-ranking models |
| `src/upload_pipeline.py` | Scene detection, keyframe extraction, embedding, transcription, indexing |
| `src/search.py` | Visual and speech scoring, rank fusion, de-duplication, re-ranking |
| `src/reranker.py` | Qwen3-VL yes/no match check |
| `src/query_corrector.py` | Spelling and grammar correction for queries |
| `src/evaluate.py` | Search-quality evaluation |
| `evaluation/queries.json` | Labelled evaluation queries |
| `tests/` | pytest suite |
| `data/uploads/` | Uploaded videos (git-ignored) |
| `data/processed/<video>/<model>/frames/` | Extracted keyframes (git-ignored) |
| `artifacts/uploaded/<video>/visual/<model>/` | Visual index, frame metadata, segments (git-ignored) |
| `artifacts/uploaded/<video>/transcript/` | Transcript passages and speech index (git-ignored) |

## Limitations

- Each search covers one video; there is no library-wide search yet.
- Processing runs inside the Streamlit request, so the page is busy until it finishes (about 30 seconds for a 2-minute video and 1–2 minutes for a 9-minute one on an RTX 4060 Laptop GPU, plus model loading on first use).
- Uploaded videos and generated files are never cleaned up automatically.
- Videos processed before this version must be processed again.
