"""Measure search quality against a labelled query set.

Usage:
    python -m src.evaluate                      # current models
    python -m src.evaluate --process            # index missing videos first
    python -m src.evaluate --visual-model ViT-B-32 --pretrained laion2b_s34b_b79k --process

The query file (evaluation/queries.json) lists, per uploaded video, the
time ranges where each query's answer appears. A result counts as a hit
when its time range overlaps any labelled range (with a small tolerance).
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from src import config, models
from src.search import index_status, load_video_index, search_video
from src.upload_pipeline import index_transcript, process_uploaded_video

TOLERANCE_SECONDS = 1.0

CONFIGURATIONS = {
    "visual_plain": {"templates": ("{}",), "use_speech": False, "rerank": False},
    "visual_prompts": {"templates": config.PROMPT_TEMPLATES, "use_speech": False, "rerank": False},
    # None = the model's default prompt set (see config.prompt_templates_for).
    "fused": {"templates": None, "use_speech": True, "rerank": False},
    "fused_rerank": {"templates": None, "use_speech": True, "rerank": True},
}


def load_queries(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return data["videos"], data["queries"]


def is_hit(moment, label, tolerance=TOLERANCE_SECONDS):
    return any(
        moment["start_seconds"] <= end + tolerance
        and moment["end_seconds"] >= start - tolerance
        for start, end in label["ranges"]
    )


def first_hit_rank(moments, label):
    for rank, moment in enumerate(moments, start=1):
        if is_hit(moment, label):
            return rank
    return None


def summarize(rows):
    frame = pd.DataFrame(rows)
    # Misses arrive as None and become NaN in a numeric column.
    rank = frame["rank"].astype(float)
    frame["r@1"] = rank == 1
    frame["r@5"] = rank <= 5
    frame["mrr"] = (1.0 / rank).fillna(0.0)

    overall = frame.groupby("config")[["r@1", "r@5", "mrr"]].mean()
    by_type = frame.pivot_table(index="config", columns="type", values="r@5", aggfunc="mean")
    by_type.columns = [f"r@5 {column}" for column in by_type.columns]
    return overall.join(by_type).round(3), frame


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--queries", default=config.EVALUATION_DIR / "queries.json")
    parser.add_argument("--visual-model", default=config.VISUAL_MODEL)
    parser.add_argument("--pretrained", default=config.VISUAL_PRETRAINED)
    parser.add_argument("--configs", nargs="+", default=list(CONFIGURATIONS), choices=list(CONFIGURATIONS))
    parser.add_argument("--process", action="store_true", help="index videos that are not processed yet")
    parser.add_argument("--interval", type=int, default=2)
    parser.add_argument("--details", action="store_true", help="print every query's rank")
    args = parser.parse_args()

    videos, queries = load_queries(args.queries)
    encoder = models.load_visual_encoder(args.visual_model, args.pretrained)
    needs_speech = any(CONFIGURATIONS[name]["use_speech"] for name in args.configs)
    text_embedder = models.load_text_embedder() if needs_speech else None

    indexes = {}
    for upload_id, filename in videos.items():
        visual_ready, transcript_manifest = index_status(upload_id, args.visual_model, args.pretrained)
        video_path = config.UPLOAD_DIR / filename

        if not visual_ready:
            if not args.process:
                raise SystemExit(f"{filename} is not indexed for {args.visual_model}; rerun with --process.")
            print(f"Processing {filename} with {args.visual_model}...")
            process_uploaded_video(
                video_path,
                encoder,
                interval_seconds=args.interval,
                transcribe=False,
            )

        if needs_speech and transcript_manifest is None:
            if not args.process:
                raise SystemExit(f"{filename} has no speech index; rerun with --process.")
            print(f"Transcribing {filename}...")
            index_transcript(video_path, text_embedder)

        indexes[upload_id] = load_video_index(upload_id, args.visual_model, args.pretrained)

    reranker = None
    rows = []

    for name in args.configs:
        settings = CONFIGURATIONS[name]
        if settings["rerank"] and reranker is None:
            reranker = models.load_reranker()

        for label in queries:
            moments = search_video(
                indexes[label["video"]],
                encoder,
                [label["query"]],
                text_embedder=text_embedder,
                use_speech=settings["use_speech"],
                reranker=reranker if settings["rerank"] else None,
                top_k=5,
                templates=settings["templates"],
            )
            rows.append({
                "config": name,
                "type": label["type"],
                "query": label["query"],
                "rank": first_hit_rank(moments, label),
            })

    overall, frame = summarize(rows)
    print(f"\nModel: {args.visual_model} ({args.pretrained}), {len(queries)} queries\n")
    print(overall.to_string())

    if args.details:
        print()
        print(frame.pivot_table(index="query", columns="config", values="rank", aggfunc="first").to_string())


if __name__ == "__main__":
    main()
