
from pathlib import Path

import faiss
import open_clip
import pandas as pd
import streamlit as st
import torch
from PIL import Image


ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = ROOT / "artifacts"

INDEX_FILE = ARTIFACTS_DIR / "frames.faiss"
METADATA_FILE = ARTIFACTS_DIR / "frame_metadata.csv"

MODEL_NAME = "ViT-B-32"
PRETRAINED = "laion2b_s34b_b79k"


st.set_page_config(
    page_title="Multimodal Video Search",
    page_icon="🎬",
    layout="wide",
)

st.title("🎬 Multimodal Video Search")
st.write(
    "Search videos using natural language. "
    "Results are ranked by CLIP visual-text similarity."
)


@st.cache_resource
def load_resources():
    """Load the model and index once per Streamlit session cache."""

    if not INDEX_FILE.exists():
        raise FileNotFoundError(f"FAISS index not found: {INDEX_FILE}")

    if not METADATA_FILE.exists():
        raise FileNotFoundError(
            f"Frame metadata not found: {METADATA_FILE}"
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, _, preprocess = open_clip.create_model_and_transforms(
        MODEL_NAME,
        pretrained=PRETRAINED,
        device=device,
    )
    tokenizer = open_clip.get_tokenizer(MODEL_NAME)
    model.eval()

    index = faiss.read_index(str(INDEX_FILE))
    metadata = pd.read_csv(METADATA_FILE)

    if index.ntotal != len(metadata):
        raise ValueError(
            "FAISS index size does not match frame metadata."
        )

    return model, tokenizer, device, index, metadata


def encode_query(query, model, tokenizer, device):
    tokens = tokenizer([query]).to(device)

    with torch.inference_mode():
        features = model.encode_text(tokens)
        features = features / features.norm(
            dim=-1, keepdim=True
        ).clamp_min(1e-12)

    return features.cpu().numpy().astype("float32")


def search_videos(query, model, tokenizer, device, index, metadata, top_k):
    query_vector = encode_query(query, model, tokenizer, device)

    scores, indices = index.search(query_vector, index.ntotal)

    results = metadata.iloc[indices[0]].copy()
    results["similarity"] = scores[0]

    # Select each video's strongest matching frame.
    best_frames = (
        results.sort_values("similarity", ascending=False)
        .drop_duplicates(subset="video_id")
        .head(top_k)
        .copy()
    )

    return best_frames.reset_index(drop=True)


try:
    model, tokenizer, device, index, metadata = load_resources()
except Exception as exc:
    st.error(f"Could not load search resources: {exc}")
    st.stop()


st.caption(
    f"Model: {MODEL_NAME} · Device: {device} · "
    f"Indexed frames: {index.ntotal}"
)

with st.form("search_form"):
    query = st.text_input(
        "Describe the scene you want to find",
        placeholder="e.g. a person playing guitar",
    )
    top_k = st.slider("Number of video results", 1, 10, 5)
    submitted = st.form_submit_button("Search videos")

if submitted:
    if not query.strip():
        st.warning("Enter a description before searching.")
    else:
        with st.spinner("Searching video frames..."):
            results = search_videos(
                query.strip(),
                model,
                tokenizer,
                device,
                index,
                metadata,
                top_k,
            )

        st.subheader(f"Results for: {query.strip()}")

        if results.empty:
            st.info("No results were found.")
        else:
            for rank, (_, row) in enumerate(results.iterrows(), start=1):
                frame_path = ROOT / str(row["frame_path"])

                with st.container(border=True):
                    left, right = st.columns([1, 2])

                    with left:
                        st.markdown(f"### #{rank}")
                        if frame_path.exists():
                            st.image(
                                str(frame_path),
                                caption=f"Frame at {row['timestamp_seconds']:.1f}s",
                                use_container_width=True,
                            )
                        else:
                            st.warning("Preview frame not found.")

                    with right:
                        st.markdown(f"**Video:** `{row['video']}`")
                        st.metric(
                            "Cosine similarity",
                            f"{row['similarity']:.4f}",
                        )
                        st.write(
                            f"**Best timestamp:** "
                            f"{row['timestamp_seconds']:.1f} seconds"
                        )
                        st.caption(
                            "Similarity is a ranking score, not a probability."
                        )
