"""Model loaders shared by the app, the pipeline, and the evaluation script.

Loaders are cached per process, so Streamlit reruns reuse loaded models.
"""

import gc
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import open_clip
import torch

from src import config


def default_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def _normalize(features):
    return features / features.norm(dim=-1, keepdim=True).clamp_min(1e-12)


@dataclass
class VisualEncoder:
    """A CLIP-family model that embeds images and text into one space."""

    model: torch.nn.Module
    preprocess: object
    tokenizer: object
    device: str
    dim: int
    name: str
    pretrained: str

    def _autocast(self):
        # Half-precision weights need matching inputs; autocast handles the
        # casting per layer, whatever mix of dtypes the model ended up with.
        return torch.autocast("cuda", dtype=torch.float16, enabled=self.device == "cuda")

    def encode_images(self, batch):
        """Embed a preprocessed image batch. Returns normalized float32 rows."""
        with torch.inference_mode(), self._autocast():
            features = self.model.encode_image(batch.to(self.device))
        return _normalize(features.float()).cpu().numpy()

    def encode_texts(self, texts):
        """Embed raw strings. Returns normalized float32 rows."""
        tokens = self.tokenizer(list(texts)).to(self.device)
        with torch.inference_mode(), self._autocast():
            features = self.model.encode_text(tokens)
        return _normalize(features.float()).cpu().numpy()

    def encode_query(self, query, templates=None):
        """Average the embeddings of several phrasings of one query."""
        templates = templates or config.prompt_templates_for(self.name)
        texts = [template.format(query) for template in templates]
        mean = self.encode_texts(texts).mean(axis=0, keepdims=True)
        return (mean / np.linalg.norm(mean, axis=1, keepdims=True)).astype(np.float32)


@lru_cache(maxsize=2)
def load_visual_encoder(
    name=config.VISUAL_MODEL,
    pretrained=config.VISUAL_PRETRAINED,
    device=None,
):
    device = device or default_device()
    model, _, preprocess = open_clip.create_model_and_transforms(
        name,
        pretrained=pretrained,
        device=device,
        # Half precision halves GPU memory with no visible accuracy loss.
        precision="fp16" if device == "cuda" else "fp32",
    )
    model.eval()
    tokenizer = open_clip.get_tokenizer(name)

    # Measure the embedding size rather than relying on model attributes,
    # which differ between CLIP, SigLIP, and timm-based towers.
    with torch.inference_mode():
        dim = int(model.encode_text(tokenizer(["probe"]).to(device)).shape[-1])

    return VisualEncoder(model, preprocess, tokenizer, device, dim, name, pretrained)


@lru_cache(maxsize=1)
def load_text_embedder(
    name=config.TEXT_EMBED_MODEL,
    device=config.TEXT_EMBED_DEVICE,
):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(name, device=device)


def embed_texts(embedder, texts):
    """Embed transcript text or queries as normalized float32 rows."""
    vectors = embedder.encode(
        list(texts),
        batch_size=16,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return np.asarray(vectors, dtype=np.float32)


def load_asr(name=config.ASR_MODEL, device=None):
    """Load Whisper. Not cached: callers free it once transcription ends."""
    from transformers import pipeline

    device = device or default_device()
    return pipeline(
        "automatic-speech-recognition",
        model=name,
        dtype=torch.float16 if device == "cuda" else torch.float32,
        device=device,
    )


@lru_cache(maxsize=1)
def load_reranker(name=config.RERANK_MODEL, device=None):
    from src.reranker import VisionLanguageReranker

    return VisionLanguageReranker(name, device or default_device())


def unload_reranker():
    """Free the re-ranker's GPU memory, e.g. before running Whisper."""
    load_reranker.cache_clear()
    free_gpu_memory()


def free_gpu_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
