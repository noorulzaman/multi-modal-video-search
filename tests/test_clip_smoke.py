import os

import pytest

pytestmark = [
    pytest.mark.clip,
    pytest.mark.skipif(
        os.environ.get("RUN_CLIP_TESTS") != "1",
        reason="Set RUN_CLIP_TESTS=1 to load the real CLIP model.",
    ),
]


def test_clip_text_embeddings_are_normalized():
    import open_clip
    import torch

    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k", device="cpu"
    )
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    tokens = tokenizer(["A person riding a bicycle", "A dog running in a park"])

    with torch.inference_mode():
        features = model.encode_text(tokens)
        features = features / features.norm(dim=-1, keepdim=True)

    assert features.shape == (2, model.visual.output_dim)
    assert torch.allclose(features.norm(dim=-1), torch.ones(2), atol=1e-5)
