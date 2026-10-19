
import torch
import open_clip

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

model, _, _ = open_clip.create_model_and_transforms(
    "ViT-B-32",
    pretrained="laion2b_s34b_b79k",
    device=device,
)

tokenizer = open_clip.get_tokenizer("ViT-B-32")
texts = [
    "A person riding a bicycle",
    "A dog running in a park",
]

tokens = tokenizer(texts).to(device)

with torch.no_grad():
    features = model.encode_text(tokens)
    features = features / features.norm(dim=-1, keepdim=True)

print("Embedding shape:", tuple(features.shape))
print("Embedding device:", features.device)
print("Embedding dimensions:", features.shape[-1])
print("CLIP test successful!")
