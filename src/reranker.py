"""Second-pass check of search candidates with a vision-language model."""

import torch
from PIL import Image


class VisionLanguageReranker:
    """
    Ask a VLM whether a frame (plus nearby speech) matches the query and
    read the probability of "yes" from its next-token logits.
    """

    def __init__(self, model_name, device):
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.device = device
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_name,
            dtype=torch.bfloat16 if device == "cuda" else torch.float32,
            device_map=device,
        )
        self.model.eval()

        tokenizer = self.processor.tokenizer
        self.yes_ids = self._token_ids(tokenizer, ["Yes", "yes"])
        self.no_ids = self._token_ids(tokenizer, ["No", "no"])

    @staticmethod
    def _token_ids(tokenizer, words):
        ids = set()
        for word in words:
            encoded = tokenizer.encode(word, add_special_tokens=False)
            if encoded:
                ids.add(encoded[0])
        return sorted(ids)

    def build_prompt(self, query, transcript_text=None):
        if transcript_text:
            return (
                f'Speech heard around this moment: "{transcript_text[:400]}"\n'
                f'Does this video moment (the image and/or the speech) match '
                f'the description "{query}"? Answer only yes or no.'
            )
        return (
            f'Does this image match the description "{query}"? '
            f"Answer only yes or no."
        )

    def score(self, image_path, query, transcript_text=None):
        """Return P(yes) in [0, 1]."""
        with Image.open(image_path) as image:
            image = image.convert("RGB")

        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": self.build_prompt(query, transcript_text)},
            ],
        }]

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)

        with torch.inference_mode():
            logits = self.model(**inputs).logits[0, -1].float()

        yes = torch.logsumexp(logits[self.yes_ids], dim=0)
        no = torch.logsumexp(logits[self.no_ids], dim=0)
        return float(torch.sigmoid(yes - no))
