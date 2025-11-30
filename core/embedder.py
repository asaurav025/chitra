from __future__ import annotations
from pathlib import Path

import numpy as np
import torch
from transformers import CLIPModel, CLIPProcessor

from core.extractor import load_image

# Default labels for auto-tagging
DEFAULT_LABELS = [
    "person", "portrait", "landscape", "nature", "animal", "city", "indoor",
    "outdoor", "building", "food", "night", "day", "selfie", "group photo",
    "car", "sky", "mountains", "beach", "water", "architecture",
]


class ClipEmbedder:
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)

    # ------------------------------------------------------------
    # IMAGE → EMBEDDING
    # ------------------------------------------------------------
    def image_embedding(self, file_path: str) -> np.ndarray:
        img = load_image(Path(file_path))

        inputs = self.processor(images=img, return_tensors="pt").to(self.device)

        with torch.no_grad():
            feats = self.model.get_image_features(**inputs)

        v = feats[0].cpu().numpy().astype("float32")
        return v / (np.linalg.norm(v) + 1e-9)

    # ------------------------------------------------------------
    # TEXT → EMBEDDING
    # ------------------------------------------------------------
    def text_embedding(self, text: str) -> np.ndarray:
        inputs = self.processor(text=[text], return_tensors="pt").to(self.device)

        with torch.no_grad():
            feats = self.model.get_text_features(**inputs)

        v = feats[0].cpu().numpy().astype("float32")
        return v / (np.linalg.norm(v) + 1e-9)

    # ------------------------------------------------------------
    # LABEL RANKING (used by auto_tags)
    # ------------------------------------------------------------
    def rank_labels(self, image_vec: np.ndarray, top_k: int = 5):
        """
        Score DEFAULT_LABELS against an image embedding.
        Returns: list of (label, score)
        """
        labels = DEFAULT_LABELS

        inputs = self.processor(
            text=labels,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.device)

        with torch.no_grad():
            feats = self.model.get_text_features(**inputs)

        text_vecs = feats.cpu().numpy().astype("float32")
        text_vecs = text_vecs / (np.linalg.norm(text_vecs, axis=1, keepdims=True) + 1e-9)

        sims = text_vecs @ image_vec

        idx = sims.argsort()[::-1][:top_k]
        return [(labels[i], float(sims[i])) for i in idx]
