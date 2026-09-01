from __future__ import annotations
from pathlib import Path

import numpy as np
import torch
from transformers import CLIPModel, CLIPProcessor

from core.extractor import load_image


class ClipEmbedder:
    """
    Wrapper around CLIP for image/text embeddings.
    Provides:
      - image_embedding(path)
      - text_embedding(text)
      - rank_labels(image_path, labels, top_k)
    """

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        # use_fast=False is deliberate and load-bearing. transformers is moving
        # `from_pretrained` to return the torchvision-backed *fast* image
        # processor by default; it still returns the slow one for this
        # checkpoint only because a slow config was saved alongside the
        # weights, and it warns that this will stop. Measured on the synthetic
        # reference in tests/test_embedder_stability.py, the fast processor
        # moves the output vector by 1.08e-03 — 10x that test's tolerance —
        # because it resizes and rescales differently.
        #
        # Every one of the ~1,800 vectors in `embeddings` was computed with the
        # slow path. Inheriting a default that flips would recompute new ones in
        # a subtly different space, with no error anywhere and search quietly
        # getting worse. Pinning costs nothing: the kwarg has been accepted
        # since at least 4.35.0.
        #
        # Moving to the fast processor is a re-embed decision, not a speed
        # tweak — it needs the whole corpus recomputed in the same pass.
        self.processor = CLIPProcessor.from_pretrained(model_name, use_fast=False)

    # ------------------------------------------------------------
    # IMAGE → EMBEDDING
    # ------------------------------------------------------------
    def image_embedding(self, file_path: str) -> np.ndarray:
        img = load_image(Path(file_path))

        inputs = self.processor(
            images=img,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            feats = self.model.get_image_features(**inputs)

        v = feats[0].cpu().numpy().astype("float32")
        return v / (np.linalg.norm(v) + 1e-9)

    # ------------------------------------------------------------
    # TEXT → EMBEDDING
    # ------------------------------------------------------------
    def text_embedding(self, text: str) -> np.ndarray:
        inputs = self.processor(
            text=[text],
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            feats = self.model.get_text_features(**inputs)

        v = feats[0].cpu().numpy().astype("float32")
        return v / (np.linalg.norm(v) + 1e-9)

    # ------------------------------------------------------------
    # LABEL RANKING (used by auto_tags)
    # ------------------------------------------------------------
    def rank_labels(
        self,
        image_path: str,
        labels: list[str],
        top_k: int = 6,
    ):
        """
        Score a list of text labels against an image.
        Returns: list of (label, score) pairs sorted by score desc.
        """
        if not labels:
            return []

        # 1) Image → embedding
        image_vec = self.image_embedding(image_path)  # already L2-normalized

        # 2) Embed all labels at once
        inputs = self.processor(
            text=labels,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.device)

        with torch.no_grad():
            feats = self.model.get_text_features(**inputs)

        text_vecs = feats.cpu().numpy().astype("float32")

        # Normalize text embeddings
        text_vecs = text_vecs / (np.linalg.norm(text_vecs, axis=1, keepdims=True) + 1e-9)

        # Cosine similarity (image_vec is already normalized)
        sims = text_vecs @ image_vec

        # Sort and pick top-k
        idx = sims.argsort()[::-1][:top_k]

        return [(labels[i], float(sims[i])) for i in idx]
