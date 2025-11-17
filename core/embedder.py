from __future__ import annotations
from typing import Tuple, List
import numpy as np
import torch
import open_clip
from PIL import Image

_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class ClipEmbedder:
    def __init__(self, model_name: str = "ViT-B-32", pretrained: str = "openai"):
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained, device=_DEVICE)
        self.tokenizer = open_clip.get_tokenizer(model_name)

    @torch.no_grad()
    def image_embedding(self, image_path: str) -> np.ndarray:
        im = Image.open(image_path).convert("RGB")
        im = self.preprocess(im).unsqueeze(0).to(_DEVICE)
        feats = self.model.encode_image(im)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.squeeze(0).detach().cpu().numpy().astype(np.float32)

    @torch.no_grad()
    def text_embedding(self, text: str) -> np.ndarray:
        tokens = self.tokenizer([text]).to(_DEVICE)
        feats = self.model.encode_text(tokens)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.squeeze(0).detach().cpu().numpy().astype(np.float32)

    @torch.no_grad()
    def rank_labels(self, image_path: str, labels: List[str]) -> List[Tuple[str, float]]:
        im = Image.open(image_path).convert("RGB")
        im = self.preprocess(im).unsqueeze(0).to(_DEVICE)
        tok = self.tokenizer(labels).to(_DEVICE)
        with torch.no_grad():
            img = self.model.encode_image(im)
            txt = self.model.encode_text(tok)
            img = img / img.norm(dim=-1, keepdim=True)
            txt = txt / txt.norm(dim=-1, keepdim=True)
            sims = (img @ txt.T).squeeze(0).detach().cpu().numpy()
        order = np.argsort(-sims)
        return [(labels[i], float(sims[i])) for i in order]
