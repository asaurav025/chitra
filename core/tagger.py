from __future__ import annotations
from typing import List, Tuple

from core.embedder import ClipEmbedder

# Very simple default labels; you can later externalize this.
DEFAULT_LABELS = [
    "portrait",
    "selfie",
    "group photo",
    "family",
    "friends",
    "landscape",
    "city",
    "night",
    "sunset",
    "food",
    "pets",
    "indoors",
    "outdoors",
    "travel",
    "wedding",
    "party",
    "sports",
]


def auto_tags(
    embedder: ClipEmbedder,
    image_path: str,
    k: int = 6,
) -> List[Tuple[str, float]]:
    """
    Use CLIP to auto-tag an image with simple labels.
    Returns list of (tag, score) sorted by score desc.
    """
    ranked = embedder.rank_labels(image_path, DEFAULT_LABELS, top_k=k)
    return ranked
