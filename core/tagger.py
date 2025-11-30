from __future__ import annotations
from typing import List, Tuple
from .embedder import ClipEmbedder

# A lightweight default label set; extend or override via CLI options in future
DEFAULT_LABELS = [
    "person", "selfie", "group photo", "baby", "family", "friends", "pet", "dog", "cat",
    "city", "building", "architecture", "monument", "temple", "mountain", "beach", "sea", "lake",
    "sunset", "sunrise", "forest", "garden", "road trip", "bike", "motorcycle", "car",
    "food", "dessert", "coffee", "party", "wedding", "festival", "Diwali", "Holi",
    "sports", "cricket", "football", "gym", "yoga", "concert", "stage", "museum",
]


def auto_tags(embedder: ClipEmbedder, image_path: str, k: int = 6) -> List[Tuple[str, float]]:
    """
    Generate top-K text labels for an image using CLIP.
    """
    # 1. Get image embedding
    embedding = embedder.image_embedding(image_path)

    # 2. Rank labels (internally uses DEFAULT_LABELS)
    ranked = embedder.rank_labels(embedding, top_k=k)

    # 3. Return top-k (label, score)
    return ranked
