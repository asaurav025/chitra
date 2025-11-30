from __future__ import annotations
from typing import List, Dict, Any
import numpy as np
import insightface
from insightface.app import FaceAnalysis

# Cache model in memory
_engine = None


def _get_engine():
    """
    Initialize InsightFace model once.
    """
    global _engine
    if _engine is None:
        _engine = FaceAnalysis(
            name="buffalo_l",
            root="./.insightface",
            providers=["CPUExecutionProvider"]
        )
        _engine.prepare(ctx_id=0, det_size=(640, 640))
    return _engine


def face_encodings(image_path: str) -> List[Dict[str, Any]]:
    """
    Returns a list of detected faces with:
    - bbox
    - embedding
    - score
    - landmarks
    """
    engine = _get_engine()

    try:
        from core.raw_loader import load_image_any
        img = load_image_any(image_path)
        if img is None:
            return []
    except Exception:
        return []

    faces = engine.get(img)
    result = []

    for f in faces:
        result.append({
            "bbox": f.bbox.astype(float).tolist(),
            "embedding": f.embedding.astype(np.float32).tolist(),
            "score": float(f.det_score),
            "landmarks": f.kps.astype(float).tolist(),
        })

    return result
