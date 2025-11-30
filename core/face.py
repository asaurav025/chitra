from __future__ import annotations
from pathlib import Path
from typing import List, Dict

import numpy as np

from core.extractor import load_image

try:
    from insightface.app import FaceAnalysis
    _HAS_INSIGHTFACE = True
except ImportError:
    _HAS_INSIGHTFACE = False

_app: FaceAnalysis | None = None


def _get_app() -> FaceAnalysis:
    """Singleton InsightFace app instance."""
    global _app
    if _app is not None:
        return _app

    if not _HAS_INSIGHTFACE:
        raise RuntimeError("insightface is not installed")

    # CPU-only (ctx_id=-1). If you want GPU later, you can switch to 0.
    app = FaceAnalysis(name="buffalo_l")
    app.prepare(ctx_id=-1)
    _app = app
    return _app


def face_encodings(path: str) -> List[Dict]:
    """
    Detect faces and return list of dicts:
        {
          "bbox": [x1, y1, x2, y2],
          "embedding": [...],
          "score": float,
          "landmarks": [[x, y], ...] or None
        }

    Uses RAW-safe loader so ARW/CR2/NEF also work.
    """
    if not _HAS_INSIGHTFACE:
        raise RuntimeError(
            "insightface not available. Install with: pip install insightface onnxruntime"
        )

    img_pil = load_image(Path(path))
    # PIL gives RGB, InsightFace expects BGR
    img = np.array(img_pil)[:, :, ::-1].copy()

    app = _get_app()
    faces = app.get(img)

    out: List[Dict] = []
    for f in faces:
        bbox = f.bbox.astype(int).tolist()
        emb = f.embedding.astype("float32").tolist()
        score = float(getattr(f, "det_score", 0.0))
        kps = getattr(f, "kps", None)
        landmarks = kps.tolist() if kps is not None else None

        out.append(
            {
                "bbox": bbox,
                "embedding": emb,
                "score": score,
                "landmarks": landmarks,
            }
        )

    return out
