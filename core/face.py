from __future__ import annotations
from pathlib import Path
from typing import List, Dict

import numpy as np

from core.extractor import load_image

HAS_INSIGHTFACE = False
_FACE_APP = None


def _lazy_init_insightface():
    global HAS_INSIGHTFACE, _FACE_APP
    if _FACE_APP is not None:
        return _FACE_APP

    try:
        from insightface.app import FaceAnalysis
        import onnxruntime  # noqa: F401

        # Use CPU by default; you can tweak providers later.
        _FACE_APP = FaceAnalysis(name="buffalo_l")
        # ctx_id=-1 => CPU only
        _FACE_APP.prepare(ctx_id=-1, det_size=(640, 640))
        HAS_INSIGHTFACE = True
    except Exception as e:
        print(f"[yellow]InsightFace not available:[/yellow] {e}")
        HAS_INSIGHTFACE = False
        _FACE_APP = None

    return _FACE_APP


def face_encodings(file_path: str) -> List[Dict]:
    """
    Return a list of faces for the given image.

    Each item is:
      {
        "bbox": (x, y, w, h),
        "embedding": np.ndarray (float32),
        "score": float,
      }
    """
    app = _lazy_init_insightface()
    if not HAS_INSIGHTFACE or app is None:
        return []

    img = load_image(Path(file_path))
    # InsightFace expects numpy array in BGR or RGB; PIL -> numpy in RGB
    img_np = np.array(img)

    faces = app.get(img_np)
    results: List[Dict] = []

    for f in faces:
        # f.bbox: [x1, y1, x2, y2]
        x1, y1, x2, y2 = f.bbox.astype(int).tolist()
        w = x2 - x1
        h = y2 - y1

        emb = f.normed_embedding.astype("float32")

        results.append(
            {
                "bbox": (int(x1), int(y1), int(w), int(h)),
                "embedding": emb,
                "score": float(getattr(f, "det_score", 1.0)),
            }
        )

    return results
