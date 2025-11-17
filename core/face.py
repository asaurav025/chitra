from __future__ import annotations
from typing import List, Tuple
import numpy as np

# Optional dependency shim
try:
    import face_recognition
    HAS_FACE = True
except Exception:
    face_recognition = None
    HAS_FACE = False

def face_encodings(image_path: str) -> List[np.ndarray]:
    if not HAS_FACE:
        return []
    import numpy as np
    img = face_recognition.load_image_file(image_path)
    enc = face_recognition.face_encodings(img)
    return [np.array(e, dtype=np.float32) for e in enc]
