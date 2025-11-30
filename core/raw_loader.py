import rawpy
import imageio
import numpy as np
import cv2

def load_image_any(path: str):
    """
    Loads JPEG/PNG normally via cv2,
    loads RAW (ARW, CR2, NEF, etc.) via rawpy.
    Returns RGB uint8 image.
    """
    # Try OpenCV first
    img = load_image_any(path)
    if img is not None:
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # If OpenCV failed → try RAW loader
    try:
        with rawpy.imread(path) as raw:
            rgb = raw.postprocess()
            return rgb  # RGB uint8 array
    except Exception:
        return None
