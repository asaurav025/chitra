from __future__ import annotations
import os
import time
import hashlib
from pathlib import Path
from typing import Dict, Iterable

import exifread
import imagehash
import rawpy
import numpy as np
from PIL import Image
# Register HEIC/HEIF support
try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except ImportError:
    pass  # pillow-heif not installed, HEIC support will be limited

# Supported non-RAW formats
IMG_EXTS = {
    ".jpg", ".jpeg", ".png", ".bmp",
    ".webp", ".tif", ".tiff", ".gif",
    ".heic", ".heif",  # Apple HEIC/HEIF format
}

# Supported RAW formats
RAW_EXTS = {
    ".arw", ".cr2", ".cr3", ".nef",
    ".rw2", ".orf", ".raf", ".dng", ".srw", ".pef",
}


# ----------------------------------------------------------------------
# IMAGE ITERATOR
# ----------------------------------------------------------------------
def iter_images(root: str) -> Iterable[Path]:
    """Yield all supported image files (JPEG/PNG/RAW) under root."""
    p = Path(root).expanduser()

    for dirpath, _, filenames in os.walk(p):
        for fn in filenames:
            ext = Path(fn).suffix.lower()
            if ext in IMG_EXTS or ext in RAW_EXTS:
                yield Path(dirpath) / fn


# ----------------------------------------------------------------------
# RAW + NORMAL IMAGE LOADER
# ----------------------------------------------------------------------
def _load_raw_as_rgb(path: Path) -> np.ndarray | None:
    """Decode RAW file into an RGB uint8 numpy array."""
    try:
        with rawpy.imread(str(path)) as raw:
            rgb = raw.postprocess(
                use_auto_wb=True,
                no_auto_bright=True,
                output_bps=8,
            )
        return rgb
    except Exception:
        return None


def load_image(path: Path) -> Image.Image:
    """
    Load image from JPG/PNG/TIFF or RAW (ARW/CR2/NEF...).
    Always returns a RGB PIL.Image.
    """
    ext = path.suffix.lower()
    path_str = str(path)

    # RAW path
    if ext in RAW_EXTS:
        rgb = _load_raw_as_rgb(path)
        if rgb is None:
            raise RuntimeError(f"Failed to decode RAW image {path_str}")
        return Image.fromarray(rgb).convert("RGB")

    # Normal path
    img = Image.open(path_str)
    return img.convert("RGB")


# ----------------------------------------------------------------------
# CHECKSUM
# ----------------------------------------------------------------------
def sha1sum(path: Path) -> str:
    """Compute SHA1 checksum for deduplication."""
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


# ----------------------------------------------------------------------
# EXIF
# ----------------------------------------------------------------------
def _convert_to_degrees(value):
    """Convert EXIF GPS coordinate to decimal degrees."""
    try:
        d = float(value.values[0].num) / float(value.values[0].den)
        m = float(value.values[1].num) / float(value.values[1].den)
        s = float(value.values[2].num) / float(value.values[2].den)
        return d + (m / 60.0) + (s / 3600.0)
    except Exception:
        return None


def get_exif(path: Path) -> Dict:
    """Extract basic EXIF including GPS & DateTime."""
    try:
        with open(path, "rb") as f:
            tags = exifread.process_file(f, details=False)

        dt = str(
            tags.get("EXIF DateTimeOriginal")
            or tags.get("Image DateTime")
            or ""
        )

        lat = lon = None
        if "GPS GPSLatitude" in tags and "GPS GPSLatitudeRef" in tags:
            lat = _convert_to_degrees(tags["GPS GPSLatitude"])
            if lat and tags["GPS GPSLatitudeRef"].values[0] == "S":
                lat = -lat

        if "GPS GPSLongitude" in tags and "GPS GPSLongitudeRef" in tags:
            lon = _convert_to_degrees(tags["GPS GPSLongitude"])
            if lon and tags["GPS GPSLongitudeRef"].values[0] == "W":
                lon = -lon

        return {
            "exif_datetime": dt,
            "latitude": lat,
            "longitude": lon,
        }

    except Exception:
        return {
            "exif_datetime": "",
            "latitude": None,
            "longitude": None,
        }


# ----------------------------------------------------------------------
# PHASH (uses RAW loader so RAW works)
# ----------------------------------------------------------------------
def compute_phash(path: Path) -> str:
    """Compute perceptual hash for any supported image (RAW or not)."""
    try:
        img = load_image(path)
        return str(imagehash.phash(img))
    except Exception:
        return ""


# ----------------------------------------------------------------------
# FINAL METADATA BUILDER
# ----------------------------------------------------------------------
def collect_metadata(path: Path) -> Dict:
    """Collect file metadata + EXIF + phash for DB insertion."""
    st = path.stat()
    checksum = sha1sum(path)
    ph = compute_phash(path)
    exif = get_exif(path)

    return {
        "file_path": str(path),
        "size": st.st_size,
        "created_at": time.strftime(
            "%Y-%m-%dT%H:%M:%S", time.localtime(st.st_mtime)
        ),
        "checksum": checksum,
        "phash": ph,
        **exif,
    }
