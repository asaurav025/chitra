from __future__ import annotations
import os
import time
import hashlib
from pathlib import Path
from typing import Dict, Iterable
from datetime import datetime

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


def _normalize_exif_date(date_str: str) -> str:
    """
    Normalize EXIF date string to ISO format (YYYY-MM-DDTHH:MM:SS).
    Handles multiple EXIF date formats with fallbacks.
    
    Args:
        date_str: Date string in various formats (EXIF, ISO, etc.)
    
    Returns:
        Normalized ISO format string, or empty string if invalid
    """
    if not date_str or not date_str.strip():
        return ""
    
    date_str = date_str.strip()
    
    try:
        # Format 1: "YYYY:MM:DD HH:MM:SS" (standard EXIF)
        if ":" in date_str and " " in date_str:
            parts = date_str.split(" ", 1)
            if len(parts) == 2:
                date_part = parts[0].replace(":", "-", 2)  # "2025:12:07" -> "2025-12-07"
                time_part = parts[1]  # "10:30:45"
                normalized = f"{date_part}T{time_part}"
                # Validate
                datetime.fromisoformat(normalized)
                return normalized
        
        # Format 2: "YYYY-MM-DD HH:MM:SS" (already has dashes)
        if "-" in date_str and " " in date_str and "T" not in date_str:
            normalized = date_str.replace(" ", "T", 1)
            datetime.fromisoformat(normalized)
            return normalized
        
        # Format 3: Already ISO format "YYYY-MM-DDTHH:MM:SS"
        if "T" in date_str:
            datetime.fromisoformat(date_str)
            return date_str
        
        # Format 4: Date only "YYYY:MM:DD" or "YYYY-MM-DD"
        if " " not in date_str and "T" not in date_str:
            if ":" in date_str and len(date_str) >= 10:
                normalized = date_str.replace(":", "-", 2) + "T00:00:00"
            elif "-" in date_str and len(date_str) >= 10:
                normalized = date_str + "T00:00:00"
            else:
                return ""
            datetime.fromisoformat(normalized)
            return normalized
        
        # Try direct parsing as fallback
        # Handle various formats using datetime.strptime
        formats = [
            "%Y:%m:%d %H:%M:%S",  # EXIF standard
            "%Y-%m-%d %H:%M:%S",  # ISO-like with space
            "%Y/%m/%d %H:%M:%S",  # Alternative
            "%Y:%m:%d",           # Date only EXIF
            "%Y-%m-%d",           # Date only ISO
        ]
        
        for fmt in formats:
            try:
                dt = datetime.strptime(date_str, fmt)
                if " " in fmt or (":" in fmt and len(date_str) > 10):
                    # Has time component
                    return dt.isoformat()
                else:
                    # Date only, add midnight
                    return dt.isoformat() + "T00:00:00"
            except ValueError:
                continue
        
        # If all parsing fails, return empty
        return ""
        
    except (ValueError, AttributeError, IndexError, TypeError):
        # Invalid date format
        return ""


def get_exif(path: Path) -> Dict:
    """Extract basic EXIF including GPS & DateTime with multiple fallbacks."""
    # Try exifread first (works well for JPEG, PNG, etc.)
    try:
        with open(path, "rb") as f:
            tags = exifread.process_file(f, details=False)

        # Try multiple EXIF date fields in order of preference
        dt = str(
            tags.get("EXIF DateTimeOriginal")      # Best: When photo was actually taken
            or tags.get("EXIF DateTimeDigitized")  # Good: When photo was scanned/digitized
            or tags.get("EXIF DateTime")           # Fallback: EXIF modification time
            or tags.get("Image DateTime")         # Last resort: Image modification time
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

        # If we got a date from exifread, return it
        if dt:
            return {
                "exif_datetime": dt,
                "latitude": lat,
                "longitude": lon,
            }
    except Exception:
        pass  # Fall through to PIL method
    
    # Fallback to PIL/Pillow EXIF extraction (works better for HEIC files)
    # This is especially important for HEIC files where exifread may fail
    try:
        img = Image.open(str(path))
        
        # Try getexif() first (Pillow 8.0+)
        exif_data = None
        if hasattr(img, 'getexif'):
            exif_data = img.getexif()
        elif hasattr(img, '_getexif'):
            exif_data = img._getexif()
        
        dt = ""
        lat = lon = None
        
        if exif_data:
            # EXIF tag numbers for date fields
            # 306 = DateTime (Image DateTime)
            # 36867 = DateTimeOriginal (EXIF DateTimeOriginal)
            # 36868 = DateTimeDigitized (EXIF DateTimeDigitized)
            
            # Try date fields in order of preference
            dt = (
                exif_data.get(36867) or  # DateTimeOriginal
                exif_data.get(36868) or  # DateTimeDigitized
                exif_data.get(306) or    # DateTime
                ""
            )
            
            # Convert to string if it's not already
            if dt and not isinstance(dt, str):
                dt = str(dt)
            
            # Extract GPS coordinates if available
            # GPS data is in a separate IFD (tag 34853), but some versions expose it directly
            try:
                gps_ifd = None
                # Try to get GPS IFD (tag 34853)
                if hasattr(exif_data, 'get_ifd'):
                    try:
                        gps_ifd = exif_data.get_ifd(34853)  # GPS IFD tag
                    except Exception:
                        pass
                
                # GPS tags in GPS IFD: 1 = GPSLatitudeRef, 2 = GPSLatitude, 3 = GPSLongitudeRef, 4 = GPSLongitude
                # If GPS IFD not available, try direct access (some PIL versions)
                gps_data = gps_ifd if gps_ifd else exif_data
                
                if 2 in gps_data and 4 in gps_data:
                    # GPSLatitude and GPSLongitude are tuples of (degrees, minutes, seconds)
                    lat_tuple = gps_data.get(2)
                    lat_ref = gps_data.get(1)
                    lon_tuple = gps_data.get(4)
                    lon_ref = gps_data.get(3)
                    
                    if lat_tuple and lon_tuple:
                        # Convert tuple to decimal degrees
                        def tuple_to_degrees(tup, ref):
                            if not tup or len(tup) != 3:
                                return None
                            try:
                                # Handle both tuple and Rational types
                                def to_float(val):
                                    if hasattr(val, 'numerator') and hasattr(val, 'denominator'):
                                        return float(val.numerator) / float(val.denominator)
                                    return float(val)
                                
                                deg = to_float(tup[0])
                                min_val = to_float(tup[1])
                                sec = to_float(tup[2])
                                decimal = deg + (min_val / 60.0) + (sec / 3600.0)
                                if ref and ref in ('S', 'W'):
                                    decimal = -decimal
                                return decimal
                            except (ValueError, TypeError, IndexError):
                                return None
                        
                        lat = tuple_to_degrees(lat_tuple, lat_ref)
                        lon = tuple_to_degrees(lon_tuple, lon_ref)
            except Exception:
                pass  # GPS extraction failed, continue without it
        
        img.close()
        
        return {
            "exif_datetime": dt,
            "latitude": lat,
            "longitude": lon,
        }
        
    except Exception:
        pass  # Both methods failed
    
    # If all methods fail, return empty
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

    # Get EXIF date and normalize it to ISO format
    exif_dt_raw = exif.get("exif_datetime", "")
    normalized_exif_dt = _normalize_exif_date(exif_dt_raw) if exif_dt_raw else ""
    
    # Determine created_at: use normalized EXIF if available, otherwise file mtime
    if normalized_exif_dt:
        created_at = normalized_exif_dt
    else:
        created_at = time.strftime(
            "%Y-%m-%dT%H:%M:%S", time.localtime(st.st_mtime)
        )

    return {
        "file_path": str(path),
        "size": st.st_size,
        "created_at": created_at,
        "checksum": checksum,
        "phash": ph,
        "exif_datetime": normalized_exif_dt,  # Store normalized format
        "latitude": exif.get("latitude"),
        "longitude": exif.get("longitude"),
    }
