"""ffmpeg/ffprobe CLI wrappers for video probing, poster extraction and transcoding.

Shells out to the system `ffmpeg`/`ffprobe` binaries (install: `apt-get install ffmpeg`).
No Python video bindings are used.
"""
from __future__ import annotations

import json
import logging
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

FFMPEG = "ffmpeg"
FFPROBE = "ffprobe"

# Codecs/containers browsers can play natively without transcoding.
_WEB_SAFE_VIDEO_CODECS = {"h264", "avc1"}
_WEB_SAFE_AUDIO_CODECS = {"aac", "mp3", "mp4a"}
_WEB_SAFE_CONTAINERS = {"mp4", "m4v"}


def ensure_ffmpeg() -> bool:
    """Return True iff both ffmpeg and ffprobe are available on PATH."""
    return shutil.which(FFMPEG) is not None and shutil.which(FFPROBE) is not None


def _to_float(v: Any) -> Optional[float]:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _to_int(v: Any) -> Optional[int]:
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _parse_creation_time(raw: Any) -> Optional[str]:
    """Parse ffprobe's `creation_time` (UTC ISO-8601, e.g. "2024-01-15T10:30:45.000000Z")
    into a local-time naive ISO string ("YYYY-MM-DDTHH:MM:SS") matching how photos store
    their EXIF date. Returns None for missing/malformed values or the bogus 1970 epoch that
    some encoders emit when no real date exists.
    """
    if not raw or not isinstance(raw, str):
        return None
    s = raw.strip()
    if not s:
        return None
    # fromisoformat handles fractional seconds; normalize trailing "Z" to +00:00.
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        return None
    # Treat tz-naive timestamps as UTC (ffprobe emits UTC).
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    local = dt.astimezone()
    if local.year <= 1970:
        return None
    return local.strftime("%Y-%m-%dT%H:%M:%S")


def ffprobe_info(path: str | Path, timeout: int = 120) -> dict[str, Any]:
    """Probe a media file. Returns normalized dict:
    {video_codec, audio_codec, width, height, duration_seconds, container}.
    Raises RuntimeError if ffprobe fails.
    """
    cmd = [
        FFPROBE, "-v", "error",
        "-print_format", "json",
        "-show_streams", "-show_format",
        str(path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {proc.stderr.strip()}")

    data = json.loads(proc.stdout or "{}")
    streams = data.get("streams", [])
    fmt = data.get("format", {})

    video_stream = next((s for s in streams if s.get("codec_type") == "video"), None)
    audio_stream = next((s for s in streams if s.get("codec_type") == "audio"), None)

    duration = _to_float(fmt.get("duration"))
    if duration is None and video_stream:
        duration = _to_float(video_stream.get("duration"))

    raw_creation = fmt.get("tags", {}).get("creation_time")
    if not raw_creation and video_stream:
        raw_creation = (video_stream.get("tags") or {}).get("creation_time")
    creation_time = _parse_creation_time(raw_creation)

    return {
        "video_codec": (video_stream or {}).get("codec_name"),
        "audio_codec": (audio_stream or {}).get("codec_name"),
        "width": _to_int((video_stream or {}).get("width")),
        "height": _to_int((video_stream or {}).get("height")),
        "duration_seconds": duration,
        "container": (fmt.get("format_name") or "").lower(),
        "creation_time": creation_time,
    }


def is_web_safe(info: dict[str, Any], ext: str) -> bool:
    """True if the file plays in browsers as-is (H.264 video, AAC/MP3/no audio, mp4 container)."""
    vcodec = (info.get("video_codec") or "").lower()
    acodec = (info.get("audio_codec") or "").lower()
    ext = ext.lower().lstrip(".")

    if vcodec not in _WEB_SAFE_VIDEO_CODECS:
        return False
    if ext not in _WEB_SAFE_CONTAINERS:
        return False
    if acodec and acodec not in _WEB_SAFE_AUDIO_CODECS:
        return False
    return True


def extract_poster(
    src_path: str | Path,
    out_jpg_path: str | Path,
    duration_seconds: Optional[float] = None,
    max_size: int = 512,
    timeout: int = 120,
) -> None:
    """Extract a single keyframe as a JPEG poster. Raises RuntimeError on failure."""
    # Seek ~1s in to skip black opening frames, unless the clip is shorter than that.
    seek = "0" if (duration_seconds is not None and duration_seconds < 1.0) else "1"
    cmd = [
        FFMPEG, "-y",
        "-ss", seek,
        "-i", str(src_path),
        "-frames:v", "1",
        "-vf", f"scale='min({max_size},iw)':-2",
        "-q:v", "3",
        str(out_jpg_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0 or not Path(out_jpg_path).exists():
        raise RuntimeError(f"ffmpeg poster extraction failed: {proc.stderr.strip()[-2000:]}")


def transcode_to_h264(
    src_path: str | Path,
    out_mp4_path: str | Path,
    timeout: int = 7200,
) -> None:
    """Transcode to a web-safe H.264/AAC MP4 with faststart. Raises RuntimeError on failure."""
    cmd = [
        FFMPEG, "-y",
        "-i", str(src_path),
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "23",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", "128k",
        "-movflags", "+faststart",
        str(out_mp4_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0 or not Path(out_mp4_path).exists():
        raise RuntimeError(f"ffmpeg transcode failed: {proc.stderr.strip()[-2000:]}")
