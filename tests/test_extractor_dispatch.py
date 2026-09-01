"""`core.extractor.load_image` must dispatch on the bytes, not the filename.

63 photos in this library are named `.arw`/`.ARW` but their bytes begin
`FF D8 FF E0 JFIF` — they are JPEGs. `load_image` dispatched on `path.suffix`,
handed them to rawpy, rawpy answered `LibRawFileUnsupportedError`,
`_load_raw_as_rgb` swallowed it and returned None, and what surfaced was
`RuntimeError("Failed to decode RAW image ...")`. Both thumbnail generation and
embedding failed for all 63, and the reads succeeded — this was never storage.

These tests need no storage and no real RAW file: a temp file whose bytes are a
JPEG but whose name ends `.ARW` is the whole defect.
"""
import io
import os
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from core import extractor
from core.extractor import load_image, sniff_extension


def _write(tmpdir, name, data: bytes) -> Path:
    p = Path(tmpdir) / name
    p.write_bytes(data)
    return p


def _jpeg_bytes(size=(24, 16), color=(200, 40, 40)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, "JPEG")
    return buf.getvalue()


def _png_bytes(size=(20, 12)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, (10, 120, 200)).save(buf, "PNG")
    return buf.getvalue()


def _heic_bytes(size=(20, 12)):
    """Real HEIC bytes, or None when pillow-heif is not installed."""
    buf = io.BytesIO()
    try:
        Image.new("RGB", size, (30, 200, 90)).save(buf, "HEIF")
    except Exception:
        return None
    return buf.getvalue()


class TestLoadImageDispatchesOnContent(unittest.TestCase):
    """The 63 mislabelled files must decode."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmpdir = self._tmp.name
        self.addCleanup(self._tmp.cleanup)

    def test_jpeg_bytes_named_arw_loads(self):
        p = _write(self.tmpdir, "Snapseed.arw", _jpeg_bytes(size=(24, 16)))
        img = load_image(p)
        self.assertIsInstance(img, Image.Image)
        self.assertEqual(img.mode, "RGB")
        self.assertEqual(img.size, (24, 16))

    def test_jpeg_bytes_named_uppercase_ARW_loads(self):
        """The real filenames are a mix of `.arw` and `.ARW`."""
        p = _write(self.tmpdir, "DSC01234.ARW", _jpeg_bytes(size=(32, 8)))
        img = load_image(p)
        self.assertEqual(img.size, (32, 8))

    def test_png_bytes_named_arw_loads(self):
        p = _write(self.tmpdir, "shot.arw", _png_bytes(size=(20, 12)))
        img = load_image(p)
        self.assertEqual(img.mode, "RGB")
        self.assertEqual(img.size, (20, 12))

    def test_heic_bytes_named_arw_loads(self):
        data = _heic_bytes(size=(20, 12))
        if data is None:
            self.skipTest("pillow-heif not installed")
        p = _write(self.tmpdir, "live.arw", data)
        img = load_image(p)
        self.assertEqual(img.mode, "RGB")
        self.assertEqual(img.size, (20, 12))

    def test_honestly_named_jpeg_still_loads(self):
        p = _write(self.tmpdir, "holiday.jpg", _jpeg_bytes(size=(16, 16)))
        self.assertEqual(load_image(p).size, (16, 16))

    def test_tiff_header_named_arw_still_goes_to_rawpy(self):
        """Every Sony/Canon/Nikon RAW is TIFF-based, so a TIFF header is not
        evidence of anything. Inconclusive bytes must fall back to the
        extension, which means a `.arw` still reaches rawpy — here with a stub
        TIFF that rawpy cannot decode, so the existing RAW error surfaces."""
        p = _write(self.tmpdir, "genuine.arw", b"II*\x00" + b"\x00" * 64)
        with self.assertRaises(RuntimeError) as ctx:
            load_image(p)
        self.assertIn("Failed to decode RAW image", str(ctx.exception))

    def test_raw_path_is_still_reached_for_raw_bytes(self):
        """Guard against "sniffing" quietly deleting the rawpy branch."""
        calls = []
        real = extractor._load_raw_as_rgb

        def spy(path):
            calls.append(Path(path).name)
            return real(path)

        extractor._load_raw_as_rgb = spy
        self.addCleanup(setattr, extractor, "_load_raw_as_rgb", real)
        p = _write(self.tmpdir, "genuine.cr2", b"II*\x00" + b"\x00" * 64)
        with self.assertRaises(RuntimeError):
            load_image(p)
        self.assertEqual(calls, ["genuine.cr2"])


class TestSniffExtension(unittest.TestCase):
    """The magic-number table itself."""

    def test_jpeg(self):
        self.assertEqual(sniff_extension(b"\xff\xd8\xff\xe0" + b"\x00" * 40), ".jpg")

    def test_png(self):
        self.assertEqual(sniff_extension(b"\x89PNG\r\n\x1a\n" + b"\x00" * 40), ".png")

    def test_gif(self):
        self.assertEqual(sniff_extension(b"GIF89a" + b"\x00" * 40), ".gif")
        self.assertEqual(sniff_extension(b"GIF87a" + b"\x00" * 40), ".gif")

    def test_heic_ftyp_box(self):
        data = b"\x00\x00\x00\x18ftypheic" + b"\x00" * 40
        self.assertEqual(sniff_extension(data), ".heic")

    def test_mp4_ftyp_box_is_not_heic(self):
        data = b"\x00\x00\x00\x18ftypisom" + b"\x00" * 40
        self.assertIsNone(sniff_extension(data))

    def test_webp(self):
        data = b"RIFF" + b"\x00\x00\x00\x00" + b"WEBP" + b"\x00" * 40
        self.assertEqual(sniff_extension(data), ".webp")

    def test_tiff_is_deliberately_not_recognised(self):
        self.assertIsNone(sniff_extension(b"II*\x00" + b"\x00" * 40))
        self.assertIsNone(sniff_extension(b"MM\x00*" + b"\x00" * 40))

    def test_short_and_empty_data(self):
        self.assertIsNone(sniff_extension(b""))
        self.assertIsNone(sniff_extension(b"\xff\xd8"))


class TestSniffingIsSharedNotDuplicated(unittest.TestCase):
    """`scripts/reembed.py` sniffed magic numbers first. There must be exactly
    one implementation, in core, or the two copies will drift."""

    def test_reembed_uses_the_core_implementation(self):
        from scripts import reembed
        self.assertIs(reembed._sniff_extension, sniff_extension)


if __name__ == "__main__":
    unittest.main()
