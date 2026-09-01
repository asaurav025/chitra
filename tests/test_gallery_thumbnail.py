"""`core.gallery.ensure_thumb` must fail with a typed, machine-readable error.

Today a decode failure leaves `ensure_thumb` as a bare `Exception("Thumbnail
generation failed for ...")`, which propagates uncaught out of the HTTP handler
and becomes a bare 500 with no detail. The HTTP layer needs something it can
catch and translate, and it needs to tell "these bytes are not an image"
(the client should see 404/422, retrying will never help) apart from "I could
not read the source" (a storage problem).

`ThumbnailUnavailable` is that contract. It is the only exception `ensure_thumb`
raises for an unusable source; a broken *destination* stays an ordinary error,
because an unwritable thumbnail directory really is a server fault.
"""
import io
import os
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from core.gallery import ThumbnailUnavailable, ensure_thumb


def _jpeg_bytes(size=(40, 30)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, (10, 90, 160)).save(buf, "JPEG")
    return buf.getvalue()


class TestThumbnailUnavailable(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmpdir = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        self.thumb = str(self.tmpdir / "out" / "thumb.jpg")

    def test_it_is_an_exception_carrying_a_reason_and_a_path(self):
        exc = ThumbnailUnavailable("decode_failed", "photos/2026/04/x.arw")
        self.assertIsInstance(exc, Exception)
        self.assertEqual(exc.reason, "decode_failed")
        self.assertEqual(exc.src_path, "photos/2026/04/x.arw")
        self.assertIn("decode_failed", str(exc))

    def test_reason_is_one_of_the_documented_values(self):
        self.assertEqual(
            set(ThumbnailUnavailable.REASONS), {"decode_failed", "source_unreadable"}
        )

    def test_undecodable_bytes_raise_decode_failed(self):
        src = self.tmpdir / "not-an-image.jpg"
        src.write_bytes(b"this is not an image, not even a little bit")
        with self.assertRaises(ThumbnailUnavailable) as ctx:
            ensure_thumb(str(src), self.thumb)
        self.assertEqual(ctx.exception.reason, "decode_failed")
        self.assertEqual(ctx.exception.src_path, str(src))

    def test_undecodable_raw_raises_decode_failed_not_runtimeerror(self):
        """A genuine RAW that rawpy cannot decode surfaced as
        RuntimeError("Failed to decode RAW image ...") — a bare 500."""
        src = self.tmpdir / "broken.arw"
        src.write_bytes(b"II*\x00" + b"\x00" * 64)
        with self.assertRaises(ThumbnailUnavailable) as ctx:
            ensure_thumb(str(src), self.thumb)
        self.assertEqual(ctx.exception.reason, "decode_failed")

    def test_missing_source_raises_source_unreadable(self):
        with self.assertRaises(ThumbnailUnavailable) as ctx:
            ensure_thumb(str(self.tmpdir / "gone.jpg"), self.thumb)
        self.assertEqual(ctx.exception.reason, "source_unreadable")

    def test_unreadable_source_raises_source_unreadable(self):
        src = self.tmpdir / "locked.jpg"
        src.write_bytes(_jpeg_bytes())
        os.chmod(src, 0o000)
        self.addCleanup(os.chmod, src, 0o644)
        if os.access(src, os.R_OK):
            self.skipTest("running as a user that ignores file permissions")
        with self.assertRaises(ThumbnailUnavailable) as ctx:
            ensure_thumb(str(src), self.thumb)
        self.assertEqual(ctx.exception.reason, "source_unreadable")

    def test_the_original_cause_is_chained(self):
        src = self.tmpdir / "not-an-image.jpg"
        src.write_bytes(b"nope")
        with self.assertRaises(ThumbnailUnavailable) as ctx:
            ensure_thumb(str(src), self.thumb)
        self.assertIsNotNone(ctx.exception.__cause__)


class TestThumbnailSucceeds(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmpdir = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def test_honest_jpeg_thumbnails(self):
        src = self.tmpdir / "holiday.jpg"
        src.write_bytes(_jpeg_bytes(size=(1200, 800)))
        thumb = str(self.tmpdir / "out" / "thumb.jpg")
        ensure_thumb(str(src), thumb)
        self.assertTrue(os.path.exists(thumb))
        with Image.open(thumb) as img:
            self.assertLessEqual(max(img.size), 512)

    def test_jpeg_bytes_named_arw_thumbnails(self):
        """The 63 mislabelled photos: all have NULL thumb_path because this
        path raised. Content-based dispatch in load_image fixes them here."""
        src = self.tmpdir / "Snapseed.ARW"
        src.write_bytes(_jpeg_bytes(size=(1000, 750)))
        thumb = str(self.tmpdir / "out" / "63.jpg")
        ensure_thumb(str(src), thumb)
        self.assertTrue(os.path.exists(thumb))
        with Image.open(thumb) as img:
            self.assertEqual(img.format, "JPEG")
            self.assertLessEqual(max(img.size), 512)


if __name__ == "__main__":
    unittest.main()
