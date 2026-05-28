"""Unit tests for core.video (pure logic + ffprobe JSON parsing) and extractor video branch."""
import json
import os
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

from core import video
from core import extractor


class TestIsWebSafe(unittest.TestCase):
    def test_h264_aac_mp4_is_web_safe(self):
        info = {"video_codec": "h264", "audio_codec": "aac"}
        self.assertTrue(video.is_web_safe(info, ".mp4"))

    def test_h264_no_audio_mp4_is_web_safe(self):
        info = {"video_codec": "h264", "audio_codec": None}
        self.assertTrue(video.is_web_safe(info, "mp4"))

    def test_hevc_mov_not_web_safe(self):
        info = {"video_codec": "hevc", "audio_codec": "aac"}
        self.assertFalse(video.is_web_safe(info, ".mov"))

    def test_h264_in_mov_not_web_safe(self):
        # Right codec, wrong container -> still needs remux/transcode.
        info = {"video_codec": "h264", "audio_codec": "aac"}
        self.assertFalse(video.is_web_safe(info, ".mov"))

    def test_h264_with_opus_audio_not_web_safe(self):
        info = {"video_codec": "h264", "audio_codec": "opus"}
        self.assertFalse(video.is_web_safe(info, ".mp4"))

    def test_vp9_webm_not_web_safe(self):
        info = {"video_codec": "vp9", "audio_codec": "opus"}
        self.assertFalse(video.is_web_safe(info, ".webm"))


class TestFfprobeInfo(unittest.TestCase):
    def _fake_run(self, payload, returncode=0, stderr=""):
        proc = mock.Mock()
        proc.returncode = returncode
        proc.stdout = json.dumps(payload)
        proc.stderr = stderr
        return proc

    def test_parses_streams_and_format(self):
        payload = {
            "streams": [
                {"codec_type": "video", "codec_name": "hevc", "width": 1920, "height": 1080},
                {"codec_type": "audio", "codec_name": "aac"},
            ],
            "format": {"duration": "12.5", "format_name": "mov,mp4,m4a,3gp,3g2,mj2"},
        }
        with mock.patch("core.video.subprocess.run", return_value=self._fake_run(payload)):
            info = video.ffprobe_info("/tmp/x.mov")
        self.assertEqual(info["video_codec"], "hevc")
        self.assertEqual(info["audio_codec"], "aac")
        self.assertEqual(info["width"], 1920)
        self.assertEqual(info["height"], 1080)
        self.assertEqual(info["duration_seconds"], 12.5)
        self.assertIn("mp4", info["container"])

    def test_raises_on_failure(self):
        with mock.patch("core.video.subprocess.run", return_value=self._fake_run({}, returncode=1, stderr="boom")):
            with self.assertRaises(RuntimeError):
                video.ffprobe_info("/tmp/missing.mov")

    def test_handles_missing_audio_stream(self):
        payload = {
            "streams": [{"codec_type": "video", "codec_name": "h264", "width": 640, "height": 480}],
            "format": {"duration": "3.0", "format_name": "mp4"},
        }
        with mock.patch("core.video.subprocess.run", return_value=self._fake_run(payload)):
            info = video.ffprobe_info("/tmp/x.mp4")
        self.assertIsNone(info["audio_codec"])
        self.assertTrue(video.is_web_safe(info, ".mp4"))

    def test_creation_time_from_format_tags(self):
        payload = {
            "streams": [{"codec_type": "video", "codec_name": "h264"}],
            "format": {
                "duration": "3.0",
                "format_name": "mp4",
                "tags": {"creation_time": "2024-01-15T10:30:45.000000Z"},
            },
        }
        with mock.patch("core.video.subprocess.run", return_value=self._fake_run(payload)):
            info = video.ffprobe_info("/tmp/x.mp4")
        expected = datetime(2024, 1, 15, 10, 30, 45, tzinfo=timezone.utc).astimezone().strftime(
            "%Y-%m-%dT%H:%M:%S"
        )
        self.assertEqual(info["creation_time"], expected)

    def test_creation_time_falls_back_to_stream_tags(self):
        payload = {
            "streams": [
                {
                    "codec_type": "video",
                    "codec_name": "h264",
                    "tags": {"creation_time": "2024-01-15T10:30:45Z"},
                }
            ],
            "format": {"duration": "3.0", "format_name": "mp4"},
        }
        with mock.patch("core.video.subprocess.run", return_value=self._fake_run(payload)):
            info = video.ffprobe_info("/tmp/x.mp4")
        self.assertIsNotNone(info["creation_time"])

    def test_creation_time_absent(self):
        payload = {
            "streams": [{"codec_type": "video", "codec_name": "h264"}],
            "format": {"duration": "3.0", "format_name": "mp4"},
        }
        with mock.patch("core.video.subprocess.run", return_value=self._fake_run(payload)):
            info = video.ffprobe_info("/tmp/x.mp4")
        self.assertIsNone(info["creation_time"])


class TestParseCreationTime(unittest.TestCase):
    def _expected_local(self, y, mo, d, h, mi, s):
        return datetime(y, mo, d, h, mi, s, tzinfo=timezone.utc).astimezone().strftime(
            "%Y-%m-%dT%H:%M:%S"
        )

    def test_utc_with_fractional_and_z(self):
        self.assertEqual(
            video._parse_creation_time("2024-01-15T10:30:45.000000Z"),
            self._expected_local(2024, 1, 15, 10, 30, 45),
        )

    def test_naive_treated_as_utc(self):
        self.assertEqual(
            video._parse_creation_time("2024-01-15T10:30:45"),
            self._expected_local(2024, 1, 15, 10, 30, 45),
        )

    def test_epoch_zero_rejected(self):
        self.assertIsNone(video._parse_creation_time("1970-01-01T00:00:00.000000Z"))

    def test_malformed_returns_none(self):
        self.assertIsNone(video._parse_creation_time("not a date"))

    def test_empty_and_none_return_none(self):
        self.assertIsNone(video._parse_creation_time(""))
        self.assertIsNone(video._parse_creation_time(None))


class TestExtractorVideoBranch(unittest.TestCase):
    def test_is_video(self):
        self.assertTrue(extractor.is_video("clip.mp4"))
        self.assertTrue(extractor.is_video("/a/b/IMG_1234.MOV"))
        self.assertFalse(extractor.is_video("photo.jpg"))
        self.assertFalse(extractor.is_video("raw.arw"))

    def test_collect_metadata_video_skips_image_steps(self):
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(b"not a real video, just bytes for checksum")
            tmp = f.name
        try:
            # Guard: must not call the image-only loaders on a video. ffprobe yields a date.
            with mock.patch("core.extractor.compute_phash", side_effect=AssertionError("phash called on video")), \
                 mock.patch("core.extractor.get_exif", side_effect=AssertionError("exif called on video")), \
                 mock.patch("core.video.ffprobe_info", return_value={"creation_time": "2024-01-15T10:30:45"}):
                meta = extractor.collect_metadata(Path(tmp))
            self.assertEqual(meta["media_type"], "video")
            self.assertEqual(meta["phash"], "")
            self.assertEqual(meta["created_at"], "2024-01-15T10:30:45")
            self.assertEqual(meta["exif_datetime"], "2024-01-15T10:30:45")
            self.assertIsNone(meta["latitude"])
            self.assertTrue(meta["checksum"])
        finally:
            os.unlink(tmp)

    def test_collect_metadata_video_falls_back_to_mtime(self):
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(b"not a real video, just bytes for checksum")
            tmp = f.name
        try:
            # No creation_time (or ffprobe failure) -> mtime for created_at, empty exif_datetime.
            with mock.patch("core.video.ffprobe_info", return_value={"creation_time": None}):
                meta = extractor.collect_metadata(Path(tmp))
            self.assertEqual(meta["media_type"], "video")
            self.assertEqual(meta["exif_datetime"], "")
            self.assertTrue(meta["created_at"])
        finally:
            os.unlink(tmp)

    def test_collect_metadata_video_survives_ffprobe_error(self):
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(b"not a real video, just bytes for checksum")
            tmp = f.name
        try:
            with mock.patch("core.video.ffprobe_info", side_effect=RuntimeError("ffprobe boom")):
                meta = extractor.collect_metadata(Path(tmp))
            self.assertEqual(meta["media_type"], "video")
            self.assertEqual(meta["exif_datetime"], "")
            self.assertTrue(meta["created_at"])
        finally:
            os.unlink(tmp)


if __name__ == "__main__":
    unittest.main()
