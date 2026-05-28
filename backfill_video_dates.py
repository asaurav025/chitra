#!/usr/bin/env python3
"""Backfill capture dates for already-uploaded videos.

Videos ingested before the creation_time fix have `created_at` set to their upload
time and an empty `exif_datetime`. This script re-probes each stored video from
MinIO with ffprobe, extracts `creation_time`, and updates the capture-date columns.

Idempotent: safe to re-run. Videos with no resolvable creation_time are left as-is.
"""

import os
import sys
import tempfile
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from core import db, video
from core.jobs import _get_storage_client

DB_PATH = os.environ.get("CHITRA_DB_PATH", db.DB_DEFAULT_PATH)


def main():
    print("=" * 60)
    print("Backfilling video capture dates from ffprobe creation_time")
    print("=" * 60)
    print(f"Database: {DB_PATH}")
    print()

    if not os.path.exists(DB_PATH):
        print(f"Error: Database not found at {DB_PATH}")
        sys.exit(1)

    if not video.ensure_ffmpeg():
        print("Error: ffmpeg/ffprobe not found on PATH; cannot probe videos")
        sys.exit(1)

    conn = db.connect(DB_PATH)
    storage = _get_storage_client()

    cur = conn.cursor()
    cur.execute(
        "SELECT id, file_path FROM photos WHERE media_type='video' ORDER BY id ASC"
    )
    rows = cur.fetchall()
    print(f"Found {len(rows)} video(s).\n")

    updated = skipped = failed = 0
    for row in rows:
        photo_id, file_path = row["id"], row["file_path"]
        tmp_path = None
        try:
            ext = Path(file_path).suffix
            with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
                tmp_path = tmp.name
            storage.download_to_path(file_path, tmp_path)

            creation_time = video.ffprobe_info(tmp_path).get("creation_time")
            if creation_time:
                db.update_capture_date(conn, photo_id, creation_time, creation_time)
                updated += 1
                print(f"  [{photo_id}] {file_path} -> {creation_time}")
            else:
                skipped += 1
                print(f"  [{photo_id}] {file_path} -> no creation_time, skipped")
        except Exception as e:
            failed += 1
            print(f"  [{photo_id}] {file_path} -> ERROR: {e}")
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

    conn.close()
    print()
    print(f"Done. updated={updated} skipped={skipped} failed={failed}")


if __name__ == "__main__":
    main()
