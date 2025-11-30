from __future__ import annotations
import os
from pathlib import Path
from typing import List, Dict

from core.extractor import load_image


def ensure_thumb(src_path: str, thumb_path: str, size=(320, 320)) -> None:
    """
    Create a thumbnail for src_path and save to thumb_path (JPEG).
    Works for RAW and normal images via load_image().
    """
    src = Path(src_path)
    dst = Path(thumb_path)
    dst.parent.mkdir(parents=True, exist_ok=True)

    img = load_image(src)
    img.thumbnail(size)
    # Always save thumbnail as JPEG
    img.save(dst, format="JPEG", quality=85)


def build_gallery(items: List[Dict], output_dir: str) -> str:
    """
    Build a very simple static HTML gallery.

    items: list of dicts like:
        {
          "file": "/absolute/path/to/original",
          "thumb": "thumbnails/DSC02700.ARW.jpg",
          "tags": "cat, garden",
        }

    Returns the path to the generated HTML file as a string.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    html_path = out_dir / "index.html"

    rows = []
    for it in items:
        thumb_rel = it["thumb"]
        file_path = it["file"]
        tags = it.get("tags", "")

        rows.append(
            f"""
            <div class="item">
              <a href="{file_path}" target="_blank">
                <img src="{thumb_rel}" loading="lazy" />
              </a>
              <div class="meta">
                <div class="path">{file_path}</div>
                <div class="tags">{tags}</div>
              </div>
            </div>
            """
        )

    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Chitra Gallery</title>
  <style>
    body {{
      font-family: system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
      background: #111;
      color: #eee;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
      gap: 12px;
      padding: 16px;
    }}
    .item {{
      background: #1b1b1b;
      border-radius: 8px;
      padding: 8px;
      box-shadow: 0 2px 4px rgba(0,0,0,0.4);
    }}
    img {{
      max-width: 100%;
      border-radius: 4px;
      display: block;
      margin-bottom: 6px;
    }}
    .meta {{
      font-size: 12px;
      color: #aaa;
      word-break: break-all;
    }}
    .tags {{
      margin-top: 4px;
      color: #6cf;
    }}
  </style>
</head>
<body>
  <h1 style="padding:16px;">Chitra Gallery</h1>
  <div class="grid">
    {''.join(rows)}
  </div>
</body>
</html>
"""

    html_path.write_text(html, encoding="utf-8")
    return str(html_path)
