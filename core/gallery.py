from __future__ import annotations
import os
from pathlib import Path
from typing import List, Dict
from PIL import Image
from jinja2 import Template

GALLERY_HTML = """<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Chitra Gallery</title>
  <style>
    body { font-family: ui-sans-serif, system-ui, Arial; margin: 0; padding: 1rem; background:#111; color:#eee;}
    .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr)); gap: 10px; }
    .card { background:#1a1a1a; padding:8px; border-radius:10px; }
    .thumb { width: 100%; height: 170px; object-fit: cover; border-radius: 8px; }
    .meta { font-size: 12px; color:#aaa; margin-top:6px;}
    h1 { font-weight: 600; }
  </style>
</head>
<body>
  <h1>Chitra Gallery</h1>
  <div class="grid">
    {% for item in items %}
      <div class="card">
        <img class="thumb" src="{{ item.thumb }}" alt="thumb"/>
        <div class="meta">{{ item.file }}</div>
        {% if item.tags %}<div class="meta">tags: {{ item.tags }}</div>{% endif %}
      </div>
    {% endfor %}
  </div>
</body>
</html>
"""

def ensure_thumb(image_path: str, thumb_path: str, size=(320, 320)) -> None:
    os.makedirs(os.path.dirname(thumb_path), exist_ok=True)
    try:
        img = Image.open(image_path).convert("RGB")
        img.thumbnail(size)
        img.save(thumb_path, "JPEG", quality=85, optimize=True)
    except Exception:
        pass

def build_gallery(items: List[Dict], output_dir: str, base_dir: str = ".") -> str:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    html_path = os.path.join(output_dir, "index.html")
    tpl = Template(GALLERY_HTML)
    with open(html_path, "w") as f:
        f.write(tpl.render(items=items))
    return html_path
