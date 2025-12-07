from __future__ import annotations
import json
from pathlib import Path
from typing import List, Dict
from core.extractor import load_image

def ensure_thumb(src_path: str, thumb_path: str, size=(256, 256)):
    """
    Always create a JPG thumbnail even for RAW images.
    Raises exception on failure instead of silently failing.
    """
    try:
        img = load_image(Path(src_path))
        if img is None:
            raise Exception(f"Failed to load image from {src_path}")
        img.thumbnail(size)
        # Ensure parent directory exists
        thumb_path_obj = Path(thumb_path)
        thumb_path_obj.parent.mkdir(parents=True, exist_ok=True)
        img.save(thumb_path, "JPEG", quality=100)
        # Verify file was created
        if not Path(thumb_path).exists():
            raise Exception(f"Thumbnail file was not created at {thumb_path}")
    except Exception as e:
        # Re-raise with more context
        raise Exception(f"Thumbnail generation failed for {src_path}: {str(e)}") from e


def build_gallery(
    items: List[Dict],
    faces: List[Dict] | None = None,
    output_dir: str = "./chitra",
) -> str:
    """
    Build an HTML gallery with:
      - all photos
      - optional faces/person sidebar.

    items: list of {file, thumb, tags}
    faces: list of {
        "face_id", "thumb", "person_name" or None, "photo_path"
    }
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    html_path = out_dir / "index.html"

    faces = faces or []

    items_json = json.dumps(items)
    faces_json = json.dumps(faces)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Chitra Gallery</title>
  <style>
    body {{
      margin: 0;
      font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      display: flex;
      height: 100vh;
      color: #222;
    }}
    .sidebar {{
      width: 260px;
      border-right: 1px solid #ddd;
      padding: 16px;
      box-sizing: border-box;
      overflow-y: auto;
      background: #fafafa;
    }}
    .sidebar h2 {{
      margin-top: 0;
      font-size: 18px;
    }}
    .sidebar-section {{
      margin-bottom: 20px;
    }}
    .sidebar ul {{
      list-style: none;
      padding-left: 0;
      margin: 8px 0 0 0;
    }}
    .sidebar li {{
      margin-bottom: 6px;
      cursor: pointer;
      padding: 4px 8px;
      border-radius: 6px;
    }}
    .sidebar li.active {{
      background: #e91e63;
      color: white;
    }}
    .sidebar li:hover {{
      background: #f0f0f0;
    }}
    .main {{
      flex: 1;
      display: flex;
      flex-direction: column;
      overflow: hidden;
    }}
    .topbar {{
      padding: 10px 16px;
      border-bottom: 1px solid #ddd;
      display: flex;
      align-items: center;
      justify-content: space-between;
    }}
    .topbar h1 {{
      font-size: 20px;
      margin: 0;
    }}
    .grid {{
      flex: 1;
      overflow-y: auto;
      padding: 16px;
      box-sizing: border-box;
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(160px, 1fr));
      gap: 12px;
    }}
    .card {{
      border-radius: 10px;
      box-shadow: 0 1px 3px rgba(0,0,0,0.12);
      overflow: hidden;
      background: white;
      display: flex;
      flex-direction: column;
    }}
    .card img {{
      width: 100%;
      height: 160px;
      object-fit: cover;
      display: block;
    }}
    .card-body {{
      padding: 8px;
      font-size: 12px;
    }}
    .tags {{
      color: #666;
      font-size: 11px;
      margin-top: 4px;
    }}
    .face-thumb {{
      width: 32px;
      height: 32px;
      border-radius: 999px;
      object-fit: cover;
      margin-right: 8px;
      border: 1px solid #ddd;
      background: #f5f5f5;
    }}
    .sidebar-face-item {{
      display: flex;
      align-items: center;
    }}
    .sidebar-face-label {{
      font-size: 13px;
    }}
    .pill {{
      display: inline-block;
      padding: 2px 6px;
      border-radius: 999px;
      background: #eee;
      font-size: 10px;
      margin-left: 6px;
    }}
  </style>
</head>
<body>
  <div class="sidebar">
    <div class="sidebar-section">
      <h2>View</h2>
      <ul id="view-list">
        <li data-view="all" class="active">All photos</li>
        <li data-view="faces">Faces</li>
      </ul>
    </div>
    <div class="sidebar-section" id="faces-section" style="display:none;">
      <h2>Faces</h2>
      <ul id="faces-list">
        <!-- filled by JS -->
      </ul>
    </div>
  </div>
  <div class="main">
    <div class="topbar">
      <h1 id="title">All photos</h1>
      <div id="info"></div>
    </div>
    <div class="grid" id="grid">
      <!-- cards injected by JS -->
    </div>
  </div>

  <script>
    const ITEMS = {items_json};
    const FACES = {faces_json};

    let currentView = "all";
    let currentPerson = null;

    function renderAllPhotos() {{
      const grid = document.getElementById("grid");
      grid.innerHTML = "";
      document.getElementById("title").innerText = "All photos";
      document.getElementById("info").innerText = ITEMS.length + " photos";

      ITEMS.forEach(it => {{
        const card = document.createElement("div");
        card.className = "card";
        const img = document.createElement("img");
        img.src = it.thumb;
        img.alt = it.file;
        const body = document.createElement("div");
        body.className = "card-body";
        const p = document.createElement("div");
        p.textContent = it.file;
        const tags = document.createElement("div");
        tags.className = "tags";
        tags.textContent = it.tags || "";
        body.appendChild(p);
        body.appendChild(tags);
        card.appendChild(img);
        card.appendChild(body);
        grid.appendChild(card);
      }});
    }}

    function buildFacesSidebar() {{
      const section = document.getElementById("faces-section");
      const list = document.getElementById("faces-list");
      list.innerHTML = "";

      if (!FACES.length) {{
        section.style.display = "none";
        return;
      }}
      section.style.display = "block";

      // Group by person_name
      const groups = {{}};
      for (const f of FACES) {{
        const key = f.person_name || "Unknown";
        if (!groups[key]) groups[key] = [];
        groups[key].push(f);
      }}

      Object.keys(groups).sort().forEach(name => {{
        const faces = groups[name];
        const li = document.createElement("li");
        li.className = "sidebar-face-item";
        li.dataset.person = name;

        const btn = document.createElement("div");
        btn.style.display = "flex";
        btn.style.alignItems = "center";
        btn.style.width = "100%";

        const img = document.createElement("img");
        img.className = "face-thumb";
        img.src = faces[0].thumb || faces[0].photo_path;  // fallback

        const label = document.createElement("div");
        label.className = "sidebar-face-label";
        label.innerHTML = name + '<span class="pill">' + faces.length + '</span>';

        btn.appendChild(img);
        btn.appendChild(label);
        li.appendChild(btn);

        li.addEventListener("click", () => {{
          currentView = "faces";
          currentPerson = name;
          Array.from(document.querySelectorAll("#faces-list li")).forEach(x => x.classList.remove("active"));
          li.classList.add("active");
          renderFacesForPerson(name);
          // also set sidebar main view:
          document.querySelector('#view-list li[data-view="faces"]').classList.add("active");
          document.querySelector('#view-list li[data-view="all"]').classList.remove("active");
        }});

        list.appendChild(li);
      }});
    }}

    function renderFacesForPerson(name) {{
      const grid = document.getElementById("grid");
      grid.innerHTML = "";

      let subset = FACES;
      if (name) {{
        subset = FACES.filter(f => (f.person_name || "Unknown") === name);
      }}

      document.getElementById("title").innerText = "Faces: " + (name || "All");
      document.getElementById("info").innerText = subset.length + " faces";

      subset.forEach(f => {{
        const card = document.createElement("div");
        card.className = "card";
        const img = document.createElement("img");
        img.src = f.thumb || f.photo_path;
        img.alt = f.photo_path;
        const body = document.createElement("div");
        body.className = "card-body";
        const p = document.createElement("div");
        p.textContent = f.photo_path;
        const tags = document.createElement("div");
        tags.className = "tags";
        tags.textContent = f.person_name || "Unknown face";
        body.appendChild(p);
        body.appendChild(tags);
        card.appendChild(img);
        card.appendChild(body);
        grid.appendChild(card);
      }});
    }}

    function setupViewSwitch() {{
      const list = document.getElementById("view-list");
      Array.from(list.querySelectorAll("li")).forEach(li => {{
        li.addEventListener("click", () => {{
          Array.from(list.querySelectorAll("li")).forEach(x => x.classList.remove("active"));
          li.classList.add("active");
          const view = li.dataset.view;
          currentView = view;

          if (view === "all") {{
            currentPerson = null;
            renderAllPhotos();
          }} else if (view === "faces") {{
            currentPerson = null;
            renderFacesForPerson(null);
          }}
        }});
      }});
    }}

    // Init
    renderAllPhotos();
    buildFacesSidebar();
    setupViewSwitch();
  </script>
</body>
</html>
"""

    html_path.write_text(html, encoding="utf-8")
    return str(html_path)
