from __future__ import annotations
import os
import sys
import json
from pathlib import Path
from typing import Optional, Dict, List

import numpy as np
import typer
from rich import print
from rich.table import Table
from tqdm import tqdm

from core import db
from core.extractor import iter_images, collect_metadata, load_image
from core.embedder import ClipEmbedder
from core.tagger import auto_tags
from core.cluster import threshold_clusters
from core.gallery import build_gallery
from core.face import face_encodings

app = typer.Typer(add_completion=False, help="Chitra — Photo Intelligence CLI")


# ---------------------------------------------------------------------------
# INIT
# ---------------------------------------------------------------------------
@app.command()
def init(db_path: str = typer.Option("photo.db", "--db", help="SQLite database path")):
    """Create / migrate the database schema."""
    db.init_db(db_path)
    print(f"[green]Initialized[/green] SQLite at {db_path}")


# ---------------------------------------------------------------------------
# SCAN
# ---------------------------------------------------------------------------
@app.command()
def scan(
    path: str = typer.Argument(..., help="Root folder to scan"),
    db_path: str = typer.Option("photo.db", "--db"),
    incremental: bool = typer.Option(True, help="Skip files already in DB with same checksum"),
):
    """Index file metadata + EXIF (no embeddings)."""
    conn = db.connect(db_path)

    existing = {}
    if incremental:
        cur = conn.cursor()
        cur.execute("SELECT file_path, checksum FROM photos")
        existing = {row["file_path"]: row["checksum"] for row in cur.fetchall()}

    total = 0
    skipped = 0

    # If you have many files, not wrapping list() avoids full materialization
    for p in tqdm(list(iter_images(path)), desc="Scanning"):
        file_path_str = str(p)

        if incremental and file_path_str in existing:
            # Quick skip based on file size
            st = p.stat()
            cur = conn.cursor()
            cur.execute("SELECT size FROM photos WHERE file_path=?", (file_path_str,))
            row = cur.fetchone()
            if row and row["size"] == st.st_size:
                skipped += 1
                continue

        meta = collect_metadata(p)
        db.upsert_photo(conn, **meta)
        total += 1

    conn.close()
    print(f"[cyan]Indexed {total} photos[/cyan]", end="")
    if skipped > 0:
        print(f" [dim](skipped {skipped} unchanged)[/dim]")
    else:
        print()


# ---------------------------------------------------------------------------
# ANALYZE (CLIP embeddings + auto-tags)
# ---------------------------------------------------------------------------
@app.command()
def analyze(
    db_path: str = typer.Option("photo.db", "--db"),
    limit: Optional[int] = typer.Option(None, help="Process only N photos (for testing)"),
    tag_k: int = typer.Option(5, help="Top-K tags to store"),
    incremental: bool = typer.Option(
        True,
        "--incremental/--no-incremental",
        help="Skip photos that already have embeddings",
    ),
):
    """Compute CLIP embeddings + auto-tags and store into SQLite."""
    conn = db.connect(db_path)
    em = ClipEmbedder()

    if incremental:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT p.id, p.file_path
            FROM photos p
            LEFT JOIN embeddings e ON p.id = e.photo_id
            WHERE e.photo_id IS NULL
            """
        )
        rows = cur.fetchall()
    else:
        rows = list(db.iter_photos(conn))

    if limit:
        rows = rows[:limit]

    if not rows:
        print("[green]All photos already analyzed![/green]")
        conn.close()
        return

    for row in tqdm(rows, desc="Analyzing"):
        if isinstance(row, tuple):
            pid, file_path = row
        else:
            pid, file_path = row["id"], row["file_path"]

        try:
            vec = em.image_embedding(file_path)
            db.put_embedding(conn, pid, vec.tobytes(), vec.shape[0])

            # Clear old tags and add new ones
            conn.execute("DELETE FROM tags WHERE photo_id=?", (pid,))
            for tag, score in auto_tags(em, file_path, k=tag_k):
                db.add_tag(conn, pid, tag, float(score))

        except Exception as e:
            print(f"[red]Failed[/red] {file_path}: {e}")

    conn.close()
    print("[green]Analyze complete[/green]")


# ---------------------------------------------------------------------------
# CLUSTER (photo-level)
# ---------------------------------------------------------------------------
@app.command()
def cluster(
    threshold: float = typer.Option(
        0.78,
        "--threshold",
        help="Similarity threshold (0 to 1)",
    ),
    db_path: str = typer.Option("photo.db", "--db"),
):
    """Group similar photos using FAISS + threshold union-find."""
    conn = db.connect(db_path)
    items = db.get_embeddings(conn)

    if not items:
        print("[yellow]No embeddings found. Run 'analyze' first.[/yellow]")
        conn.close()
        return

    vectors: Dict[int, np.ndarray] = {}
    for photo_id, dim, vec_bytes in items:
        v = np.frombuffer(vec_bytes, dtype=np.float32)
        if v.shape[0] != dim:
            # Older data or corrupted row
            continue
        v = v / (np.linalg.norm(v) + 1e-9)
        vectors[photo_id] = v

    clusters = threshold_clusters(vectors, threshold=threshold)

    for cid, members in clusters.items():
        for pid in members:
            db.assign_cluster(conn, pid, cid, 1.0)

    conn.close()
    total = sum(len(m) for m in clusters.values())
    print(f"[green]Clustered[/green] {total} photos into {len(clusters)} clusters")


# ---------------------------------------------------------------------------
# FACES — DETECT
# ---------------------------------------------------------------------------
@app.command("faces-detect")
def faces_detect(
    db_path: str = typer.Option("photo.db", "--db"),
    limit: Optional[int] = typer.Option(None, help="Process only N photos"),
    min_score: float = typer.Option(0.5, help="Minimum face detection score"),
    thumb_size: int = typer.Option(160, help="Face thumbnail size (px)"),
):
    """
    Detect faces using InsightFace and store them in the DB, with thumbnails.

    Steps:
      - Iterates over all photos
      - Detect faces
      - Stores embedding + bbox + person_id (None)
      - Saves face thumbnails under ./faces/<face_id>.jpg (later)
    """
    from PIL import Image

    conn = db.connect(db_path)
    rows = list(db.iter_photos(conn))
    if limit:
        rows = rows[:limit]

    if not rows:
        print("[yellow]No photos found in DB. Run 'scan' first.[/yellow]")
        conn.close()
        return

    face_count = 0
    faces_root = Path("faces")
    faces_root.mkdir(exist_ok=True)

    for pid, file_path in tqdm(rows, desc="Detecting faces"):
        try:
            faces = face_encodings(file_path)
            if not faces:
                continue

            img = load_image(Path(file_path))

            for face_idx, f in enumerate(faces):
                if f["score"] < min_score:
                    continue

                embedding = np.array(f["embedding"], dtype=np.float32)
                bbox_x, bbox_y, bbox_w, bbox_h = f["bbox"]

                # Save face row
                db.add_face(
                    conn,
                    photo_id=pid,
                    face_index=face_idx,
                    embedding_bytes=embedding.tobytes(),
                    bbox_x=bbox_x,
                    bbox_y=bbox_y,
                    bbox_w=bbox_w,
                    bbox_h=bbox_h,
                    person_id=None,
                )

                # Build thumbnail path using temporary id logic
                # We'll fetch actual face_id afterwards
                cur = conn.cursor()
                cur.execute(
                    "SELECT id FROM faces WHERE photo_id=? AND face_index=?",
                    (pid, face_idx),
                )
                row = cur.fetchone()
                if row:
                    face_id = row["id"]
                    x, y, w, h = bbox_x, bbox_y, bbox_w, bbox_h
                    crop = img.crop((x, y, x + w, y + h))
                    crop = crop.resize((thumb_size, thumb_size))
                    thumb_path = faces_root / f"face_{face_id}.jpg"
                    crop.save(thumb_path, "JPEG", quality=100)
                    db.set_face_thumb(conn, face_id, str(thumb_path))

                    face_count += 1

        except Exception as e:
            print(f"[red]Failed[/red] {file_path}: {e}")

    conn.close()
    print(f"[green]Detected {face_count} faces[/green]")


# ---------------------------------------------------------------------------
# FACES — CLUSTER → PERSONS
# ---------------------------------------------------------------------------
@app.command("faces-cluster")
def faces_cluster(
    db_path: str = typer.Option("photo.db", "--db"),
    threshold: float = typer.Option(0.75, help="Cosine similarity threshold for same person"),
):
    """
    Cluster faces into persons using FAISS on face embeddings.
    Creates automatic person names like 'Person 1', 'Person 2', etc.
    """
    import faiss

    conn = db.connect(db_path)
    rows = db.get_faces_embeddings(conn)
    if not rows:
        print("[yellow]No faces found. Run 'faces-detect' first.[/yellow]")
        conn.close()
        return

    face_ids = []
    vecs = []

    for row in rows:
        fid = row["id"]
        emb_bytes = row["embedding"]
        v = np.frombuffer(emb_bytes, dtype=np.float32)
        if v.size == 0:
            continue
        face_ids.append(fid)
        vecs.append(v)

    xb = np.stack(vecs).astype("float32")
    faiss.normalize_L2(xb)
    dim = xb.shape[1]

    index = faiss.IndexFlatIP(dim)
    index.add(xb)

    # For each face, find neighbors and cluster via a simple union-find
    n = xb.shape[0]
    k = min(10, n)  # check top-10 nearest faces per face

    # Union-Find
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    D, I = index.search(xb, k)
    for i in range(n):
        for j in range(1, k):  # skip self at j=0
            if I[i, j] < 0:
                continue
            if D[i, j] >= threshold:
                union(i, I[i, j])

    # Collect clusters
    clusters: Dict[int, List[int]] = {}
    for i in range(n):
        root = find(i)
        clusters.setdefault(root, []).append(i)

    # Assign persons
    person_idx = 1
    for root, members in clusters.items():
        # create person name
        person_name = f"Person {person_idx}"
        person_id = db.get_or_create_person(conn, person_name)
        person_idx += 1

        for m in members:
            fid = face_ids[m]
            db.set_face_person(conn, fid, person_id)

    conn.close()
    print(f"[green]Clustered faces into {len(clusters)} persons[/green]")


# ---------------------------------------------------------------------------
# FACES — LIST
# ---------------------------------------------------------------------------
@app.command("faces-list")
def faces_list(
    db_path: str = typer.Option("photo.db", "--db"),
):
    """List all detected faces."""
    conn = db.connect(db_path)
    rows = db.iter_faces(conn)

    table = Table(title="Faces")
    table.add_column("face_id", justify="right")
    table.add_column("person", justify="left")
    table.add_column("photo_id", justify="right")
    table.add_column("file_path", justify="left")

    for r in rows:
        table.add_row(
            str(r["id"]),
            r["person_name"] or "Unknown",
            str(r["photo_id"]),
            r["file_path"],
        )

    print(table)
    conn.close()


# ---------------------------------------------------------------------------
# FACES — TAG (assign/rename)
# ---------------------------------------------------------------------------
@app.command("faces-tag")
def faces_tag(
    face_id: int = typer.Argument(..., help="Face ID (see faces-list)"),
    name: str = typer.Argument(..., help="Person name"),
    db_path: str = typer.Option("photo.db", "--db"),
):
    """
    Assign a face to a person name (will create person if needed).
    """
    conn = db.connect(db_path)
    person_id = db.get_or_create_person(conn, name)
    db.set_face_person(conn, face_id, person_id)
    conn.close()
    print(f"[green]Face {face_id}[/green] tagged as [bold]{name}[/bold]")


# ---------------------------------------------------------------------------
# SEARCH (CLIP text → photo)
# ---------------------------------------------------------------------------
@app.command()
def search(
    query: str,
    db_path: str = typer.Option("photo.db", "--db"),
    top_k: int = typer.Option(20),
):
    """Natural language search using CLIP text embedding against image embeddings."""
    import faiss

    conn = db.connect(db_path)
    em = ClipEmbedder()
    q = em.text_embedding(query)

    rows = db.get_embeddings(conn)
    if not rows:
        print("[yellow]No embeddings found. Run 'analyze' first.[/yellow]")
        conn.close()
        return

    photo_ids = []
    vectors = []
    for pid, dim, vec_bytes in rows:
        v = np.frombuffer(vec_bytes, dtype=np.float32)
        if v.shape[0] != dim:
            continue
        photo_ids.append(pid)
        vectors.append(v)

    xb = np.stack(vectors).astype("float32")
    # Normalize
    xb = xb / (np.linalg.norm(xb, axis=1, keepdims=True) + 1e-9)

    dim = xb.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(xb)

    qn = q.reshape(1, -1).astype("float32")
    qn = qn / (np.linalg.norm(qn) + 1e-9)
    sims, idxs = index.search(qn, min(top_k, len(photo_ids)))

    table = Table(title=f"Search: {query}")
    table.add_column("score", justify="right")
    table.add_column("photo_id")
    table.add_column("path")

    cur = conn.cursor()
    for i, idx in enumerate(idxs[0]):
        score = sims[0][i]
        pid = photo_ids[idx]
        cur.execute("SELECT file_path FROM photos WHERE id=?", (pid,))
        row = cur.fetchone()
        if not row:
            continue
        table.add_row(f"{score:.3f}", str(pid), row["file_path"])

    print(table)
    conn.close()


# ---------------------------------------------------------------------------
# DUPLICATES (phash)
# ---------------------------------------------------------------------------
@app.command()
def duplicates(
    db_path: str = typer.Option("photo.db", "--db"),
    threshold: int = typer.Option(5, help="Hamming distance threshold for phash similarity"),
):
    """Find potential duplicate photos using perceptual hash comparison."""
    import imagehash

    conn = db.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT id, file_path, phash FROM photos WHERE phash != ''")
    rows = cur.fetchall()

    if len(rows) < 2:
        print("[yellow]Not enough photos with phash to check for duplicates[/yellow]")
        conn.close()
        return

    duplicates_found = []
    checked = set()

    for i, row1 in enumerate(rows):
        id1, path1, phash1 = row1["id"], row1["file_path"], row1["phash"]
        if id1 in checked:
            continue
        hash1 = imagehash.hex_to_hash(phash1)
        group = [(id1, path1)]

        for row2 in rows[i + 1 :]:
            id2, path2, phash2 = row2["id"], row2["file_path"], row2["phash"]
            if id2 in checked:
                continue
            hash2 = imagehash.hex_to_hash(phash2)
            distance = hash1 - hash2
            if distance <= threshold:
                group.append((id2, path2))
                checked.add(id2)

        if len(group) > 1:
            duplicates_found.append(group)
            checked.add(id1)

    conn.close()

    if not duplicates_found:
        print("[green]No duplicates found![/green]")
        return

    print(f"\n[cyan]Found {len(duplicates_found)} groups of potential duplicates:[/cyan]\n")
    for idx, group in enumerate(duplicates_found, 1):
        print(f"[yellow]Group {idx}:[/yellow]")
        for photo_id, path in group:
            print(f"  - ID {photo_id}: {path}")
        print()


# ---------------------------------------------------------------------------
# EXPORT (HTML gallery including faces)
# ---------------------------------------------------------------------------
@app.command()
def export(
    what: str = typer.Argument(..., help="`gallery` only for now"),
    output: str = typer.Option("./chitra", "--output", help="Output directory"),
    db_path: str = typer.Option("photo.db", "--db", help="SQLite DB path"),
):
    """
    Export artifacts (static HTML gallery).

    - Thumbnails for ALL photos (JPEG, works for RAW via `ensure_thumb`)
    - Optional faces section that uses the *photo thumbnail* for now
      (so browser never sees .ARW directly).
    """
    if what != "gallery":
        print("[red]Only 'gallery' export is supported currently[/red]")
        raise typer.Exit(code=1)

    from core.gallery import ensure_thumb, build_gallery

    conn = db.connect(db_path)
    cur = conn.cursor()

    # ------------------------------------------------------------------
    # 1) Build photo items + thumbnail map
    # ------------------------------------------------------------------
    cur.execute(
        """
        SELECT p.id, p.file_path, COALESCE(group_concat(t.tag, ', '), '')
        FROM photos p
        LEFT JOIN tags t ON p.id = t.photo_id
        GROUP BY p.id
        """
    )
    rows = cur.fetchall()

    items = []
    output_path = Path(output)
    thumb_dir = output_path / "thumbnails"
    thumb_dir.mkdir(parents=True, exist_ok=True)

    # photo_id -> relative thumbnail path (for re-use by faces)
    photo_thumb_by_id: dict[int, str] = {}

    for photo_id, file_path, tags in tqdm(rows, desc="Thumbnails"):
        thumb_filename = Path(file_path).name + ".jpg"
        thumb_abs_path = thumb_dir / thumb_filename

        # This uses load_image under the hood, so RAW files are ok.
        ensure_thumb(file_path, str(thumb_abs_path))

        thumb_rel_path = str(Path("thumbnails") / thumb_filename)

        items.append(
            {
                "photo_id": photo_id,
                "file": file_path,
                "thumb": thumb_rel_path,
                "tags": tags or "",
            }
        )
        photo_thumb_by_id[photo_id] = thumb_rel_path

    # ------------------------------------------------------------------
    # 2) Build faces list for Faces view
    #    -> uses photo thumbnail instead of RAW path
    # ------------------------------------------------------------------
    faces = []
    try:
        cur.execute(
            """
            SELECT f.id, f.photo_id, f.person_id, p.file_path
            FROM faces f
            JOIN photos p ON p.id = f.photo_id
            """
        )
        face_rows = cur.fetchall()

        for face_id, photo_id, person_id, file_path in face_rows:
            thumb_rel = photo_thumb_by_id.get(photo_id)

            # Basic label; can be overridden later when you add manual naming
            if person_id is None:
                person_label = "Unknown"
            else:
                person_label = f"Person {person_id}"

            faces.append(
                {
                    "face_id": face_id,
                    "photo_id": photo_id,
                    "photo_path": file_path,
                    "thumb": thumb_rel,        # << key fix: always a JPG thumbnail
                    "person_name": person_label,
                }
            )
    except Exception as e:
        # If faces table/schema is missing, just skip faces section
        print(f"[yellow]Skipping faces section in gallery:[/yellow] {e}")
        faces = []

    html = build_gallery(items, faces=faces, output_dir=output)
    conn.close()
    print(f"[green]Gallery built:[/green] {html}")

    """Export artifacts (static HTML gallery)."""
    if what != "gallery":
        print("[red]Only 'gallery' export is supported currently[/red]")
        raise typer.Exit(code=1)

    conn = db.connect(db_path)
    cur = conn.cursor()
    cur.execute(
        """
        SELECT p.file_path, group_concat(t.tag, ', ') as tags
        FROM photos p
        LEFT JOIN tags t ON p.id = t.photo_id
        GROUP BY p.id
        """
    )
    rows = cur.fetchall()

    items = []
    output_path = Path(output)
    thumb_dir = output_path / "thumbnails"
    thumb_dir.mkdir(parents=True, exist_ok=True)

    from core.gallery import build_gallery
    from core.gallery import Path as GPath  # type alias hack to avoid confusion
    from core.gallery import json as gjson  # not actually needed, keep minimal

    from core.gallery import build_gallery as _bg  # just to reassure lints

    from core.gallery import build_gallery as build_gallery_fn  # final alias
    # (above imports are no-ops logically, but avoid circular from refactoring)

    # For thumbnails
    from core.extractor import load_image

    # Actually generate thumbnails for each photo
    for row in tqdm(rows, desc="Thumbnails"):
        file_path = row["file_path"]
        tags = row["tags"] or ""
        fp = Path(file_path)
        # thumbnail name: stable
        thumb_filename = fp.name + ".jpg"
        thumb_abs_path = thumb_dir / thumb_filename

        if not thumb_abs_path.exists():
            try:
                img = load_image(fp)
                img.thumbnail((320, 320))
                thumb_abs_path.parent.mkdir(parents=True, exist_ok=True)
                img.save(thumb_abs_path, "JPEG", quality=100)
            except Exception as e:
                print(f"[red]Failed thumb[/red] {file_path}: {e}")
                continue

        thumb_rel_path = os.path.join("thumbnails", thumb_filename)
        items.append({"file": file_path, "thumb": thumb_rel_path, "tags": tags})

    # Faces info for gallery
    face_rows = db.get_face_thumbs(conn)
    faces = []
    for r in face_rows:
        faces.append(
            {
                "face_id": r["face_id"],
                "photo_id": r["photo_id"],
                "photo_path": r["file_path"],
                "thumb": r["thumb_path"],
                "person_id": r["person_id"],
                "person_name": r["person_name"],
            }
        )

    html = build_gallery(items, faces=faces, output_dir=output)
    conn.close()
    print(f"[green]Gallery built:[/green] {html}")


# ---------------------------------------------------------------------------
# TUI
# ---------------------------------------------------------------------------
@app.command()
def tui(db_path: str = typer.Option("photo.db", "--db")):
    """Launch the Textual TUI to browse photos and clusters."""
    try:
        from .tui import run_tui
    except Exception:
        from cli.tui import run_tui
    run_tui(db_path)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app()
