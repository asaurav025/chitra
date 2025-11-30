from __future__ import annotations
import os, sys, json
from pathlib import Path
from typing import Optional, Dict, List
import typer
from rich import print
from rich.table import Table
from tqdm import tqdm
import numpy as np

from core import db
from core.extractor import iter_images, collect_metadata
from core.embedder import ClipEmbedder
from core.tagger import auto_tags
from core.cluster import threshold_clusters
from core.gallery import ensure_thumb, build_gallery
from core.face import face_encodings

app = typer.Typer(add_completion=False, help="Chitra — Photo Intelligence CLI")


# ------------------------------------------------------------------------------------
# INIT
# ------------------------------------------------------------------------------------
@app.command()
def init(
    db_path: str = typer.Option("photo.db", "--db", help="SQLite database path")
):
    """Create the database schema."""
    db.init_db(db_path)
    print(f"[green]Initialized[/green] SQLite at {db_path}")


# ------------------------------------------------------------------------------------
# SCAN
# ------------------------------------------------------------------------------------
@app.command()
def scan(
    path: str = typer.Argument(..., help="Root folder to scan"),
    db_path: str = typer.Option("photo.db", "--db"),
    incremental: bool = typer.Option(
        True, "--incremental", help="Skip files already in DB if unchanged"
    ),
):
    """Index files metadata + EXIF (no ML)."""
    conn = db.connect(db_path)

    existing = {}
    if incremental:
        cur = conn.cursor()
        cur.execute("SELECT file_path, checksum FROM photos")
        existing = {row[0]: row[1] for row in cur.fetchall()}

    total = 0
    skipped = 0

    for p in tqdm(list(iter_images(path)), desc="Scanning"):
        fpath = str(p)

        if incremental and fpath in existing:
            st = p.stat()
            cur = conn.cursor()
            cur.execute("SELECT size, created_at FROM photos WHERE file_path=?", (fpath,))
            row = cur.fetchone()

            if row:
                old_size, _ = row
                if old_size == st.st_size:
                    skipped += 1
                    continue

        meta = collect_metadata(p)
        db.upsert_photo(conn, **meta)
        total += 1

    conn.close()
    print(f"[cyan]Indexed {total} photos[/cyan]", end="")
    if skipped:
        print(f" [dim](skipped {skipped})[/dim]")
    else:
        print()


# ------------------------------------------------------------------------------------
# ANALYZE (CLIP + AUTOTAGS)
# ------------------------------------------------------------------------------------
@app.command()
def analyze(
    db_path: str = typer.Option("photo.db", "--db"),
    limit: Optional[int] = typer.Option(None, "--limit", help="Process only N photos"),
    tag_k: int = typer.Option(5, "--tag-k", help="Top-K tags to store"),
    incremental: bool = typer.Option(
        True, "--incremental/--no-incremental", help="Skip photos that already have embeddings"
    ),
):
    """Compute CLIP embeddings + auto-tags and store in DB."""
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
        print("[green]All photos already analyzed[/green]")
        conn.close()
        return

    for pid, file_path in tqdm(rows, desc="Analyzing"):
        try:
            vec = em.image_embedding(file_path)
            db.put_embedding(conn, pid, vec.tobytes(), vec.shape[0])

            conn.execute("DELETE FROM tags WHERE photo_id=?", (pid,))
            for tag, score in auto_tags(em, file_path, k=tag_k):
                db.add_tag(conn, pid, tag, float(score))

        except Exception as e:
            print(f"[red]Failed[/red] {file_path}: {e}")

    conn.close()
    print("[green]Analyze complete[/green]")


# ------------------------------------------------------------------------------------
# CLUSTER
# ------------------------------------------------------------------------------------
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

    vectors: Dict[int, np.ndarray] = {}
    for pid, dim, vec_bytes in items:
        v = np.frombuffer(vec_bytes, dtype=np.float32).reshape(dim)
        v = v / (np.linalg.norm(v) + 1e-9)
        vectors[pid] = v

    clusters = threshold_clusters(vectors, threshold=threshold)

    for cid, members in clusters.items():
        for pid in members:
            db.assign_cluster(conn, pid, cid, 1.0)

    conn.close()
    print(
        f"[green]Clustered[/green] {sum(len(m) for m in clusters.values())} photos into {len(clusters)} clusters"
    )


# ------------------------------------------------------------------------------------
# FACES (InsightFace)
# ------------------------------------------------------------------------------------
@app.command()
def faces(
    db_path: str = typer.Option("photo.db", "--db"),
    limit: Optional[int] = typer.Option(None, "--limit", help="Only process first N"),
):
    """Detect and store InsightFace face embeddings."""
    conn = db.connect(db_path)
    rows = list(db.iter_photos(conn))

    if limit:
        rows = rows[:limit]

    face_count = 0

    for pid, file_path in tqdm(rows, desc="Detecting faces"):
        try:
            detected = face_encodings(file_path)

            for face_id, f in enumerate(detected):
                embedding = np.array(f["embedding"], dtype=np.float32)
                db.add_face(conn, pid, face_id, embedding.tobytes())
                face_count += 1

        except Exception as e:
            print(f"[red]Failed[/red] {file_path}: {e}")

    conn.close()
    print(f"[green]Detected {face_count} faces[/green]")


# ------------------------------------------------------------------------------------
# SEARCH
# ------------------------------------------------------------------------------------
@app.command()
def search(
    query: str,
    db_path: str = typer.Option("photo.db", "--db"),
    top_k: int = typer.Option(20, "--top-k", help="Return top-K results"),
):
    """Natural language search using CLIP text embedding."""
    import faiss

    conn = db.connect(db_path)
    em = ClipEmbedder()

    q = em.text_embedding(query)

    rows = db.get_embeddings(conn)
    if not rows:
        print("[yellow]No embeddings found. Run analyze first.[/yellow]")
        conn.close()
        return

    photo_ids = []
    vectors = []

    for pid, dim, vec_bytes in rows:
        v = np.frombuffer(vec_bytes, dtype=np.float32)
        vectors.append(v)
        photo_ids.append(pid)

    xb = np.stack(vectors).astype("float32")
    faiss.normalize_L2(xb)

    dim = xb.shape[1]
    index = faiss.IndexFlatIP(dim)  # cosine similarity
    index.add(xb)

    qn = q.reshape(1, -1).astype("float32")
    faiss.normalize_L2(qn)

    sims, idxs = index.search(qn, min(top_k, len(photo_ids)))

    table = Table(title=f"Search results: {query}")
    table.add_column("score", justify="right")
    table.add_column("photo_id")
    table.add_column("path")

    cur = conn.cursor()
    for i, idx in enumerate(idxs[0]):
        pid = photo_ids[idx]
        score = sims[0][i]

        cur.execute("SELECT file_path FROM photos WHERE id=?", (pid,))
        fpath = cur.fetchone()[0]

        table.add_row(f"{score:.3f}", str(pid), fpath)

    print(table)
    conn.close()


# ------------------------------------------------------------------------------------
# DUPLICATES
# ------------------------------------------------------------------------------------
@app.command()
def duplicates(
    db_path: str = typer.Option("photo.db", "--db"),
    threshold: int = typer.Option(
        5, "--threshold", help="Max Hamming distance for phash duplicates"
    ),
):
    """Find visually similar duplicate images using perceptual hash."""
    conn = db.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT id, file_path, phash FROM photos WHERE phash != ''")
    rows = cur.fetchall()

    if len(rows) < 2:
        print("[yellow]Not enough photos with phash[/yellow]")
        conn.close()
        return

    import imagehash

    duplicates_found = []
    checked = set()

    for i, (id1, path1, ph1) in enumerate(rows):
        if id1 in checked:
            continue

        h1 = imagehash.hex_to_hash(ph1)
        group = [(id1, path1)]

        for id2, path2, ph2 in rows[i + 1 :]:
            if id2 in checked:
                continue

            h2 = imagehash.hex_to_hash(ph2)
            dist = h1 - h2

            if dist <= threshold:
                checked.add(id2)
                group.append((id2, path2))

        if len(group) > 1:
            duplicates_found.append(group)
            checked.add(id1)

    conn.close()

    if not duplicates_found:
        print("[green]No duplicates found[/green]")
        return

    print(f"\n[cyan]Found {len(duplicates_found)} duplicate groups:[/cyan]\n")
    for i, group in enumerate(duplicates_found, 1):
        print(f"[yellow]Group {i}[/yellow]")
        for pid, path in group:
            print(f"  - ID {pid}: {path}")
        print()


# ------------------------------------------------------------------------------------
# EXPORT (GALLERY)
# ------------------------------------------------------------------------------------
@app.command()
def export(
    what: str = typer.Argument(..., help="`gallery` only"),
    output: str = typer.Option(
        "./chitra", "--output", help="Output directory for gallery"
    ),
):
    """Export static HTML gallery."""
    if what != "gallery":
        print("[red]Only 'gallery' export is supported[/red]")
        raise typer.Exit(code=1)

    conn = db.connect()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT p.file_path, group_concat(t.tag, ', ') 
        FROM photos p
        LEFT JOIN tags t ON p.id = t.photo_id
        GROUP BY p.id
        """
    )

    rows = cur.fetchall()

    output_path = Path(output)
    thumb_dir = output_path / "thumbnails"
    thumb_dir.mkdir(parents=True, exist_ok=True)

    items = []

    for file_path, tags in tqdm(rows, desc="Thumbnails"):
        fname = Path(file_path).name + ".jpg"
        thumb_abs = thumb_dir / fname

        ensure_thumb(file_path, str(thumb_abs))

        items.append(
            {
                "file": file_path,
                "thumb": os.path.join("thumbnails", fname),
                "tags": tags or "",
            }
        )

    html = build_gallery(items, output_dir=output)
    conn.close()
    print(f"[green]Gallery built:[/green] {html}")


# ------------------------------------------------------------------------------------
# TUI
# ------------------------------------------------------------------------------------
@app.command()
def tui(
    db_path: str = typer.Option("photo.db", "--db")
):
    """Launch the Textual TUI browser."""
    try:
        from .tui import run_tui
    except Exception:
        from cli.tui import run_tui
    run_tui(db_path)


# ------------------------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------------------------
if __name__ == "__main__":
    app()
