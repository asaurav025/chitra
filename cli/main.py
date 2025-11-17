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
from core.face import face_encodings, HAS_FACE

app = typer.Typer(add_completion=False, help="Chitra — Photo Intelligence CLI")

@app.command()
def init(db_path: str = typer.Option("photo.db", "--db", help="SQLite database path")):
    """Create the database schema."""
    db.init_db(db_path)
    print(f"[green]Initialized[/green] SQLite at {db_path}")

@app.command()
def scan(
    path: str = typer.Argument(..., help="Root folder to scan"),
    db_path: str = typer.Option("photo.db", "--db"),
    incremental: bool = typer.Option(True, help="Skip files already in DB with same checksum"),
):
    """Index files metadata and EXIF (no ML)."""
    conn = db.connect(db_path)
    
    # Get existing files if incremental mode
    existing = {}
    if incremental:
        cur = conn.cursor()
        cur.execute("SELECT file_path, checksum FROM photos")
        existing = {row[0]: row[1] for row in cur.fetchall()}
    
    total = 0
    skipped = 0
    for p in tqdm(list(iter_images(path)), desc="Scanning"):
        file_path_str = str(p)
        
        # Skip if file exists with same checksum
        if incremental and file_path_str in existing:
            # Quick check: if file size/mtime hasn't changed, skip checksum computation
            import os, time
            st = p.stat()
            cur = conn.cursor()
            cur.execute("SELECT size, created_at FROM photos WHERE file_path=?", (file_path_str,))
            row = cur.fetchone()
            if row:
                old_size, old_time = row
                if old_size == st.st_size:
                    # Size matches, assume unchanged
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

@app.command()
def analyze(
    db_path: str = typer.Option("photo.db", "--db"),
    limit: Optional[int] = typer.Option(None, help="Process only N photos (for testing)"),
    tag_k: int = typer.Option(5, help="Top-K tags to store"),
    incremental: bool = typer.Option(True, help="Skip photos that already have embeddings"),
):
    """Compute CLIP embeddings + auto-tags and store into SQLite."""
    conn = db.connect(db_path)
    em = ClipEmbedder()
    
    # Get photos without embeddings if incremental
    if incremental:
        cur = conn.cursor()
        cur.execute("""
            SELECT p.id, p.file_path 
            FROM photos p 
            LEFT JOIN embeddings e ON p.id = e.photo_id 
            WHERE e.photo_id IS NULL
        """)
        rows = cur.fetchall()
    else:
        rows = list(db.iter_photos(conn))
    
    if limit:
        rows = rows[:limit]
    
    if not rows:
        print("[green]All photos already analyzed![/green]")
        conn.close()
        return
    
    for pid, file_path in tqdm(rows, desc="Analyzing"):
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

@app.command()
def cluster(
    db_path: str = typer.Option("photo.db", "--db"),
    threshold: float = typer.Option(0.78, min=0.0, max=1.0),
):
    """Group similar photos using FAISS + threshold union-find."""
    conn = db.connect(db_path)
    items = db.get_embeddings(conn)
    vectors: Dict[int, np.ndarray] = {}
    for photo_id, dim, vec_bytes in items:
        v = np.frombuffer(vec_bytes, dtype=np.float32)
        v = v.reshape(dim)
        v = v / (np.linalg.norm(v) + 1e-9)
        vectors[photo_id] = v

    clusters = threshold_clusters(vectors, threshold=threshold)

    # Assign cluster IDs
    for cid, members in clusters.items():
        for pid in members:
            # naive similarity marker (1.0 for same cluster)
            db.assign_cluster(conn, pid, cid, 1.0)
    conn.close()
    print(f"[green]Clustered[/green] {sum(len(m) for m in clusters.values())} photos into {len(clusters)} clusters")

@app.command()
def faces(
    db_path: str = typer.Option("photo.db", "--db"),
    limit: Optional[int] = typer.Option(None, help="Process only N photos (for testing)"),
):
    """Detect and cluster faces in photos (requires face_recognition + dlib)."""
    if not HAS_FACE:
        print("[yellow]Skipping faces:[/yellow] face_recognition and dlib not installed")
        print("Install with: pip install face_recognition dlib")
        return
    
    conn = db.connect(db_path)
    rows = list(db.iter_photos(conn))
    if limit:
        rows = rows[:limit]
    
    face_count = 0
    for pid, file_path in tqdm(rows, desc="Detecting faces"):
        try:
            encodings = face_encodings(file_path)
            for face_id, enc in enumerate(encodings):
                db.add_face(conn, pid, face_id, enc.tobytes())
                face_count += 1
        except Exception as e:
            print(f"[red]Failed[/red] {file_path}: {e}")
    
    conn.close()
    print(f"[green]Detected {face_count} faces[/green]")

@app.command()
def search(
    query: str,
    db_path: str = typer.Option("photo.db", "--db"),
    top_k: int = typer.Option(20)
):
    """Natural language search using CLIP text embedding against image embeddings."""
    import faiss
    
    conn = db.connect(db_path)
    em = ClipEmbedder()
    q = em.text_embedding(query)
    
    # Load all embeddings and build FAISS index
    rows = db.get_embeddings(conn)
    if not rows:
        print("[yellow]No embeddings found. Run 'analyze' first.[/yellow]")
        conn.close()
        return
    
    photo_ids = []
    vectors = []
    for pid, dim, vec_bytes in rows:
        v = np.frombuffer(vec_bytes, dtype=np.float32)
        photo_ids.append(pid)
        vectors.append(v)
    
    # Build FAISS index for fast similarity search
    xb = np.stack(vectors).astype('float32')
    faiss.normalize_L2(xb)  # Normalize for cosine similarity
    
    dim = xb.shape[1]
    index = faiss.IndexFlatIP(dim)  # Inner product = cosine for normalized vectors
    index.add(xb)
    
    # Search
    q_normalized = q.reshape(1, -1).astype('float32')
    faiss.normalize_L2(q_normalized)
    similarities, indices = index.search(q_normalized, min(top_k, len(photo_ids)))
    
    # Display results
    table = Table(title=f"Search: {query}")
    table.add_column("score", justify="right")
    table.add_column("photo_id")
    table.add_column("path")
    
    cur = conn.cursor()
    for i, idx in enumerate(indices[0]):
        score = similarities[0][i]
        pid = photo_ids[idx]
        cur.execute("SELECT file_path FROM photos WHERE id=?", (pid,))
        path = cur.fetchone()[0]
        table.add_row(f"{score:.3f}", str(pid), path)
    
    print(table)
    conn.close()

@app.command()
def duplicates(
    db_path: str = typer.Option("photo.db", "--db"),
    threshold: int = typer.Option(5, help="Hamming distance threshold for phash similarity")
):
    """Find potential duplicate photos using perceptual hash comparison."""
    conn = db.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT id, file_path, phash FROM photos WHERE phash != ''")
    rows = cur.fetchall()
    
    if len(rows) < 2:
        print("[yellow]Not enough photos with phash to check for duplicates[/yellow]")
        conn.close()
        return
    
    import imagehash
    
    # Group photos by similar phash
    duplicates_found = []
    checked = set()
    
    for i, (id1, path1, phash1) in enumerate(rows):
        if id1 in checked:
            continue
        
        hash1 = imagehash.hex_to_hash(phash1)
        group = [(id1, path1)]
        
        for id2, path2, phash2 in rows[i+1:]:
            if id2 in checked:
                continue
            hash2 = imagehash.hex_to_hash(phash2)
            distance = hash1 - hash2  # Hamming distance
            
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
    
    # Display results
    print(f"\n[cyan]Found {len(duplicates_found)} groups of potential duplicates:[/cyan]\n")
    for idx, group in enumerate(duplicates_found, 1):
        print(f"[yellow]Group {idx}:[/yellow]")
        for photo_id, path in group:
            print(f"  - ID {photo_id}: {path}")
        print()

@app.command()
def export(
    what: str = typer.Argument(..., help="`gallery` only for now"),
    output: str = typer.Option("./chitra", "--output", help="Output directory")
):
    """Export artifacts (static HTML gallery)."""
    if what != "gallery":
        print("[red]Only 'gallery' export is supported currently[/red]")
        raise typer.Exit(code=1)

    conn = db.connect()
    cur = conn.cursor()
    cur.execute("""
      SELECT p.file_path, group_concat(t.tag, ', ') as tags
      FROM photos p
      LEFT JOIN tags t ON p.id = t.photo_id
      GROUP BY p.id
    """)
    rows = cur.fetchall()
    items = []
    # Create output directory and thumbnails subdirectory
    output_path = Path(output)
    thumb_dir = output_path / "thumbnails"
    thumb_dir.mkdir(parents=True, exist_ok=True)
    
    for file_path, tags in tqdm(rows, desc="Thumbnails"):
        # Save thumbnail to output/thumbnails/
        thumb_filename = Path(file_path).name + ".jpg"
        thumb_abs_path = thumb_dir / thumb_filename
        ensure_thumb(file_path, str(thumb_abs_path))
        # Use relative path in HTML
        thumb_rel_path = os.path.join("thumbnails", thumb_filename)
        items.append({ 'file': file_path, 'thumb': thumb_rel_path, 'tags': tags or '' })

    html = build_gallery(items, output_dir=output)
    conn.close()
    print(f"[green]Gallery built:[/green] {html}")

@app.command()
def tui(db_path: str = typer.Option("photo.db", "--db")):
    """Launch the Textual TUI to browse photos and clusters."""
    try:
        from .tui import run_tui
    except Exception:
        from cli.tui import run_tui
    run_tui(db_path)

if __name__ == "__main__":
    app()
