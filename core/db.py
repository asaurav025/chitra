from __future__ import annotations
import sqlite3
from pathlib import Path
from typing import Iterable, List, Tuple, Dict, Optional, Any


DB_DEFAULT_PATH = "photo.db"


# ----------------------------------------------------------------------
# CONNECTION + INIT
# ----------------------------------------------------------------------
def connect(db_path: str = DB_DEFAULT_PATH) -> sqlite3.Connection:
    """
    Create SQLite connection with optimizations for concurrent access.
    Uses WAL mode for better read concurrency.
    """
    conn = sqlite3.connect(db_path, timeout=30.0)  # 30 second timeout for busy connections
    conn.row_factory = sqlite3.Row
    
    # Enable WAL mode for better concurrency (allows concurrent reads)
    try:
        conn.execute("PRAGMA journal_mode=WAL")
    except:
        pass  # Ignore if WAL mode not supported
    
    # Optimize for performance
    try:
        conn.execute("PRAGMA synchronous=NORMAL")  # Faster than FULL, still safe
        conn.execute("PRAGMA cache_size=-64000")  # 64MB cache
        conn.execute("PRAGMA temp_store=MEMORY")  # Use memory for temp tables
    except:
        pass
    
    return conn


def init_db(db_path: str = DB_DEFAULT_PATH):
    """Create tables if they don't exist."""
    conn = connect(db_path)
    cur = conn.cursor()

    # Photos
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS photos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_path TEXT UNIQUE,
            size INTEGER,
            created_at TEXT,
            checksum TEXT,
            phash TEXT,
            exif_datetime TEXT,
            latitude REAL,
            longitude REAL,
            thumb_path TEXT
        )
        """
    )

    # Embeddings (CLIP image embeddings)
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS embeddings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            photo_id INTEGER NOT NULL,
            dim INTEGER NOT NULL,
            vector BLOB NOT NULL,
            FOREIGN KEY(photo_id) REFERENCES photos(id) ON DELETE CASCADE
        )
        """
    )

    # Tags (auto_tags)
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS tags (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            photo_id INTEGER NOT NULL,
            tag TEXT NOT NULL,
            score REAL NOT NULL,
            FOREIGN KEY(photo_id) REFERENCES photos(id) ON DELETE CASCADE
        )
        """
    )

    # Clusters (photo similarity)
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS clusters (
            photo_id INTEGER NOT NULL,
            cluster_id INTEGER NOT NULL,
            score REAL NOT NULL,
            PRIMARY KEY(photo_id),
            FOREIGN KEY(photo_id) REFERENCES photos(id) ON DELETE CASCADE
        )
        """
    )

    # Persons (named individuals)
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS persons (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL
        )
        """
    )

    # Faces (per-photo face detections)
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS faces (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            photo_id INTEGER NOT NULL,
            face_index INTEGER NOT NULL,
            bbox_x REAL,
            bbox_y REAL,
            bbox_w REAL,
            bbox_h REAL,
            embedding BLOB NOT NULL,
            person_id INTEGER,
            UNIQUE(photo_id, face_index),
            FOREIGN KEY(photo_id) REFERENCES photos(id) ON DELETE CASCADE,
            FOREIGN KEY(person_id) REFERENCES persons(id) ON DELETE SET NULL
        )
        """
    )

    # Thumbnails for faces (optional separate table)
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS face_thumbs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            face_id INTEGER NOT NULL UNIQUE,
            thumb_path TEXT NOT NULL,
            FOREIGN KEY(face_id) REFERENCES faces(id) ON DELETE CASCADE
        )
        """
    )

    # Indexes
    cur.execute("CREATE INDEX IF NOT EXISTS idx_photos_checksum ON photos(checksum)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_photos_path ON photos(file_path)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_embeddings_photo ON embeddings(photo_id)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_tags_photo ON tags(photo_id)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_faces_photo ON faces(photo_id)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_faces_person ON faces(person_id)")

    # Migration: Add thumb_path column if it doesn't exist
    try:
        cur.execute("ALTER TABLE photos ADD COLUMN thumb_path TEXT")
    except sqlite3.OperationalError:
        # Column already exists, ignore
        pass

    conn.commit()
    conn.close()


# ----------------------------------------------------------------------
# PHOTOS
# ----------------------------------------------------------------------
def upsert_photo(conn: sqlite3.Connection, **meta: Any):
    """
    Insert or update a photo row.
    Expected keys: file_path, size, created_at, checksum, phash, exif_datetime, latitude, longitude, thumb_path
    """
    conn.execute(
        """
        INSERT INTO photos (file_path, size, created_at, checksum, phash, exif_datetime, latitude, longitude, thumb_path)
        VALUES (:file_path, :size, :created_at, :checksum, :phash, :exif_datetime, :latitude, :longitude, :thumb_path)
        ON CONFLICT(file_path) DO UPDATE SET
            size=excluded.size,
            created_at=excluded.created_at,
            checksum=excluded.checksum,
            phash=excluded.phash,
            exif_datetime=excluded.exif_datetime,
            latitude=excluded.latitude,
            longitude=excluded.longitude,
            thumb_path=COALESCE(excluded.thumb_path, photos.thumb_path)
        """,
        meta,
    )
    conn.commit()


def iter_photos(conn: sqlite3.Connection) -> Iterable[Tuple[int, str]]:
    cur = conn.cursor()
    cur.execute("SELECT id, file_path FROM photos ORDER BY id ASC")
    for row in cur.fetchall():
        yield row["id"], row["file_path"]


# ----------------------------------------------------------------------
# EMBEDDINGS
# ----------------------------------------------------------------------
def put_embedding(conn: sqlite3.Connection, photo_id: int, vec_bytes: bytes, dim: int):
    conn.execute(
        """
        INSERT INTO embeddings (photo_id, dim, vector)
        VALUES (?, ?, ?)
        """,
        (photo_id, dim, vec_bytes),
    )
    conn.commit()


def get_embeddings(conn: sqlite3.Connection) -> List[Tuple[int, int, bytes]]:
    cur = conn.cursor()
    cur.execute("SELECT photo_id, dim, vector FROM embeddings")
    return [(row["photo_id"], row["dim"], row["vector"]) for row in cur.fetchall()]


# ----------------------------------------------------------------------
# TAGS
# ----------------------------------------------------------------------
def add_tag(conn: sqlite3.Connection, photo_id: int, tag: str, score: float):
    conn.execute(
        "INSERT INTO tags (photo_id, tag, score) VALUES (?, ?, ?)",
        (photo_id, tag, score),
    )
    conn.commit()


# ----------------------------------------------------------------------
# CLUSTERS
# ----------------------------------------------------------------------
def assign_cluster(conn: sqlite3.Connection, photo_id: int, cluster_id: int, score: float):
    conn.execute(
        """
        INSERT INTO clusters (photo_id, cluster_id, score)
        VALUES (?, ?, ?)
        ON CONFLICT(photo_id) DO UPDATE SET
          cluster_id=excluded.cluster_id,
          score=excluded.score
        """,
        (photo_id, cluster_id, score),
    )
    conn.commit()


# ----------------------------------------------------------------------
# FACES + PERSONS
# ----------------------------------------------------------------------
def add_face(
    conn: sqlite3.Connection,
    photo_id: int,
    face_index: int,
    embedding_bytes: bytes,
    bbox_x: float | None = None,
    bbox_y: float | None = None,
    bbox_w: float | None = None,
    bbox_h: float | None = None,
    person_id: int | None = None,
):
    conn.execute(
        """
        INSERT INTO faces (
            photo_id, face_index,
            bbox_x, bbox_y, bbox_w, bbox_h,
            embedding, person_id
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(photo_id, face_index) DO UPDATE SET
            bbox_x=excluded.bbox_x,
            bbox_y=excluded.bbox_y,
            bbox_w=excluded.bbox_w,
            bbox_h=excluded.bbox_h,
            embedding=excluded.embedding,
            person_id=excluded.person_id
        """,
        (photo_id, face_index, bbox_x, bbox_y, bbox_w, bbox_h, embedding_bytes, person_id),
    )
    conn.commit()


def iter_faces(conn: sqlite3.Connection) -> List[sqlite3.Row]:
    cur = conn.cursor()
    cur.execute(
        """
        SELECT f.id, f.photo_id, f.face_index,
               f.bbox_x, f.bbox_y, f.bbox_w, f.bbox_h,
               f.embedding, f.person_id,
               p.name as person_name,
               ph.file_path
        FROM faces f
        JOIN photos ph ON f.photo_id = ph.id
        LEFT JOIN persons p ON f.person_id = p.id
        ORDER BY f.id ASC
        """
    )
    return cur.fetchall()


def get_faces_embeddings(conn: sqlite3.Connection):
    cur = conn.cursor()
    cur.execute(
        """
        SELECT id, embedding FROM faces
        """
    )
    return cur.fetchall()


def set_face_person(conn: sqlite3.Connection, face_id: int, person_id: int):
    conn.execute("UPDATE faces SET person_id=? WHERE id=?", (person_id, face_id))
    conn.commit()


def get_or_create_person(conn: sqlite3.Connection, name: str) -> int:
    cur = conn.cursor()
    cur.execute("SELECT id FROM persons WHERE name=?", (name,))
    row = cur.fetchone()
    if row:
        return row["id"]

    cur.execute("INSERT INTO persons (name) VALUES (?)", (name,))
    conn.commit()
    return cur.lastrowid


def create_person(conn: sqlite3.Connection, name: str) -> int:
    """Create a new person. Raises error if name already exists."""
    cur = conn.cursor()
    cur.execute("INSERT INTO persons (name) VALUES (?)", (name,))
    conn.commit()
    return cur.lastrowid


def rename_person(conn: sqlite3.Connection, person_id: int, new_name: str):
    conn.execute("UPDATE persons SET name=? WHERE id=?", (new_name, person_id))
    conn.commit()


def list_persons(conn: sqlite3.Connection) -> List[sqlite3.Row]:
    cur = conn.cursor()
    cur.execute(
        """
        SELECT p.id, p.name, COUNT(f.id) as face_count
        FROM persons p
        LEFT JOIN faces f ON f.person_id = p.id
        GROUP BY p.id, p.name
        ORDER BY p.id ASC
        """
    )
    return cur.fetchall()


def get_faces_for_person(conn: sqlite3.Connection, person_name: str):
    cur = conn.cursor()
    cur.execute(
        """
        SELECT f.id, f.photo_id, ph.file_path,
               f.bbox_x, f.bbox_y, f.bbox_w, f.bbox_h
        FROM faces f
        JOIN persons p ON f.person_id = p.id
        JOIN photos ph ON ph.id = f.photo_id
        WHERE p.name = ?
        ORDER BY f.id ASC
        """,
        (person_name,),
    )
    return cur.fetchall()


def set_face_thumb(conn: sqlite3.Connection, face_id: int, thumb_path: str):
    conn.execute(
        """
        INSERT INTO face_thumbs (face_id, thumb_path)
        VALUES (?, ?)
        ON CONFLICT(face_id) DO UPDATE SET
          thumb_path=excluded.thumb_path
        """,
        (face_id, thumb_path),
    )
    conn.commit()


def get_face_thumbs(conn: sqlite3.Connection):
    cur = conn.cursor()
    cur.execute(
        """
        SELECT f.id as face_id,
               f.photo_id,
               ph.file_path,
               ft.thumb_path,
               f.person_id,
               p.name as person_name
        FROM faces f
        JOIN photos ph ON ph.id = f.photo_id
        LEFT JOIN face_thumbs ft ON ft.face_id = f.id
        LEFT JOIN persons p ON p.id = f.person_id
        ORDER BY f.id ASC
        """
    )
    return cur.fetchall()

def get_face_thumbs_for_person(conn: sqlite3.Connection, person_id: int):
    """Get face thumbnails for a specific person by person_id."""
    cur = conn.cursor()
    cur.execute(
        """
        SELECT f.id as face_id,
               f.photo_id,
               ph.file_path,
               ft.thumb_path,
               f.person_id,
               p.name as person_name
        FROM faces f
        JOIN photos ph ON ph.id = f.photo_id
        LEFT JOIN face_thumbs ft ON ft.face_id = f.id
        LEFT JOIN persons p ON p.id = f.person_id
        WHERE f.person_id = ?
        ORDER BY f.id ASC
        LIMIT 10
        """,
        (person_id,),
    )
    return cur.fetchall()


def merge_persons(conn: sqlite3.Connection, source_person_id: int, target_person_id: int):
    """Merge source person into target person. All faces assigned to source will be reassigned to target."""
    if source_person_id == target_person_id:
        raise ValueError("Cannot merge a person with itself")
    
    # Update all faces from source person to target person
    conn.execute(
        "UPDATE faces SET person_id=? WHERE person_id=?",
        (target_person_id, source_person_id)
    )
    
    # Delete the source person
    conn.execute("DELETE FROM persons WHERE id=?", (source_person_id,))
    
    conn.commit()
