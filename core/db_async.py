from __future__ import annotations

import sqlite3
from contextlib import asynccontextmanager
from pathlib import Path
from typing import (
    Any,
    AsyncIterator,
    List,
    Tuple,
)

import aiosqlite


DB_DEFAULT_PATH = "photo.db"


# ----------------------------------------------------------------------
# CONNECTION + INIT (ASYNC)
# ----------------------------------------------------------------------
@asynccontextmanager
async def connect_async(db_path: str = DB_DEFAULT_PATH) -> AsyncIterator[aiosqlite.Connection]:
    """
    Async connection context manager with SQLite optimizations.
    Uses WAL mode and tuning PRAGMAs for better concurrent access.
    """
    conn = await aiosqlite.connect(db_path, timeout=30.0)
    conn.row_factory = sqlite3.Row

    # PRAGMA setup
    try:
        await conn.execute("PRAGMA journal_mode=WAL")
    except Exception:
        pass

    try:
        await conn.execute("PRAGMA synchronous=NORMAL")
        await conn.execute("PRAGMA cache_size=-64000")  # 64MB cache
        await conn.execute("PRAGMA temp_store=MEMORY")
    except Exception:
        pass

    try:
        yield conn
    finally:
        await conn.close()


async def init_db_async(db_path: str = DB_DEFAULT_PATH) -> None:
    """Create tables if they don't exist (async version)."""
    async with connect_async(db_path) as conn:
        # Photos
        await conn.execute(
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
        await conn.execute(
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
        await conn.execute(
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
        await conn.execute(
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
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS persons (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL
            )
            """
        )

        # Faces (per-photo face detections)
        await conn.execute(
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

        # Thumbnails for faces
        await conn.execute(
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
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_photos_checksum ON photos(checksum)")
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_photos_path ON photos(file_path)")
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_embeddings_photo ON embeddings(photo_id)")
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_tags_photo ON tags(photo_id)")
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_faces_photo ON faces(photo_id)")
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_faces_person ON faces(person_id)")

        # Migration: Add thumb_path column if it doesn't exist
        try:
            await conn.execute("ALTER TABLE photos ADD COLUMN thumb_path TEXT")
        except aiosqlite.OperationalError:
            # Column already exists, ignore
            pass

        await conn.commit()


# ----------------------------------------------------------------------
# PHOTOS (ASYNC)
# ----------------------------------------------------------------------
async def upsert_photo_async(conn: aiosqlite.Connection, **meta: Any) -> None:
    """
    Insert or update a photo row (async).
    Expected keys: file_path, size, created_at, checksum, phash,
                   exif_datetime, latitude, longitude, thumb_path (optional)
    """
    # Ensure thumb_path is in meta (default to None if not provided)
    if 'thumb_path' not in meta:
        meta['thumb_path'] = None
    
    await conn.execute(
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
    await conn.commit()


async def iter_photos_async(conn: aiosqlite.Connection) -> List[Tuple[int, str]]:
    """
    Fetch all photos (id, file_path) as a list.
    (If you want streaming, we can switch this to an async generator.)
    """
    async with conn.execute("SELECT id, file_path FROM photos ORDER BY id ASC") as cur:
        rows = await cur.fetchall()
    return [(row["id"], row["file_path"]) for row in rows]


# ----------------------------------------------------------------------
# EMBEDDINGS (ASYNC)
# ----------------------------------------------------------------------
async def put_embedding_async(
    conn: aiosqlite.Connection,
    photo_id: int,
    vec_bytes: bytes,
    dim: int,
) -> None:
    await conn.execute(
        """
        INSERT INTO embeddings (photo_id, dim, vector)
        VALUES (?, ?, ?)
        """,
        (photo_id, dim, vec_bytes),
    )
    await conn.commit()


async def get_embeddings_async(conn: aiosqlite.Connection) -> List[Tuple[int, int, bytes]]:
    async with conn.execute("SELECT photo_id, dim, vector FROM embeddings") as cur:
        rows = await cur.fetchall()
    return [(row["photo_id"], row["dim"], row["vector"]) for row in rows]


# ----------------------------------------------------------------------
# TAGS (ASYNC)
# ----------------------------------------------------------------------
async def add_tag_async(conn: aiosqlite.Connection, photo_id: int, tag: str, score: float) -> None:
    await conn.execute(
        "INSERT INTO tags (photo_id, tag, score) VALUES (?, ?, ?)",
        (photo_id, tag, score),
    )
    await conn.commit()


# ----------------------------------------------------------------------
# CLUSTERS (ASYNC)
# ----------------------------------------------------------------------
async def assign_cluster_async(
    conn: aiosqlite.Connection,
    photo_id: int,
    cluster_id: int,
    score: float,
) -> None:
    await conn.execute(
        """
        INSERT INTO clusters (photo_id, cluster_id, score)
        VALUES (?, ?, ?)
        ON CONFLICT(photo_id) DO UPDATE SET
          cluster_id=excluded.cluster_id,
          score=excluded.score
        """,
        (photo_id, cluster_id, score),
    )
    await conn.commit()


# ----------------------------------------------------------------------
# FACES + PERSONS (ASYNC)
# ----------------------------------------------------------------------
async def add_face_async(
    conn: aiosqlite.Connection,
    photo_id: int,
    face_index: int,
    embedding_bytes: bytes,
    bbox_x: float | None = None,
    bbox_y: float | None = None,
    bbox_w: float | None = None,
    bbox_h: float | None = None,
    person_id: int | None = None,
) -> None:
    await conn.execute(
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
    await conn.commit()


async def iter_faces_async(conn: aiosqlite.Connection) -> List[sqlite3.Row]:
    async with conn.execute(
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
    ) as cur:
        rows = await cur.fetchall()
    return rows


async def get_faces_embeddings_async(conn: aiosqlite.Connection) -> List[sqlite3.Row]:
    async with conn.execute("SELECT id, embedding FROM faces") as cur:
        rows = await cur.fetchall()
    return rows


async def get_unassigned_faces_embeddings_async(conn: aiosqlite.Connection) -> List[sqlite3.Row]:
    """Get only unassigned faces (person_id IS NULL) with embeddings."""
    async with conn.execute(
        "SELECT id, embedding FROM faces WHERE person_id IS NULL AND embedding IS NOT NULL"
    ) as cur:
        rows = await cur.fetchall()
    return rows


async def get_person_faces_embeddings_async(conn: aiosqlite.Connection) -> List[sqlite3.Row]:
    """Get faces that are already assigned to persons with their embeddings."""
    async with conn.execute(
        """
        SELECT f.id, f.embedding, f.person_id, p.name as person_name
        FROM faces f
        JOIN persons p ON f.person_id = p.id
        WHERE f.embedding IS NOT NULL
        """
    ) as cur:
        rows = await cur.fetchall()
    return rows


async def set_face_person_async(conn: aiosqlite.Connection, face_id: int, person_id: int) -> None:
    """Set person for a single face (for backward compatibility)."""
    await conn.execute("UPDATE faces SET person_id=? WHERE id=?", (person_id, face_id))
    await conn.commit()


async def set_faces_person_batch_async(
    conn: aiosqlite.Connection,
    assignments: List[tuple[int, int]]
) -> None:
    """
    Batch update person assignments for multiple faces.
    
    Args:
        conn: Database connection
        assignments: List of (face_id, person_id) tuples
    """
    if not assignments:
        return
    
    # SQL expects (person_id, face_id) but assignments are (face_id, person_id)
    # Swap the order to match SQL: SET person_id=? WHERE id=?
    await conn.executemany(
        "UPDATE faces SET person_id=? WHERE id=?",
        [(person_id, face_id) for face_id, person_id in assignments]
    )
    await conn.commit()


async def get_or_create_person_async(conn: aiosqlite.Connection, name: str) -> int:
    async with conn.execute("SELECT id FROM persons WHERE name=?", (name,)) as cur:
        row = await cur.fetchone()

    if row:
        return row["id"]

    async with conn.execute("INSERT INTO persons (name) VALUES (?)", (name,)) as cur:
        await conn.commit()
        return cur.lastrowid


async def create_person_async(conn: aiosqlite.Connection, name: str) -> int:
    """Create a new person. Raises error if name already exists."""
    async with conn.execute("INSERT INTO persons (name) VALUES (?)", (name,)) as cur:
        await conn.commit()
        return cur.lastrowid


async def rename_person_async(conn: aiosqlite.Connection, person_id: int, new_name: str) -> None:
    await conn.execute("UPDATE persons SET name=? WHERE id=?", (new_name, person_id))
    await conn.commit()


async def list_persons_async(conn: aiosqlite.Connection) -> List[sqlite3.Row]:
    async with conn.execute(
        """
        SELECT p.id, p.name, COUNT(f.id) as face_count
        FROM persons p
        LEFT JOIN faces f ON f.person_id = p.id
        GROUP BY p.id, p.name
        ORDER BY p.id ASC
        """
    ) as cur:
        rows = await cur.fetchall()
    return rows


async def get_faces_for_person_async(
    conn: aiosqlite.Connection,
    person_name: str,
) -> List[sqlite3.Row]:
    async with conn.execute(
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
    ) as cur:
        rows = await cur.fetchall()
    return rows


async def set_face_thumb_async(
    conn: aiosqlite.Connection,
    face_id: int,
    thumb_path: str,
) -> None:
    await conn.execute(
        """
        INSERT INTO face_thumbs (face_id, thumb_path)
        VALUES (?, ?)
        ON CONFLICT(face_id) DO UPDATE SET
          thumb_path=excluded.thumb_path
        """,
        (face_id, thumb_path),
    )
    await conn.commit()


async def get_face_thumbs_async(conn: aiosqlite.Connection) -> List[sqlite3.Row]:
    async with conn.execute(
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
    ) as cur:
        rows = await cur.fetchall()
    return rows


async def get_face_thumbs_for_person_async(
    conn: aiosqlite.Connection,
    person_id: int,
) -> List[sqlite3.Row]:
    """Get face thumbnails for a specific person by person_id."""
    async with conn.execute(
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
        """,
        (person_id,),
    ) as cur:
        rows = await cur.fetchall()
    return rows


async def merge_persons_async(
    conn: aiosqlite.Connection,
    source_person_id: int,
    target_person_id: int,
) -> None:
    """Merge source person into target person. All faces assigned to source will be reassigned to target."""
    if source_person_id == target_person_id:
        raise ValueError("Cannot merge a person with itself")

    # Update all faces from source person to target person
    await conn.execute(
        "UPDATE faces SET person_id=? WHERE person_id=?",
        (target_person_id, source_person_id),
    )

    # Delete the source person
    await conn.execute("DELETE FROM persons WHERE id=?", (source_person_id,))

    await conn.commit()


async def delete_persons_without_faces_async(conn: aiosqlite.Connection) -> int:
    """
    Delete all persons that have no faces assigned to them.
    
    Returns:
        Number of persons deleted
    """
    # Delete persons that have no faces
    cursor = await conn.execute(
        """
        DELETE FROM persons
        WHERE id NOT IN (SELECT DISTINCT person_id FROM faces WHERE person_id IS NOT NULL)
        """
    )
    deleted_count = cursor.rowcount
    await conn.commit()
    return deleted_count
