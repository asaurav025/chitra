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

from core import db


DB_DEFAULT_PATH = "photo.db"

# Re-exported so callers do not have to know which of the two duplicated schema
# modules owns the constant. There is exactly one definition, in `core.db`.
DEFAULT_EMBED_MODEL = db.DEFAULT_EMBED_MODEL
DEFAULT_TAG_SOURCE = db.DEFAULT_TAG_SOURCE

# Same rule for the read-side model switch: the sync CLI paths need it too, so
# it is defined once in `core.db` and re-exported here rather than copied.
# `core.db_async.active_embed_model` stays the spelling the API and the scoped
# rules use, and it is the same function object.
active_embed_model = db.active_embed_model


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

    # Foreign key enforcement is per-connection and OFF by default in SQLite.
    # Without it every ON DELETE CASCADE in the schema (tags, embeddings,
    # clusters, faces, face_thumbs) is silently a no-op and deleting a photo
    # leaves its child rows behind as orphans. Deliberately NOT wrapped in a
    # swallowing try/except: this is a correctness pragma, not a perf tuning
    # one, and a silent failure here would reintroduce the bug invisibly.
    await conn.execute("PRAGMA foreign_keys=ON")

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
                thumb_path TEXT,
                media_type TEXT,
                duration_seconds REAL,
                width INTEGER,
                height INTEGER,
                playback_path TEXT,
                transcode_status TEXT,
                video_codec TEXT
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

        # Users (authentication)
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL CHECK(role IN ('admin', 'user')),
                is_active BOOLEAN DEFAULT 1,
                is_whitelisted BOOLEAN DEFAULT 0,
                whitelisted_at TEXT,
                whitelisted_by INTEGER,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                last_login TEXT,
                FOREIGN KEY(whitelisted_by) REFERENCES users(id)
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
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_users_username ON users(username)")
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_users_email ON users(email)")
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_users_whitelisted ON users(is_whitelisted)")

        # Migration: Add thumb_path column if it doesn't exist
        try:
            await conn.execute("ALTER TABLE photos ADD COLUMN thumb_path TEXT")
        except aiosqlite.OperationalError:
            # Column already exists, ignore
            pass

        # Migration: Add video columns (NULL on existing rows; NULL media_type == photo)
        for _ddl in (
            "ALTER TABLE photos ADD COLUMN media_type TEXT",
            "ALTER TABLE photos ADD COLUMN duration_seconds REAL",
            "ALTER TABLE photos ADD COLUMN width INTEGER",
            "ALTER TABLE photos ADD COLUMN height INTEGER",
            "ALTER TABLE photos ADD COLUMN playback_path TEXT",
            "ALTER TABLE photos ADD COLUMN transcode_status TEXT",
            "ALTER TABLE photos ADD COLUMN video_codec TEXT",
        ):
            try:
                await conn.execute(_ddl)
            except aiosqlite.OperationalError:
                pass

        # Migration: Add users table columns if they don't exist (for existing databases)
        try:
            await conn.execute("ALTER TABLE users ADD COLUMN is_whitelisted BOOLEAN DEFAULT 0")
        except aiosqlite.OperationalError:
            pass
        try:
            await conn.execute("ALTER TABLE users ADD COLUMN whitelisted_at TEXT")
        except aiosqlite.OperationalError:
            pass
        try:
            await conn.execute("ALTER TABLE users ADD COLUMN whitelisted_by INTEGER")
        except aiosqlite.OperationalError:
            pass

        # Migration: per-row model/vocabulary provenance, and the uniqueness it
        # keys. The statements are imported from `core.db` rather than retyped:
        # these two modules already diverged once (the sync copy has no `users`
        # table) and this migration must not become the second divergence.
        #
        # The `embeddings` key is (photo_id, model), NOT (photo_id) — a bare
        # photo_id key would make the first SigLIP row evict the CLIP row that
        # search is still answering from, destroying both coexistence during
        # the migration and rollback-by-config.
        for _ddl in db.MIGRATION_COLUMNS:
            try:
                await conn.execute(_ddl)
            except aiosqlite.OperationalError:
                pass
        for _sql, _params in db.MIGRATION_BACKFILLS:
            try:
                await conn.execute(_sql, _params)
            except aiosqlite.OperationalError:
                pass
        for _name, _ddl in db.MIGRATION_UNIQUE_INDEXES:
            try:
                await conn.execute(_ddl)
            except aiosqlite.OperationalError as exc:
                # Never swallowed: a missing unique index means the table holds
                # duplicates and every writer's upsert will fail. Silence here
                # would leave the constraint absent with nobody aware.
                print(
                    f"SCHEMA WARNING: could not create unique index {_name}: {exc}. "
                    f"The table almost certainly holds duplicate rows — writers "
                    f"will fail until they are removed.",
                    flush=True,
                )
        for _ddl in db.MIGRATION_INDEXES:
            try:
                await conn.execute(_ddl)
            except aiosqlite.OperationalError:
                pass

        await conn.commit()


# ----------------------------------------------------------------------
# PHOTOS (ASYNC)
# ----------------------------------------------------------------------
_VIDEO_FIELD_COLUMNS = (
    "media_type",
    "duration_seconds",
    "width",
    "height",
    "playback_path",
    "transcode_status",
    "video_codec",
)


async def upsert_photo_async(conn: aiosqlite.Connection, **meta: Any) -> None:
    """
    Insert or update a photo row (async).
    Expected keys: file_path, size, created_at, checksum, phash,
                   exif_datetime, latitude, longitude, thumb_path, media_type,
                   width, height, duration_seconds (all optional)
    """
    meta.setdefault("thumb_path", None)
    meta.setdefault("media_type", "photo")
    meta.setdefault("width", None)
    meta.setdefault("height", None)
    meta.setdefault("duration_seconds", None)

    await conn.execute(
        """
        INSERT INTO photos (file_path, size, created_at, checksum, phash, exif_datetime,
                            latitude, longitude, thumb_path, media_type, width, height, duration_seconds)
        VALUES (:file_path, :size, :created_at, :checksum, :phash, :exif_datetime,
                :latitude, :longitude, :thumb_path, :media_type, :width, :height, :duration_seconds)
        ON CONFLICT(file_path) DO UPDATE SET
            size=excluded.size,
            created_at=excluded.created_at,
            checksum=excluded.checksum,
            phash=excluded.phash,
            exif_datetime=excluded.exif_datetime,
            latitude=excluded.latitude,
            longitude=excluded.longitude,
            thumb_path=COALESCE(excluded.thumb_path, photos.thumb_path),
            media_type=excluded.media_type,
            width=COALESCE(excluded.width, photos.width),
            height=COALESCE(excluded.height, photos.height),
            duration_seconds=COALESCE(excluded.duration_seconds, photos.duration_seconds)
        """,
        meta,
    )
    await conn.commit()


async def update_video_fields_async(conn: aiosqlite.Connection, photo_id: int, **fields: Any) -> None:
    """Update video-specific columns for a photo (async)."""
    cols = [k for k in fields if k in _VIDEO_FIELD_COLUMNS]
    if not cols:
        return
    set_clause = ", ".join(f"{c}=?" for c in cols)
    params = [fields[c] for c in cols]
    params.append(photo_id)
    await conn.execute(f"UPDATE photos SET {set_clause} WHERE id=?", params)
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
    model: str | None = None,
) -> None:
    """Async mirror of `db.put_embedding` — see that docstring for why the key
    is `(photo_id, model)` rather than `(photo_id)`."""
    await conn.execute(
        """
        INSERT INTO embeddings (photo_id, dim, vector, model)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(photo_id, model) DO UPDATE SET
          dim = excluded.dim,
          vector = excluded.vector
        """,
        (photo_id, dim, vec_bytes, model or DEFAULT_EMBED_MODEL),
    )
    await conn.commit()


async def get_embeddings_async(
    conn: aiosqlite.Connection,
    model: str | None = None,
) -> List[Tuple[int, int, bytes]]:
    """Read stored vectors, optionally restricted to one embedding model.

    The table is keyed on `(photo_id, model)` so two generations can coexist
    during a re-embed. Callers that stack the vectors into a matrix — search,
    `/api/photos/{id}/similar` — MUST pass `model`: rows from two models have
    different dimensions, and `np.stack` on a mixed list raises
    `ValueError: all input arrays must have the same shape`, which turns one
    768-d row into a 500 on every query. Filtering here rather than in the
    caller means the mixed list never exists.

    Scores from two models are not comparable anyway (CLIP cosine occupies
    0.16-0.28; a sigmoid-trained model does not), so merging them would be
    silently meaningless even where the shapes happened to agree.

    `model=None` stays an unfiltered read for callers that genuinely want
    every row — migration and coverage counting.
    """
    if model is None:
        sql, params = "SELECT photo_id, dim, vector FROM embeddings", ()
    else:
        sql = "SELECT photo_id, dim, vector FROM embeddings WHERE model = ?"
        params = (model,)
    async with conn.execute(sql, params) as cur:
        rows = await cur.fetchall()
    return [(row["photo_id"], row["dim"], row["vector"]) for row in rows]


# ----------------------------------------------------------------------
# TAGS (ASYNC)
# ----------------------------------------------------------------------
async def add_tag_async(
    conn: aiosqlite.Connection,
    photo_id: int,
    tag: str,
    score: float,
    source: str | None = None,
) -> None:
    """Async mirror of `db.add_tag`. Keyed on `(photo_id, tag)`; `source` is
    provenance and deliberately not part of the key."""
    await conn.execute(
        """
        INSERT INTO tags (photo_id, tag, score, source)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(photo_id, tag) DO UPDATE SET
          score = excluded.score,
          source = excluded.source
        """,
        (photo_id, tag, score, source or DEFAULT_TAG_SOURCE),
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


# ----------------------------------------------------------------------
# USERS (AUTHENTICATION) (ASYNC)
# ----------------------------------------------------------------------
async def create_user_async(
    conn: aiosqlite.Connection,
    username: str,
    password_hash: str,
    email: Optional[str] = None,
    role: str = "user",
    auto_whitelist: bool = False
) -> int:
    """
    Create a new user.
    
    Args:
        conn: Database connection
        username: Username (must be unique)
        password_hash: Hashed password
        email: Optional email address
        role: User role ('admin' or 'user')
        auto_whitelist: If True, automatically whitelist the user (for first admin)
    
    Returns:
        User ID
    """
    is_whitelisted = 1 if (auto_whitelist or role == "admin") else 0
    whitelisted_at = None
    if is_whitelisted:
        from datetime import datetime
        whitelisted_at = datetime.utcnow().isoformat()
    
    async with conn.execute(
        """
        INSERT INTO users (username, email, password_hash, role, is_whitelisted, whitelisted_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (username, email, password_hash, role, is_whitelisted, whitelisted_at)
    ) as cur:
        await conn.commit()
        return cur.lastrowid


async def get_user_by_username_async(
    conn: aiosqlite.Connection,
    username: str
) -> Optional[sqlite3.Row]:
    """Get user by username."""
    async with conn.execute(
        "SELECT * FROM users WHERE username = ?",
        (username,)
    ) as cur:
        return await cur.fetchone()


async def get_user_by_id_async(
    conn: aiosqlite.Connection,
    user_id: int
) -> Optional[sqlite3.Row]:
    """Get user by ID."""
    async with conn.execute(
        "SELECT * FROM users WHERE id = ?",
        (user_id,)
    ) as cur:
        return await cur.fetchone()


async def update_last_login_async(
    conn: aiosqlite.Connection,
    user_id: int
) -> None:
    """Update user's last login timestamp."""
    from datetime import datetime
    await conn.execute(
        "UPDATE users SET last_login = ? WHERE id = ?",
        (datetime.utcnow().isoformat(), user_id)
    )
    await conn.commit()


async def get_pending_users_async(conn: aiosqlite.Connection) -> List[sqlite3.Row]:
    """Get all users waiting for whitelist approval."""
    async with conn.execute(
        "SELECT * FROM users WHERE is_whitelisted = 0 AND is_active = 1 ORDER BY created_at ASC"
    ) as cur:
        rows = await cur.fetchall()
    return rows


async def whitelist_user_async(
    conn: aiosqlite.Connection,
    user_id: int,
    whitelisted_by: int
) -> None:
    """Whitelist a user (admin action)."""
    from datetime import datetime
    await conn.execute(
        """
        UPDATE users 
        SET is_whitelisted = 1,
            whitelisted_at = ?,
            whitelisted_by = ?
        WHERE id = ?
        """,
        (datetime.utcnow().isoformat(), whitelisted_by, user_id)
    )
    await conn.commit()


async def unwhitelist_user_async(conn: aiosqlite.Connection, user_id: int) -> None:
    """Remove user from whitelist."""
    await conn.execute(
        """
        UPDATE users 
        SET is_whitelisted = 0, 
            whitelisted_at = NULL, 
            whitelisted_by = NULL 
        WHERE id = ?
        """,
        (user_id,)
    )
    await conn.commit()


async def list_users_async(
    conn: aiosqlite.Connection,
    include_pending: bool = False
) -> List[sqlite3.Row]:
    """List users, optionally including pending (non-whitelisted) ones."""
    if include_pending:
        async with conn.execute(
            "SELECT * FROM users ORDER BY created_at DESC"
        ) as cur:
            rows = await cur.fetchall()
    else:
        async with conn.execute(
            "SELECT * FROM users WHERE is_whitelisted = 1 ORDER BY created_at DESC"
        ) as cur:
            rows = await cur.fetchall()
    return rows


async def update_user_role_async(
    conn: aiosqlite.Connection,
    user_id: int,
    role: str
) -> None:
    """Update user role."""
    if role not in ("admin", "user"):
        raise ValueError("Role must be 'admin' or 'user'")
    await conn.execute(
        "UPDATE users SET role = ? WHERE id = ?",
        (role, user_id)
    )
    await conn.commit()


async def deactivate_user_async(conn: aiosqlite.Connection, user_id: int) -> None:
    """Deactivate a user account."""
    await conn.execute(
        "UPDATE users SET is_active = 0 WHERE id = ?",
        (user_id,)
    )
    await conn.commit()
