from __future__ import annotations
import logging
import os
import sqlite3
from pathlib import Path
from typing import Iterable, List, Tuple, Dict, Optional, Any

_log = logging.getLogger(__name__)


DB_DEFAULT_PATH = "photo.db"

# Identifies which model produced a vector. Stored per row rather than assumed
# globally, because the CLIP -> SigLIP migration needs both generations resident
# in `embeddings` at once: search answers from one of them while the other is
# still being written, and rollback is then a config change rather than a full
# re-embed off a failing disk.
DEFAULT_EMBED_MODEL = "openai/clip-vit-base-patch32"
# Which model *and vocabulary* produced a tag. Provenance, not identity — it is
# deliberately not part of the tags unique key.
DEFAULT_TAG_SOURCE = "clip-vitb32/vocab-v1"


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

    # Foreign key enforcement is per-connection and OFF by default in SQLite.
    # Without it every ON DELETE CASCADE in the schema (tags, embeddings,
    # clusters, faces, face_thumbs) is silently a no-op and deleting a photo
    # leaves its child rows behind as orphans. Deliberately NOT wrapped in a
    # swallowing try/except: this is a correctness pragma, not a perf tuning
    # one, and a silent failure here would reintroduce the bug invisibly.
    conn.execute("PRAGMA foreign_keys=ON")

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
            cur.execute(_ddl)
        except sqlite3.OperationalError:
            pass

    migrate_embeddings_and_tags(conn)

    conn.commit()
    conn.close()


# Kept as data so `db_async.py` can mirror it verbatim. The two modules carry
# duplicated DDL that has already diverged once (the sync copy has no `users`
# table); this migration must not become the second divergence.
MIGRATION_COLUMNS = (
    "ALTER TABLE embeddings ADD COLUMN model TEXT",
    "ALTER TABLE tags ADD COLUMN source TEXT",
)
MIGRATION_BACKFILLS = (
    ("UPDATE embeddings SET model = ? WHERE model IS NULL", (DEFAULT_EMBED_MODEL,)),
    ("UPDATE tags SET source = ? WHERE source IS NULL", (DEFAULT_TAG_SOURCE,)),
)
MIGRATION_UNIQUE_INDEXES = (
    ("idx_embeddings_photo_model",
     "CREATE UNIQUE INDEX IF NOT EXISTS idx_embeddings_photo_model "
     "ON embeddings(photo_id, model)"),
    ("idx_tags_photo_tag",
     "CREATE UNIQUE INDEX IF NOT EXISTS idx_tags_photo_tag ON tags(photo_id, tag)"),
)
MIGRATION_INDEXES = (
    "CREATE INDEX IF NOT EXISTS idx_tags_tag ON tags(tag)",
)


def migrate_embeddings_and_tags(conn: sqlite3.Connection) -> None:
    """Add the `model`/`source` columns and the uniqueness they key.

    Additive and idempotent, like every other migration here — there is no
    migration framework and no schema-version table, so this runs on every
    startup and must be safe to run on an already-migrated database.

    The unique index on `embeddings` is `(photo_id, model)` and **not**
    `(photo_id)`. A bare `photo_id` key would make the first SigLIP row evict
    the CLIP row it is meant to run alongside, which destroys both coexistence
    during the migration and the ability to roll back by config.

    A failed `ADD COLUMN` is expected (the column is already there) and
    swallowed. A failed **unique index** is not: it means the table already
    holds duplicates, the constraint is absent, and every writer's upsert will
    then fail with "ON CONFLICT clause does not match any ... UNIQUE
    constraint". That has to be loud — `scripts/dedupe_embeddings.py` is the
    escape hatch, not silence.
    """
    for ddl in MIGRATION_COLUMNS:
        try:
            conn.execute(ddl)
        except sqlite3.OperationalError:
            pass  # column already present

    for sql, params in MIGRATION_BACKFILLS:
        try:
            conn.execute(sql, params)
        except sqlite3.OperationalError:
            pass  # table not created yet on a partial schema

    for name, ddl in MIGRATION_UNIQUE_INDEXES:
        try:
            conn.execute(ddl)
        except sqlite3.OperationalError as exc:
            print(
                f"SCHEMA WARNING: could not create unique index {name}: {exc}. "
                f"The table almost certainly holds duplicate rows — writers will "
                f"fail until they are removed.",
                flush=True,
            )

    for ddl in MIGRATION_INDEXES:
        try:
            conn.execute(ddl)
        except sqlite3.OperationalError:
            pass

    conn.commit()


# ----------------------------------------------------------------------
# PHOTOS
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


def upsert_photo(conn: sqlite3.Connection, **meta: Any):
    """
    Insert or update a photo row.
    Expected keys: file_path, size, created_at, checksum, phash, exif_datetime,
                   latitude, longitude, thumb_path, media_type, width, height, duration_seconds
    """
    meta.setdefault("thumb_path", None)
    meta.setdefault("media_type", "photo")
    meta.setdefault("width", None)
    meta.setdefault("height", None)
    meta.setdefault("duration_seconds", None)

    conn.execute(
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
    conn.commit()


def update_video_fields(conn: sqlite3.Connection, photo_id: int, **fields: Any):
    """Update video-specific columns for a photo (transcode job writeback)."""
    cols = [k for k in fields if k in _VIDEO_FIELD_COLUMNS]
    if not cols:
        return
    set_clause = ", ".join(f"{c}=?" for c in cols)
    params = [fields[c] for c in cols]
    params.append(photo_id)
    conn.execute(f"UPDATE photos SET {set_clause} WHERE id=?", params)
    conn.commit()


def update_capture_date(
    conn: sqlite3.Connection, photo_id: int, created_at: str, exif_datetime: str
):
    """Set the capture-date columns for a photo/video (used by the video date backfill).

    `update_video_fields` only writes the video-specific columns, so it can't touch
    `created_at`/`exif_datetime`.
    """
    conn.execute(
        "UPDATE photos SET created_at=?, exif_datetime=? WHERE id=?",
        (created_at, exif_datetime, photo_id),
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
def put_embedding(
    conn: sqlite3.Connection,
    photo_id: int,
    vec_bytes: bytes,
    dim: int,
    model: Optional[str] = None,
):
    """Store this photo's vector *for this model*, replacing any it already had.

    This used to be a plain INSERT into a table with no unique constraint, so
    embedding the same photo twice left two rows — and `search_photos` stacks
    every row `get_embeddings` returns, so the photo would occupy two result
    slots and score twice. Nothing noticed because nothing had ever re-embedded.

    Scoped to `(photo_id, model)`: writing a SigLIP vector must **not** delete
    the CLIP vector that search is still answering from. That is what makes the
    512 -> 768 migration incremental and its rollback a config change.
    """
    conn.execute(
        """
        INSERT INTO embeddings (photo_id, dim, vector, model)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(photo_id, model) DO UPDATE SET
          dim = excluded.dim,
          vector = excluded.vector
        """,
        (photo_id, dim, vec_bytes, model or DEFAULT_EMBED_MODEL),
    )
    conn.commit()


def get_embeddings(
    conn: sqlite3.Connection,
    model: Optional[str] = None,
) -> List[Tuple[int, int, bytes]]:
    """Read stored vectors, optionally restricted to one embedding model.

    Sync twin of `db_async.get_embeddings_async`, with the same contract and
    for the same reason. The table is keyed on `(photo_id, model)` so two
    generations coexist during a re-embed; callers that stack the vectors into
    a matrix — the CLI `search` and `cluster` commands — MUST pass `model`,
    because rows from two models have different dimensions and `np.stack` on a
    mixed list raises `ValueError: all input arrays must have the same shape`.
    Filtering here rather than in the caller means the mixed list never exists.

    Scores from two models are not comparable anyway (CLIP cosine occupies
    0.16-0.28; a sigmoid-trained model does not), so merging them would be
    silently meaningless even where the shapes happened to agree.

    `model=None` stays an unfiltered read for callers that genuinely want every
    row — migration and coverage counting.
    """
    cur = conn.cursor()
    if model is None:
        cur.execute("SELECT photo_id, dim, vector FROM embeddings")
    else:
        cur.execute(
            "SELECT photo_id, dim, vector FROM embeddings WHERE model = ?",
            (model,),
        )
    return [(row["photo_id"], row["dim"], row["vector"]) for row in cur.fetchall()]


def active_embed_model() -> str:
    """The single model every ranking read filters to.

    Lives here, not in `db_async`, for the same reason `DEFAULT_EMBED_MODEL`
    and `migrate_embeddings_and_tags` do: both halves of the duplicated schema
    layer need it, and a second copy would be a second default model name. The
    CLIP identifier is already spelled as an independent literal in four
    modules; two spellings of it write rows under one name while search filters
    on another, and every query silently returns nothing.

    Read at call time, not import time: the cutover in the re-embed plan is one
    environment variable flip plus a restart, and the rollback is flipping it
    back. Defaulting to the CLIP identifier means an unset variable keeps
    answering from the rows every existing photo already has, rather than from
    nothing.

    Deliberately *not* `CHITRA_EMBED_MODEL`, which says what the sidecar
    computes with. The migration needs a window where the sidecar already
    writes 768-d SigLIP rows while reads still answer from the complete set of
    512-d CLIP rows; one variable driving both would collapse that window.
    """
    return os.environ.get("CHITRA_ACTIVE_EMBED_MODEL", DEFAULT_EMBED_MODEL)


#: Default `min_score` for `/api/search/photos`, per embedding model.
#:
#: A raw text-image cosine is not comparable across models, so neither is a
#: floor on it. CLIP's cosines sit in 0.16-0.28 on this library and 0.2 was a
#: sensible cut. SigLIP 2's sigmoid objective puts the *best* hit for any query
#: near 0.135 (measured 2026-09-02 over 2,721 vectors; library median ~0.05),
#: so the same 0.2 returned nothing for every query, with a 200 and a green
#: health check. 0.09 was chosen from that probe: nonsense queries clear it 0-1
#: times, real ones 5-140 times, and 0.08 lets ~25 junk rows through while
#: 0.10 starves rare-but-real subjects to a single hit.
#:
#: Keyed on the model that produced the vectors, so a cutover or rollback of
#: `CHITRA_ACTIVE_EMBED_MODEL` carries the right floor with it and clients
#: never need to know which model is live.
SEARCH_MIN_SCORE_BY_MODEL = {
    "openai/clip-vit-base-patch32": 0.2,
    "google/siglip2-base-patch16-224": 0.09,
}

#: Operator override for the search floor, in `.env.production`. Tuning is
#: then a config change plus a restart, exactly like the model switch.
SEARCH_MIN_SCORE_ENV = "CHITRA_SEARCH_MIN_SCORE"


def search_min_score(model: Optional[str] = None) -> float:
    """The floor below which a photo is not a search result. Server-owned.

    `CHITRA_SEARCH_MIN_SCORE` wins when set. Otherwise the entry for `model`
    (the active model when unset) in `SEARCH_MIN_SCORE_BY_MODEL`; a model with
    no tuned floor gets 0.0 — a ranked list of everything — rather than
    borrowing another model's number, which is precisely the failure this
    exists to prevent.

    Clients get no say. Every client used to send CLIP's 0.2 and the SigLIP
    cutover answered `200 []` for an afternoon. A malformed or out-of-range
    override is logged and ignored rather than turning every search into a
    500 — or, worse, into a silent empty list.
    """
    raw = os.environ.get(SEARCH_MIN_SCORE_ENV, "").strip()
    if raw:
        try:
            value = float(raw)
            if 0.0 <= value <= 1.0:
                return value
            _log.warning("%s=%r is outside 0.0-1.0; using the model floor instead",
                         SEARCH_MIN_SCORE_ENV, raw)
        except ValueError:
            _log.warning("%s=%r is not a number; using the model floor instead",
                         SEARCH_MIN_SCORE_ENV, raw)
    return SEARCH_MIN_SCORE_BY_MODEL.get(model or active_embed_model(), 0.0)


# ----------------------------------------------------------------------
# TAGS
# ----------------------------------------------------------------------
def add_tag(
    conn: sqlite3.Connection,
    photo_id: int,
    tag: str,
    score: float,
    source: Optional[str] = None,
):
    """Attach one tag, replacing that same tag on that photo if it exists.

    Same hazard as `put_embedding`: `tags` had no unique constraint, so
    re-tagging a photo appended a second copy of every label. Keyed on
    `(photo_id, tag)` — `source` records which model and vocabulary produced
    the label and is provenance, not identity, so re-scoring a photo under a
    new vocabulary updates the label rather than duplicating it.
    """
    conn.execute(
        """
        INSERT INTO tags (photo_id, tag, score, source)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(photo_id, tag) DO UPDATE SET
          score = excluded.score,
          source = excluded.source
        """,
        (photo_id, tag, score, source or DEFAULT_TAG_SOURCE),
    )
    conn.commit()


def replace_tags(
    conn: sqlite3.Connection,
    photo_id: int,
    tags: Iterable[Tuple[str, float]],
    source: Optional[str] = None,
):
    """Swap a photo's entire tag set in one transaction.

    `add_tag` in a loop cannot drop a label that the new vocabulary no longer
    predicts, so a photo re-tagged after the label list changed would keep the
    union of every list it was ever scored against.
    """
    with conn:
        conn.execute("DELETE FROM tags WHERE photo_id = ?", (photo_id,))
        conn.executemany(
            "INSERT INTO tags (photo_id, tag, score, source) VALUES (?, ?, ?, ?)",
            [(photo_id, tag, float(score), source or DEFAULT_TAG_SOURCE)
             for tag, score in tags],
        )


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
