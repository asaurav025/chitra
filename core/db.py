from __future__ import annotations
import sqlite3
from typing import Iterable, Tuple, List, Optional

DEFAULT_DB = "photo.db"

SCHEMA = [
    """CREATE TABLE IF NOT EXISTS photos (
        id INTEGER PRIMARY KEY,
        file_path TEXT UNIQUE,
        size INTEGER,
        created_at TEXT,
        checksum TEXT,
        phash TEXT,
        exif_datetime TEXT,
        latitude REAL,
        longitude REAL
    );""",
    """CREATE TABLE IF NOT EXISTS embeddings (
        photo_id INTEGER PRIMARY KEY,
        dim INTEGER NOT NULL,
        vector BLOB NOT NULL,
        FOREIGN KEY(photo_id) REFERENCES photos(id)
    );""",
    """CREATE TABLE IF NOT EXISTS tags (
        photo_id INTEGER,
        tag TEXT,
        score REAL,
        FOREIGN KEY(photo_id) REFERENCES photos(id)
    );""",
    """CREATE TABLE IF NOT EXISTS clusters (
        photo_id INTEGER,
        cluster_id INTEGER,
        similarity REAL,
        FOREIGN KEY(photo_id) REFERENCES photos(id)
    );""",
    """CREATE TABLE IF NOT EXISTS faces (
        photo_id INTEGER,
        face_id INTEGER,
        encoding BLOB,
        FOREIGN KEY(photo_id) REFERENCES photos(id)
    );"""
]

def connect(db_path: str = DEFAULT_DB) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    return conn

def init_db(db_path: str = DEFAULT_DB) -> None:
    conn = connect(db_path)
    try:
        for stmt in SCHEMA:
            conn.execute(stmt)
        conn.commit()
    finally:
        conn.close()

def upsert_photo(conn: sqlite3.Connection, **fields) -> int:
    cur = conn.cursor()
    cur.execute(
        """INSERT INTO photos (file_path, size, created_at, checksum, phash, exif_datetime, latitude, longitude)
               VALUES (:file_path, :size, :created_at, :checksum, :phash, :exif_datetime, :latitude, :longitude)
               ON CONFLICT(file_path) DO UPDATE SET
                   size=excluded.size,
                   created_at=excluded.created_at,
                   checksum=excluded.checksum,
                   phash=excluded.phash,
                   exif_datetime=excluded.exif_datetime,
                   latitude=excluded.latitude,
                   longitude=excluded.longitude
        """, fields,
    )
    conn.commit()
    cur.execute("SELECT id FROM photos WHERE file_path=?", (fields["file_path"],))
    return cur.fetchone()[0]

def put_embedding(conn: sqlite3.Connection, photo_id: int, vector_bytes: bytes, dim: int) -> None:
    conn.execute(
        "INSERT INTO embeddings (photo_id, dim, vector) VALUES (?, ?, ?) ON CONFLICT(photo_id) DO UPDATE SET dim=excluded.dim, vector=excluded.vector",
        (photo_id, dim, vector_bytes),
    )
    conn.commit()

def add_tag(conn: sqlite3.Connection, photo_id: int, tag: str, score: float) -> None:
    conn.execute("INSERT INTO tags (photo_id, tag, score) VALUES (?, ?, ?)", (photo_id, tag, score))
    conn.commit()

def assign_cluster(conn: sqlite3.Connection, photo_id: int, cluster_id: int, similarity: float) -> None:
    conn.execute("INSERT INTO clusters (photo_id, cluster_id, similarity) VALUES (?,?,?)", (photo_id, cluster_id, similarity))
    conn.commit()

def add_face(conn: sqlite3.Connection, photo_id: int, face_id: int, encoding_bytes: bytes) -> None:
    conn.execute("INSERT INTO faces (photo_id, face_id, encoding) VALUES (?, ?, ?)", (photo_id, face_id, encoding_bytes))
    conn.commit()

def iter_photos(conn: sqlite3.Connection) -> Iterable[Tuple[int, str]]:
    for row in conn.execute("SELECT id, file_path FROM photos"):
        yield row

def get_embeddings(conn: sqlite3.Connection):
    cur = conn.cursor()
    cur.execute("SELECT photo_id, dim, vector FROM embeddings")
    return cur.fetchall()
