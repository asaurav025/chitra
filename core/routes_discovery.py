"""Read-only discovery endpoints: tags, tag search, similar photos, duplicates.

Four things the library could already answer and did not expose. Every one of
them reads SQLite and nothing else — no MinIO object, no model, no FAISS index
— which is why they could ship ahead of the re-embed campaign.

They live here rather than in `app_fastapi.py` because that file is a
2,500-line god-file holding middleware, DI, DTO mappers and all 40 routes;
`.claude/rules/http-layer.md` asks for new endpoint groups as routers. The
whole of `app_fastapi.py`'s share of this feature is one import and one
`include_router` call.

**Why a factory rather than a module-level `router`.** These routes need
`get_current_active_user`, `get_db_async` and `row_to_photo_dto`, all of which
live in `app_fastapi.py` — which imports this module. Importing it back would
be a cycle whose failure mode depends on which module a process happens to
import first, so the dependencies are passed in instead. That also keeps
`app.dependency_overrides[app_fastapi.get_db_async]` working: FastAPI matches
overrides by object identity, and these are the same objects.

**Nothing here may import torch or `core.embedder`.** `import app_fastapi` has
a 200 MB budget with no ML resident (`tests/test_api_memory_budget.py`), and
this module is on that import path.
"""
from __future__ import annotations

import asyncio
import functools
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Query

from core import db_async
from core.schemas import PhotoResponse, SearchResultsResponse


# ----------------------------------------------------------------------
# pHash near-duplicate grouping
# ----------------------------------------------------------------------
# All-pairs Hamming over the library's 64-bit pHashes: measured 0.18 s and
# 23.5 MB peak over 1,715 hashes, SQLite only. Kept a pure function so the
# handler can hand it to an executor unchanged, and so the closure below is
# testable without a database.
_DEFAULT_CHUNK = 512


def _parse_hex64(value: Any) -> Optional[int]:
    """A 64-bit pHash as stored by `imagehash.phash` — 16 hex characters.

    Anything unparseable is skipped rather than raised on: a single bad row
    must not take the endpoint down for the whole library.
    """
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = int(value, 16)
    except ValueError:
        return None
    if parsed < 0 or parsed >= (1 << 64):
        return None
    return parsed


def group_by_phash(
    rows: Sequence[Tuple[int, Any]],
    max_distance: int = 8,
    chunk: int = _DEFAULT_CHUNK,
) -> List[List[int]]:
    """Group photo ids whose pHashes are within `max_distance` bits.

    Grouping is the **transitive closure** of the "within max_distance"
    relation, not a list of pairs. A burst of near-identical frames drifts:
    A~B and B~C while A and C are twice as far apart, and reporting that as
    two overlapping pairs makes the caller do the union itself and get it
    wrong. Union-find over the pairs gives one group of three.

    The distance matrix is built in row blocks so peak memory stays bounded
    as the library grows — the full N x N XOR is 32 MB at today's 2,013 rows
    but 3.2 GB at 20,000, and this endpoint must not be the thing that OOMs
    the API tier. `chunk` only affects memory, never the answer.

    Returns groups of size >= 2, members ascending, largest group first.
    """
    ids: List[int] = []
    values: List[int] = []
    for photo_id, raw in rows:
        parsed = _parse_hex64(raw)
        if parsed is None:
            continue
        ids.append(photo_id)
        values.append(parsed)

    n = len(ids)
    if n < 2:
        return []

    hashes = np.asarray(values, dtype=np.uint64)

    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    step = max(1, int(chunk))
    for start in range(0, n, step):
        block = hashes[start:start + step]
        # uint8 popcount of the XOR: Hamming distance, vectorised.
        dist = np.bitwise_count(block[:, None] ^ hashes[None, :])
        local, other = np.nonzero(dist <= max_distance)
        rows_i = local + start
        # Each unordered pair once, and never a row against itself.
        keep = other > rows_i
        for i, j in zip(rows_i[keep].tolist(), other[keep].tolist()):
            union(i, j)

    buckets: Dict[int, List[int]] = {}
    for idx in range(n):
        buckets.setdefault(find(idx), []).append(ids[idx])

    groups = [sorted(members) for members in buckets.values() if len(members) > 1]
    groups.sort(key=lambda members: (-len(members), members[0]))
    return groups


# ----------------------------------------------------------------------
# Router
# ----------------------------------------------------------------------
def build_router(
    *,
    current_user_dep: Callable[..., Any],
    db_dep: Callable[..., Any],
    photo_dto: Callable[[Any], Dict[str, Any]],
) -> APIRouter:
    """Build the discovery router against the app's own DI callables.

    `current_user_dep` is passed rather than re-derived so there is exactly
    one auth implementation in the process; a second copy would drift, and a
    route with no auth dependency at all is silent — it simply serves anyone
    who can reach the tunnel.
    """
    router = APIRouter(prefix="/api", tags=["discovery"])

    # ------------------------------------------------------------------
    @router.get("/tags")
    async def list_tags(
        limit: int = Query(500, ge=1, le=5000),
        current_user=Depends(current_user_dep),
        conn=Depends(db_dep),
    ):
        """The tag vocabulary actually present in the library, with counts.

        The score range matters as much as the count. Zero-shot CLIP cosine
        spans 0.158-0.278 for *everything*, and fixed top-6 tagging put
        `travel` on 80.6% of the library at a score indistinguishable from
        the ones it beat. A count alone hides that; min/avg/max shows it.
        """
        async with conn.execute(
            """
            SELECT tag,
                   COUNT(*)   AS count,
                   AVG(score) AS avg_score,
                   MIN(score) AS min_score,
                   MAX(score) AS max_score
            FROM tags
            GROUP BY tag
            ORDER BY count DESC, tag ASC
            LIMIT ?
            """,
            (limit,),
        ) as cur:
            rows = await cur.fetchall()

        tags = [
            {
                "tag": row["tag"],
                "count": row["count"],
                "avg_score": round(float(row["avg_score"]), 6),
                "min_score": round(float(row["min_score"]), 6),
                "max_score": round(float(row["max_score"]), 6),
            }
            for row in rows
        ]
        return {"tags": tags, "total": len(tags)}

    # ------------------------------------------------------------------
    @router.get("/search/by-tag", response_model=SearchResultsResponse)
    async def search_by_tag(
        tag: Optional[str] = Query(None),
        min_score: float = Query(0.0, ge=0.0, le=1.0),
        limit: int = Query(50, ge=1, le=200),
        current_user=Depends(current_user_dep),
        conn=Depends(db_dep),
    ):
        """Photos carrying a tag, ranked by that tag's score.

        `min_score` defaults to 0.0, not to the 0.2 `/api/search/photos`
        uses. A stored tag row already survived tagging's own threshold, and
        an absolute cut over a 0.16-0.28 range would silently drop most of
        the library on a filter the caller never asked for.
        """
        if not tag:
            raise HTTPException(status_code=400, detail="missing_tag")

        async with conn.execute(
            """
            SELECT p.*, t.score AS tag_score
            FROM tags t
            JOIN photos p ON p.id = t.photo_id
            WHERE t.tag = ? AND t.score >= ?
            ORDER BY t.score DESC, p.id DESC
            LIMIT ?
            """,
            (tag, min_score, limit),
        ) as cur:
            rows = await cur.fetchall()

        results = []
        for row in rows:
            record = dict(row)
            dto = photo_dto(record)
            dto["score"] = float(record["tag_score"])
            results.append(PhotoResponse(**dto))

        return SearchResultsResponse(query=tag, results=results)

    # ------------------------------------------------------------------
    @router.get("/photos/{photo_id}/similar")
    async def similar_photos(
        photo_id: int,
        limit: int = Query(20, ge=1, le=100),
        current_user=Depends(current_user_dep),
        conn=Depends(db_dep),
    ):
        """Nearest neighbours of one photo in the stored embedding space.

        Filtered to the active model, and not optionally: `embeddings` is
        keyed on `(photo_id, model)` so a re-embed writes 768-d rows beside
        the 512-d ones, and `np.stack` over a mixed list raises
        `ValueError: all input arrays must have the same shape`. Unfiltered,
        this endpoint dies the moment the first SigLIP row lands.

        Brute force over the whole matrix, exactly like `/api/search/photos`
        — 3.31 MiB and a single GEMV. There is no photo-level FAISS index and
        at this scale there is no case for one.
        """
        cur = await conn.execute("SELECT 1 FROM photos WHERE id = ?", (photo_id,))
        if await cur.fetchone() is None:
            raise HTTPException(status_code=404, detail="photo_not_found")

        rows = await db_async.get_embeddings_async(
            conn, model=db_async.active_embed_model()
        )

        photo_ids: List[int] = []
        vecs: List[np.ndarray] = []
        probe: Optional[np.ndarray] = None
        for other_id, dim, vec_bytes in rows:
            vec = np.frombuffer(vec_bytes, dtype="float32")
            # Belt and braces behind the model filter: catches a row whose
            # blob and `dim` column disagree, which the filter cannot.
            if vec.shape[0] != dim:
                continue
            if other_id == photo_id:
                probe = vec
                continue
            photo_ids.append(other_id)
            vecs.append(vec)

        if probe is None:
            # The photo exists but carries no vector for the active model —
            # unembedded, a video, or mid-cutover.
            raise HTTPException(status_code=404, detail="no_embedding")
        if not vecs:
            return {"photo_id": photo_id, "results": []}

        probe = probe / (np.linalg.norm(probe) + 1e-9)
        mat = np.stack(vecs, axis=0)
        mat = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-9)
        sims = mat @ probe

        order = np.argsort(-sims)[:limit]

        results = []
        for i in order:
            neighbour_id = photo_ids[int(i)]
            cur = await conn.execute("SELECT * FROM photos WHERE id = ?", (neighbour_id,))
            row = await cur.fetchone()
            if row is None:
                continue
            dto = photo_dto(dict(row))
            dto["score"] = float(sims[int(i)])
            results.append(PhotoResponse(**dto))

        return {"photo_id": photo_id, "results": results}

    # ------------------------------------------------------------------
    @router.get("/duplicates")
    async def list_duplicates(
        max_distance: int = Query(8, ge=0, le=64),
        current_user=Depends(current_user_dep),
        conn=Depends(db_dep),
    ):
        """Groups of near-identical photos by pHash Hamming distance.

        Distance 8 over the production library gives 242 pairs; 0 gives 15,
        4 gives 112, 12 gives 388. The default is deliberately loose enough
        to catch a re-save or a resize and tight enough to keep two different
        photos of the same scene apart.

        The scan is CPU-bound numpy — 0.18 s measured — and runs under
        `run_in_executor`. Inline it would stall the single uvicorn event
        loop for that whole time, so every other request waits on a
        housekeeping query.
        """
        async with conn.execute(
            "SELECT id, phash FROM photos WHERE phash IS NOT NULL AND phash != ''"
        ) as cur:
            rows = [(row["id"], row["phash"]) for row in await cur.fetchall()]

        loop = asyncio.get_running_loop()
        groups = await loop.run_in_executor(
            None, functools.partial(group_by_phash, rows, max_distance)
        )

        photos: Dict[int, Dict[str, Any]] = {}
        wanted = [pid for group in groups for pid in group]
        if wanted:
            placeholders = ",".join("?" * len(wanted))
            async with conn.execute(
                f"SELECT * FROM photos WHERE id IN ({placeholders})", wanted
            ) as cur:
                for row in await cur.fetchall():
                    photos[row["id"]] = photo_dto(dict(row))

        payload = []
        for group in groups:
            members = [PhotoResponse(**photos[pid]) for pid in group if pid in photos]
            if len(members) < 2:
                continue
            payload.append({"size": len(members), "photos": members})

        return {
            "max_distance": max_distance,
            "group_count": len(payload),
            "groups": payload,
        }

    return router
