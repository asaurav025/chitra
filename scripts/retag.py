#!/usr/bin/env python3
"""
Re-tag the whole library from the vectors already in SQLite — reading no media.

Why this script exists
----------------------
Tagging shipped as 17 hardcoded labels and a fixed top-6. Measured on
production (1,909 tagged photos, 11,456 rows):

    travel      1,554 photos  81.4%      every photo has exactly 6.0 tags
    outdoors      903         47.3%      raw cosine spans 0.158 - 0.278
    portrait      868         45.5%      best-to-worst-kept gap: ~0.03

`travel` is not a fact about the library. It is what "least unlike this photo,
out of 17 strings" returns when 17 strings do not cover the space, and top-k
hands back k labels whether any of them fit or not.

Why it reads nothing
--------------------
A CLIP tag score is `cosine(image_vec, text_vec)`, and the blobs in
`embeddings.vector` **are** the L2-normalised image vectors (measured: norm
min/max/mean 1.000000 across all 1,910 rows, every one dim=512). So re-tagging
the entire library is one `1910 x 512 @ 512 x 345` GEMM — a few hundred MFLOP
over 3.7 MiB of SQLite. The label vectors cost 345 calls to the sidecar's
`/embed/text` at ~13 ms, once, and are then cached to a `.npy` on the NVMe.

`/dev/sda` — which holds every original, thumbnail, poster and face crop, and
has 3,000+ unrecovered read errors — is touched **zero** times. That is not a
hope; `tests/test_retag.py::TestRetagReadsNoStorage` runs the real pass with a
`StorageClient` whose every attribute raises.

Calibration
-----------
See the module docstring of `core/tagger.py` for the reasoning and, more
importantly, for the honest limit. In short: thresholds are learned per label
from the corpus, so a tag means "unusually beach-like *for this library*" rather
than "depicts a beach". That is a workaround for CLIP's contrastive objective,
not a fix for it.

Usage
-----
    # dry run — the default; prints the projected distribution, writes nothing
    .venv/bin/python scripts/retag.py --db /tmp/retag_test.db

    # write
    .venv/bin/python scripts/retag.py --db photo.db --apply

Rollback is one statement, which is what `tags.source` is for:

    DELETE FROM tags WHERE source = 'clip-vitb32/vocab-v2';
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import db, tagger, vocabulary  # noqa: E402
# The label-matrix cache moved into `core.tagger` so `core/jobs.py` can read it
# without importing a script; re-exported here because it is still this
# script's cache and `retag.load_cached_matrix` is how it is addressed.
from core.tagger import (  # noqa: E402
    CacheMismatch,
    calibration_path,
    label_matrix_path,
    load_cached_matrix,
    save_cached_matrix,
    save_calibration,
)

DEFAULT_CACHE_DIR = tagger.DEFAULT_CACHE_DIR
DEFAULT_EMBED_URL = "http://127.0.0.1:5101"
DEFAULT_RETRIES = 4
DEFAULT_BACKOFF = 0.5

# Sources this script considers its own and will clear before writing. Anything
# else — a hand-added tag, a future manual-curation feature — is left alone.
# `add_tag` in a loop cannot *remove* a label the new run no longer predicts, so
# without this delete `travel` would sit on 100% of the library forever and the
# whole pass would be cosmetic.
OWNED_SOURCE_LIKE = "%/vocab-%"


# ----------------------------------------------------------------------
# TEXT EMBEDDING (the only network the script does, and only on a cache miss)
# ----------------------------------------------------------------------
class SidecarTextEmbedder:
    """Talks to the resident CLIP in `embed_service.py` over loopback.

    Deliberately *not* an in-process `ClipEmbedder`: the sidecar already holds
    the 1.1 GB model, and this script's whole point is to stay in the tens of
    MB. If the sidecar is down this raises rather than falling back — a silent
    fallback would load torch into a script that advertises not needing it.
    """

    def __init__(self, base_url: Optional[str] = None, timeout: float = 60.0,
                 token: Optional[str] = None, client=None,
                 retries: int = DEFAULT_RETRIES, backoff: float = DEFAULT_BACKOFF,
                 progress_every: int = 50):
        import httpx  # local: keeps `import scripts.retag` cheap

        self.base_url = (base_url or os.environ.get("CHITRA_EMBED_URL")
                         or DEFAULT_EMBED_URL).rstrip("/")
        self.token = token if token is not None else (os.environ.get("CHITRA_EMBED_TOKEN") or None)
        self.retries = max(1, int(retries))
        self.backoff = float(backoff)
        self.progress_every = int(progress_every)
        self._http = client if client is not None else httpx.Client(timeout=timeout)
        health = self._http.get(f"{self.base_url}/health", headers=self._headers()).json()
        if health.get("status") != "ok":
            raise RuntimeError(f"sidecar at {self.base_url} is not ready: {health}")
        self.name = health.get("model") or db.DEFAULT_EMBED_MODEL

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self.token}"} if self.token else {}

    def _post_text(self, text: str):
        """One prompt, retried on a transport error.

        The sidecar is a single-worker uvicorn shared with the API and the RQ
        workers. 345 sequential requests through it *will* meet a dropped
        connection — observed as `httpx.ReadError: [Errno 104] Connection reset
        by peer` 10 minutes into a real run — and throwing away 300 completed
        embeds because of one reset defeats the point of caching the matrix.

        Only transport errors are retried. A 4xx/5xx is a real answer from the
        service and is re-raised: retrying a 422 just asks the same wrong
        question four more times.
        """
        import base64

        import httpx

        last: Optional[Exception] = None
        for attempt in range(self.retries):
            try:
                r = self._http.post(f"{self.base_url}/embed/text",
                                    json={"text": text}, headers=self._headers())
                r.raise_for_status()
                vec = np.frombuffer(base64.b64decode(r.json()["vector_b64"]),
                                    dtype="float32")
                return vec / (np.linalg.norm(vec) + 1e-9)
            except httpx.HTTPStatusError:
                raise
            except (httpx.TransportError, httpx.HTTPError) as exc:
                last = exc
                if attempt + 1 < self.retries and self.backoff:
                    time.sleep(self.backoff * (2 ** attempt))
        raise last  # type: ignore[misc]

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        texts = list(texts)
        out = []
        for i, text in enumerate(texts, 1):
            out.append(self._post_text(text))
            if self.progress_every and i % self.progress_every == 0:
                print(f"  ...{i}/{len(texts)} label vectors", flush=True)
        return np.stack(out).astype("float32")


# ----------------------------------------------------------------------
# LABEL MATRIX + ITS CACHE
# ----------------------------------------------------------------------
def get_label_matrix(embedder, *, model: str, cache_dir=DEFAULT_CACHE_DIR,
                     labels: Optional[Sequence[str]] = None,
                     refresh: bool = False, verbose: bool = True) -> np.ndarray:
    labels = tuple(vocabulary.LABELS if labels is None else labels)
    fingerprint = vocabulary.vocab_fingerprint(labels=labels)
    path = label_matrix_path(cache_dir, model, fingerprint)

    if not refresh:
        cached = load_cached_matrix(path, model=model, fingerprint=fingerprint,
                                    labels=labels)
        if cached is not None:
            if verbose:
                print(f"label matrix: cached {path} {cached.shape}")
            return cached

    if embedder is None:
        raise RuntimeError(
            f"no cached label matrix at {path} and no text embedder available — "
            "start the sidecar (embed_service.py on :5101) or pass --vectors"
        )
    t0 = time.time()
    prompts = vocabulary.prompts(labels)
    matrix = np.asarray(embedder.embed_texts(prompts), dtype="float32")
    if matrix.shape[0] != len(labels):
        raise RuntimeError(
            f"text embedder returned {matrix.shape[0]} vectors for {len(labels)} prompts")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    matrix = matrix / np.where(norms < 1e-9, 1.0, norms)
    save_cached_matrix(path, matrix, model=model, fingerprint=fingerprint,
                       labels=labels)
    if verbose:
        print(f"label matrix: embedded {len(labels)} prompts in "
              f"{time.time() - t0:.1f}s -> {path}")
    return matrix


# ----------------------------------------------------------------------
# THE CORPUS
# ----------------------------------------------------------------------
def load_vectors(conn: sqlite3.Connection, model: str,
                 include_videos: bool = False) -> Tuple[List[int], np.ndarray]:
    """Every stored image vector for one model, as (photo_ids, N x dim).

    Filtered to a single model on purpose. `embeddings` is unique on
    `(photo_id, model)` precisely so a 768-d SigLIP row can sit next to a 512-d
    CLIP one — and stacking both is the `ValueError: all input arrays must have
    the same shape` that takes `/api/search/photos` down for every user.

    Videos are excluded by default via `COALESCE(media_type,'photo')` — 766 rows
    have a NULL media_type and are photos. That default is a phase boundary, not
    a judgement: poster embedding landed in 2b1b052, so videos will start having
    vectors, and tagging their posters is Phase 7's cutover to make. It is a
    flag rather than a buried WHERE clause so that flip is one argument, and so
    a video that is embedded but untagged has a visible reason.
    """
    rows = conn.execute(
        f"""
        SELECT e.photo_id AS photo_id, e.dim AS dim, e.vector AS vector
          FROM embeddings e
          JOIN photos p ON p.id = e.photo_id
         WHERE e.model = ?
           {"" if include_videos else "AND COALESCE(p.media_type, 'photo') != 'video'"}
         ORDER BY e.photo_id ASC
        """,
        (model,),
    ).fetchall()
    if not rows:
        return [], np.zeros((0, 0), dtype="float32")

    dims = {int(r["dim"]) for r in rows}
    if len(dims) != 1:
        raise ValueError(
            f"model {model!r} has rows of mixed dimension {sorted(dims)} — "
            "refusing to stack them"
        )
    dim = dims.pop()

    ids: List[int] = []
    vecs = np.empty((len(rows), dim), dtype="float32")
    for i, r in enumerate(rows):
        blob = r["vector"]
        if len(blob) != dim * 4:
            # A truncated blob reshapes silently into a plausible matrix and
            # every downstream number is then wrong with no error.
            raise ValueError(
                f"photo {r['photo_id']}: vector is {len(blob)} bytes, "
                f"expected {dim * 4} for dim={dim}"
            )
        vecs[i] = np.frombuffer(blob, dtype="float32")
        ids.append(int(r["photo_id"]))

    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    vecs = vecs / np.where(norms < 1e-9, 1.0, norms)
    return ids, vecs


def current_distribution(conn: sqlite3.Connection) -> Dict[str, int]:
    return {r["tag"]: r["n"] for r in conn.execute(
        "SELECT tag, COUNT(*) AS n FROM tags GROUP BY tag ORDER BY n DESC")}


# ----------------------------------------------------------------------
# THE PASS
# ----------------------------------------------------------------------
@dataclass
class RetagResult:
    model: str
    source: str
    fingerprint: str
    n_labels: int
    dim: int
    photos: int
    corpus: int
    applied: bool
    tags_written: int = 0
    tag_counts: List[int] = field(default_factory=list)
    distribution: Dict[str, int] = field(default_factory=dict)
    before: Dict[str, int] = field(default_factory=dict)
    seconds: float = 0.0
    low_percentile: float = tagger.LOW_PERCENTILE
    high_percentile: float = tagger.HIGH_PERCENTILE


def retag(
    conn: sqlite3.Connection,
    *,
    embedder=None,
    model: Optional[str] = None,
    cache_dir=DEFAULT_CACHE_DIR,
    label_matrix: Optional[np.ndarray] = None,
    labels: Optional[Sequence[str]] = None,
    apply: bool = False,
    limit: Optional[int] = None,
    min_tags: int = tagger.MIN_TAGS_PER_PHOTO,
    max_tags: int = tagger.MAX_TAGS_PER_PHOTO,
    low_percentile: float = tagger.LOW_PERCENTILE,
    high_percentile: float = tagger.HIGH_PERCENTILE,
    refresh_cache: bool = False,
    include_videos: bool = False,
    verbose: bool = False,
) -> RetagResult:
    """Score every stored vector against the vocabulary and (optionally) write.

    Reads: `embeddings`, `photos`, `tags` — SQLite only. Writes: `tags`, and
    only under `apply=True`.
    """
    t0 = time.time()
    model = model or getattr(embedder, "name", None) or db.DEFAULT_EMBED_MODEL
    labels = tuple(vocabulary.LABELS if labels is None else labels)
    source = vocabulary.tag_source(model)
    fingerprint = vocabulary.vocab_fingerprint(labels=labels)

    ids, vecs = load_vectors(conn, model, include_videos=include_videos)
    before = current_distribution(conn)
    if not ids:
        return RetagResult(model=model, source=source, fingerprint=fingerprint,
                           n_labels=len(labels), dim=0, photos=0, corpus=0,
                           applied=False, before=before,
                           seconds=time.time() - t0,
                           low_percentile=low_percentile,
                           high_percentile=high_percentile)

    if label_matrix is None:
        label_matrix = get_label_matrix(embedder, model=model, cache_dir=cache_dir,
                                        labels=labels, refresh=refresh_cache,
                                        verbose=verbose)
    label_matrix = np.asarray(label_matrix, dtype="float32")
    if label_matrix.shape[1] != vecs.shape[1]:
        raise ValueError(
            f"label vectors are dim {label_matrix.shape[1]} but the stored image "
            f"vectors are dim {vecs.shape[1]} — wrong model for this corpus"
        )

    # The whole compute budget of this script. N x dim @ dim x n_labels.
    scores = (vecs @ label_matrix.T).astype("float32")

    # Calibrate on the FULL corpus even when --limit is set: thresholds learned
    # from 5 photos are noise, and the limit is for inspecting a sample of the
    # output, not for changing what the output means.
    calibration = tagger.calibrate(scores, labels,
                                   low_percentile=low_percentile,
                                   high_percentile=high_percentile)

    # Leave the thresholds where the per-photo upload job can find them. Only
    # on an apply: a dry run's calibration describes tags nothing was written
    # with, and the job would then be scoring against a corpus state the `tags`
    # table never reflected. Written before the loop so a pass interrupted
    # part-way still leaves an artifact matching the rows it did write.
    if apply:
        save_calibration(
            calibration_path(cache_dir, model, fingerprint),
            calibration, model=model, fingerprint=fingerprint,
        )

    selected = range(len(ids)) if not limit else range(min(limit, len(ids)))
    result = RetagResult(
        model=model, source=source, fingerprint=fingerprint,
        n_labels=len(labels), dim=int(vecs.shape[1]),
        photos=len(selected), corpus=len(ids), applied=bool(apply),
        before=before, low_percentile=low_percentile,
        high_percentile=high_percentile,
    )

    for i in selected:
        chosen = tagger.tags_from_scores(scores[i], labels, calibration,
                                         min_tags=min_tags, max_tags=max_tags)
        result.tag_counts.append(len(chosen))
        for lab, _score in chosen:
            result.distribution[lab] = result.distribution.get(lab, 0) + 1
        if apply:
            _write_tags(conn, ids[i], chosen, source)
            result.tags_written += len(chosen)

    result.seconds = time.time() - t0
    return result


def _write_tags(conn: sqlite3.Connection, photo_id: int,
                chosen: Sequence[Tuple[str, float]], source: str) -> None:
    """Clear this script's own rows for the photo, then upsert the new set.

    The DELETE is scoped by `source`, not by photo: a tag added by hand (or by
    any future curation feature) does not carry a `.../vocab-*` source and is
    not ours to remove. The INSERT goes through `db.add_tag`, which is an upsert
    on `(photo_id, tag)`, so a re-run is idempotent and a partially completed
    run is safe to repeat.
    """
    conn.execute(
        "DELETE FROM tags WHERE photo_id = ? AND (source IS NULL OR source LIKE ?)",
        (photo_id, OWNED_SOURCE_LIKE),
    )
    for lab, score in chosen:
        db.add_tag(conn, photo_id, lab, float(score), source=source)
    conn.commit()


# ----------------------------------------------------------------------
# REPORTING
# ----------------------------------------------------------------------
def format_report(result: RetagResult, top: int = 30) -> str:
    out: List[str] = []
    a = out.append
    a("")
    a("=" * 74)
    a(f"{'APPLIED' if result.applied else 'DRY RUN'} — retag from stored vectors")
    a("=" * 74)
    a(f"model          {result.model}")
    a(f"source stamp   {result.source}")
    a(f"vocabulary     {result.n_labels} labels, {vocabulary.VOCAB_VERSION}, "
      f"fingerprint {result.fingerprint}")
    a(f"corpus         {result.corpus} vectors, dim {result.dim}")
    a(f"tagged         {result.photos} photos")
    a(f"calibration    per-label p{result.low_percentile:g} (floor) / "
      f"p{result.high_percentile:g} (keep)")
    a(f"elapsed        {result.seconds:.2f}s   MinIO reads: 0")
    if result.applied:
        a(f"rows written   {result.tags_written}")

    if not result.tag_counts:
        return "\n".join(out)

    counts = np.asarray(result.tag_counts)
    a("")
    a(f"tags per photo   min {counts.min()}  median {np.median(counts):.1f}  "
      f"mean {counts.mean():.2f}  max {counts.max()}")
    hist = {int(c): int((counts == c).sum()) for c in np.unique(counts)}
    a("                 " + "  ".join(
        f"{k}:{v}" for k, v in sorted(hist.items())))
    if len(hist) == 1:
        a("  !! every photo got the same number of tags — that is the old bug")

    n = float(result.photos)
    a("")
    a(f"projected distribution (top {top} of {len(result.distribution)} used labels)")
    a(f"  {'label':<26} {'facet':<12} {'after':>7} {'share':>7}   {'before':>7} {'share':>7}")
    for lab, cnt in sorted(result.distribution.items(), key=lambda kv: -kv[1])[:top]:
        b = result.before.get(lab, 0)
        bshare = f"{b / result.corpus:6.1%}" if result.corpus else "     -"
        a(f"  {lab:<26} {vocabulary.facet_of(lab):<12} {cnt:>7} {cnt / n:6.1%}   "
          f"{b:>7} {bshare}")

    a("")
    a("facet coverage (tag rows by facet)")
    per_facet: Dict[str, int] = {}
    for lab, cnt in result.distribution.items():
        f = vocabulary.facet_of(lab)
        per_facet[f] = per_facet.get(f, 0) + cnt
    total = sum(per_facet.values()) or 1
    for f, cnt in sorted(per_facet.items(), key=lambda kv: -kv[1]):
        a(f"  {f:<14} {cnt:>7} {cnt / total:6.1%}")

    if result.before:
        a("")
        a("legacy labels, before -> after")
        a(f"  {'label':<26} {'before':>7} {'share':>7}   {'after':>7} {'share':>7}")
        for lab in sorted(result.before, key=lambda k: -result.before[k]):
            b = result.before[lab]
            after = result.distribution.get(lab, 0)
            a(f"  {lab:<26} {b:>7} {b / result.corpus:6.1%}   "
              f"{after:>7} {after / n:6.1%}")
    return "\n".join(out)


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="retag.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--db", default=db.DB_DEFAULT_PATH,
                   help=f"SQLite path (default {db.DB_DEFAULT_PATH})")
    p.add_argument("--apply", action="store_true",
                   help="write the tags. Without it nothing is written and the "
                        "database is opened read-only.")
    p.add_argument("--model", default=None,
                   help="which embeddings.model to score (default: whatever the "
                        "sidecar reports it has loaded)")
    p.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR,
                   help=f"where the label matrix is cached (default {DEFAULT_CACHE_DIR}/) "
                        "— on the NVMe, never MinIO")
    p.add_argument("--include-videos", action="store_true",
                   help="also tag videos that have a poster-derived vector "
                        "(Phase 7; off while photos and videos are cut over "
                        "separately)")
    p.add_argument("--refresh-cache", action="store_true",
                   help="re-embed the label prompts even if a valid cache exists")
    p.add_argument("--limit", type=int, default=None,
                   help="tag only the first N photos (calibration still uses the "
                        "whole corpus)")
    p.add_argument("--min-tags", type=int, default=tagger.MIN_TAGS_PER_PHOTO)
    p.add_argument("--max-tags", type=int, default=tagger.MAX_TAGS_PER_PHOTO)
    p.add_argument("--low-percentile", type=float, default=tagger.LOW_PERCENTILE,
                   help="a label below its own p<LOW> is never tagged")
    p.add_argument("--high-percentile", type=float, default=tagger.HIGH_PERCENTILE,
                   help="a label above its own p<HIGH> is kept on merit")
    p.add_argument("--top", type=int, default=30, help="labels to print")
    p.add_argument("--embed-url", default=None, help="sidecar base URL")
    p.add_argument("--json", action="store_true", help="emit the result as JSON too")
    return p


def _connect(path: str, read_only: bool) -> sqlite3.Connection:
    if not read_only:
        return db.connect(path)
    # A dry run should be unable to write even if the code were wrong.
    conn = sqlite3.connect(f"file:{Path(path).resolve()}?mode=ro", uri=True,
                           timeout=30.0)
    conn.row_factory = sqlite3.Row
    return conn


def main(argv: Optional[Sequence[str]] = None, embedder=None) -> int:
    args = build_parser().parse_args(argv)

    if embedder is None:
        try:
            embedder = SidecarTextEmbedder(base_url=args.embed_url)
        except Exception as exc:  # noqa: BLE001 - the message is the point
            # Deliberately no in-process fallback: loading CLIP here would put
            # 1.7 GB into a script whose entire claim is that it needs neither a
            # model nor a media read.
            print(f"embedding sidecar unavailable: {exc}", file=sys.stderr)
            print("start embed_service.py on :5101, or pass a cached label "
                  "matrix via --cache-dir", file=sys.stderr)
            embedder = None

    conn = _connect(args.db, read_only=not args.apply)
    try:
        result = retag(
            conn,
            embedder=embedder,
            model=args.model,
            cache_dir=args.cache_dir,
            apply=args.apply,
            limit=args.limit,
            min_tags=args.min_tags,
            max_tags=args.max_tags,
            low_percentile=args.low_percentile,
            high_percentile=args.high_percentile,
            refresh_cache=args.refresh_cache,
            include_videos=args.include_videos,
            verbose=True,
        )
    finally:
        conn.close()

    print(format_report(result, top=args.top))
    if args.json:
        print(json.dumps({
            "model": result.model, "source": result.source,
            "fingerprint": result.fingerprint, "labels": result.n_labels,
            "corpus": result.corpus, "photos": result.photos,
            "applied": result.applied, "tags_written": result.tags_written,
            "distribution": result.distribution,
        }, indent=2))
    if not result.applied:
        print("\nDry run. Nothing was written. Re-run with --apply.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
