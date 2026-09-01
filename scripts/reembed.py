#!/usr/bin/env python3
"""
Re-embed photos — resumably, and reading as little as possible off a dying disk.

Why this script exists
----------------------
Every photo needs to be re-embeddable on demand: a CLIP model swap invalidates
all 1,694 stored vectors at once, and there is currently no way to redo them
short of hand-enqueuing 1,778 RQ jobs. `process_photo_embedding_job` is also
the wrong tool for a bulk pass:

  * it downloads the **entire original** — a 4.6 MB HEIC or a 25 MB ARW — into
    memory, and CLIP then throws away everything above 224x224;
  * it reloads the 1.1 GB CLIP model in every forked work-horse, which is ~90%
    of the measured 58 s job;
  * `db.put_embedding` was a plain INSERT with no unique constraint, so a
    second pass would *duplicate* every vector rather than replace it. That is
    fixed in `core/db.py`; this script is what made it matter.

The disk under MinIO (`/dev/sda`) has 2,300+ unrecovered read errors and the
count is climbing, so read volume is the thing to minimise. Measured on this
data set:

    1,713 image originals   7,959 MB
    1,713 thumbnails          396 MB    (mean 237 KB, median 237 KB, max 462 KB)
                                        -> 20.1x less data off the bad disk

Thumbnails are 512 px JPEGs produced by `core.gallery.ensure_thumb` from the
same `load_image()` decode the embedder would do, and CLIP downsamples to
224 px regardless — so the thumbnail carries every pixel the model will look
at. Measured cosine between an original-derived and a thumbnail-derived vector
on a mixed sample was 0.987-0.999 (see the campaign notes); reading a
thumbnail also skips RAW/HEIC decode entirely (~0.10 s vs ~0.25 s per embed).

Design
------
* **Dry run by default.** Nothing is read, embedded or written without
  `--apply`. A dry run does not even create the journal.
* **Skip what is already done.** Without `--force` a photo that already has a
  vector *for this model* is not re-read — today that turns a 1,778 photo pass
  into an 84 photo one. The check keys on `embeddings.model`, so a SigLIP pass
  does not think a photo is done because it has a CLIP vector.
* **Durable resumability.** Every completed photo is appended to a JSONL
  journal and fsync'd before the next read starts, so Ctrl-C costs at most one
  photo. A restart skips everything the journal already claims, which is what
  makes `--force` survivable.
* **No retry loops.** A photo that failed to read is recorded and skipped on
  subsequent runs unless `--retry-failed` is passed. Bad sectors do not heal,
  and each retry costs more medium errors.
* **Disk-health guard.** The kernel medium-error count and
  `/sys/block/sda/device/ioerr_cnt` are sampled at the start and every
  `--check-every` photos. If either climbs by more than `--max-new-errors`
  the pass aborts cleanly, keeping everything already committed.
* **Model-aware.** `--model` sets the identifier written to
  `embeddings.model`, which is half that table's unique key
  `(photo_id, model)`. That is what lets a SigLIP row land *alongside* the CLIP
  row search is still answering from, making the cutover incremental and the
  rollback a config change rather than another full pass over a failing disk.
  It defaults to whatever the backend reports it has loaded, and is also
  recorded in the journal's run header.

Usage
-----
    set -a; . ./.env.production; set +a

    # what would it do?  (reads nothing, writes nothing)
    .venv/bin/python scripts/reembed.py

    # fill the gaps
    .venv/bin/python scripts/reembed.py --apply

    # a cautious first batch
    .venv/bin/python scripts/reembed.py --apply --limit 10 --delay 1.0

    # after a model change: write a second generation alongside the first
    .venv/bin/python scripts/reembed.py --apply --model google/siglip2-base

    # a bad-sector list from a previous run, retried on purpose
    .venv/bin/python scripts/reembed.py --apply --retry-failed
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import tempfile
import time
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Sequence, Set, Tuple

# Allow running as a plain script from the repo root as well as being imported
# as `scripts.reembed` by the test suite.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DB_DEFAULT_PATH = os.environ.get("CHITRA_DB_PATH", "photo.db")
DEFAULT_JOURNAL = "reembed_state.jsonl"

DEFAULT_MODEL = "openai/clip-vit-base-patch32"
# The measured CLIP optimum on this 6-core box. The curve is sharply
# non-monotonic: 113 ms/image at 3 threads, 194-220 ms at the 6-thread default.
DEFAULT_THREADS = 3

SOURCE_THUMB = "thumb"
SOURCE_ORIGINAL = "original"
SOURCE_AUTO = "auto"
SOURCE_MODES = (SOURCE_AUTO, SOURCE_THUMB, SOURCE_ORIGINAL)

ACTION_EMBED = "embed"
ACTION_SKIP = "skip"

BACKEND_SIDECAR = "sidecar"
BACKEND_INPROCESS = "inprocess"
BACKEND_AUTO = "auto"

KERN_LOG = "/var/log/kern.log"
KERN_PATTERN = "critical medium error"
IOERR_PATH = "/sys/block/sda/device/ioerr_cnt"

# Sampled per this many photos. 20 thumbnails is ~5 MB of reads, which is a
# fine granularity for noticing a disk that has started to cascade.
DEFAULT_CHECK_EVERY = 20
# One unreadable object costs ~30 medium errors in SCSI retries, so a handful
# of bad files must not abort the pass; a genuine cascade will blow past this.
DEFAULT_MAX_NEW_ERRORS = 100


# ----------------------------------------------------------------------
# DECISION LOGIC  (pure — this is what the tests pin)
# ----------------------------------------------------------------------
class Decision(NamedTuple):
    """What to do with one photo, and why."""

    photo_id: int
    action: str
    reason: str
    source: Optional[str] = None
    key: Optional[str] = None
    file_path: Optional[str] = None
    size: Optional[int] = None


def _row_get(row: Any, name: str, default=None):
    """Read a column that may not exist on a stubbed row."""
    try:
        value = row[name]
    except (IndexError, KeyError):
        return default
    return default if value is None else value


def classify_row(
    row: Any,
    *,
    source_mode: str,
    force: bool,
    done_ids: Set[int],
    failed_ids: Set[int],
    model: str,
    has_embedding: bool,
) -> Decision:
    """Decide what happens to one photo.

    Order matters. The journal is consulted *before* `force`, because the whole
    point of `--force` is a full pass that must still survive a Ctrl-C: a
    restart that ignored the journal would re-read everything it had already
    paid for. `--reset-state` is the deliberate way to start over.
    """
    photo_id = row["id"]
    file_path = _row_get(row, "file_path")
    thumb_path = _row_get(row, "thumb_path")
    media_type = _row_get(row, "media_type", "photo")
    size = _row_get(row, "size")

    if media_type == "video":
        # Videos get no ML anywhere else in the pipeline; a bulk pass must not
        # quietly disagree with the jobs.
        return Decision(photo_id, ACTION_SKIP, "video", None, None, file_path, size)

    if photo_id in done_ids:
        return Decision(photo_id, ACTION_SKIP, "already done in this journal",
                        None, None, file_path, size)

    if photo_id in failed_ids:
        return Decision(photo_id, ACTION_SKIP, "failed on a previous run (--retry-failed to re-read)",
                        None, None, file_path, size)

    if has_embedding and not force:
        return Decision(photo_id, ACTION_SKIP, f"has a {model} embedding already",
                        None, None, file_path, size)

    if source_mode == SOURCE_ORIGINAL:
        source, key = SOURCE_ORIGINAL, file_path
    elif thumb_path:
        source, key = SOURCE_THUMB, thumb_path
    elif source_mode == SOURCE_THUMB:
        return Decision(photo_id, ACTION_SKIP, "no thumbnail and --source thumb",
                        None, None, file_path, size)
    else:
        source, key = SOURCE_ORIGINAL, file_path

    if not key:
        return Decision(photo_id, ACTION_SKIP, "no object key on the row",
                        None, None, file_path, size)

    return Decision(photo_id, ACTION_EMBED, f"read from {source}", source, key, file_path, size)


def _has_model_column(conn: sqlite3.Connection) -> bool:
    """Has the `embeddings.model` migration run on this database yet?"""
    try:
        return any(row[1] == "model" for row in conn.execute("PRAGMA table_info(embeddings)"))
    except sqlite3.Error:
        return False


def plan(
    conn: sqlite3.Connection,
    *,
    source_mode: str,
    force: bool,
    done_ids: Set[int],
    failed_ids: Set[int],
    model: str,
    ids: Optional[Sequence[int]] = None,
    limit: Optional[int] = None,
) -> List[Decision]:
    """Classify every candidate row.

    `limit` caps *photos actually read*, not rows scanned — a cautious
    `--limit 10` has to mean ten reads off the failing disk, not ten rows of
    which nine are skips.
    """
    if _has_model_column(conn):
        sql = (
            "SELECT p.id, p.file_path, p.size, p.thumb_path, p.media_type, "
            "       (SELECT COUNT(*) FROM embeddings e "
            "          WHERE e.photo_id = p.id AND e.model = ?) AS emb_count "
            "  FROM photos p"
        )
        params: List[Any] = [model]
    else:
        # Pre-migration schema — the column arrives on the first `--apply`, and
        # a dry run must still answer honestly before that. Every row in an
        # unmigrated table came from the only model that has ever run here, so
        # "any embedding" is the correct reading of "has this model's
        # embedding" for the default model, and no match for any other.
        sql = (
            "SELECT p.id, p.file_path, p.size, p.thumb_path, p.media_type, "
            "       (SELECT COUNT(*) FROM embeddings e WHERE e.photo_id = p.id) * ? "
            "         AS emb_count "
            "  FROM photos p"
        )
        params = [1 if model == DEFAULT_MODEL else 0]
    if ids:
        sql += " WHERE p.id IN (%s)" % ",".join("?" * len(ids))
        params.extend(int(i) for i in ids)
    sql += " ORDER BY p.id"

    decisions: List[Decision] = []
    embeds = 0
    for row in conn.execute(sql, params):
        d = classify_row(
            row,
            source_mode=source_mode,
            force=force,
            done_ids=done_ids,
            failed_ids=failed_ids,
            model=model,
            has_embedding=bool(row["emb_count"]),
        )
        if d.action == ACTION_EMBED:
            if limit is not None and embeds >= limit:
                continue
            embeds += 1
        decisions.append(d)
    return decisions


# ----------------------------------------------------------------------
# JOURNAL  (durable progress)
# ----------------------------------------------------------------------
def load_journal(path: str, model: str) -> Tuple[Set[int], Dict[int, str]]:
    """Replay a journal into (done ids, {failed id: reason}).

    Records are scoped to the model that wrote them, so a model change starts
    from an empty slate without anyone having to remember to delete the file.
    A truncated final line — the normal shape of a Ctrl-C — is dropped, not
    treated as a corrupt journal.
    """
    done: Set[int] = set()
    failed: Dict[int, str] = {}
    if not os.path.exists(path):
        return done, failed

    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except ValueError:
                continue  # truncated tail from an interrupted write
            if rec.get("model") != model:
                continue
            pid = rec.get("photo_id")
            if not isinstance(pid, int):
                continue
            if rec.get("type") == "done":
                done.add(pid)
                failed.pop(pid, None)
            elif rec.get("type") == "fail":
                if pid not in done:
                    failed[pid] = str(rec.get("reason", ""))
    return done, failed


class Journal:
    """Append-only JSONL progress log, fsync'd per record.

    Durability matters more than throughput here: one fsync per photo is
    invisible next to a ~0.1-15 s read, and it is what makes Ctrl-C cost at
    most a single photo.
    """

    def __init__(self, path: str, model: str, source_mode: str, force: bool):
        self.path = path
        self.model = model
        self.source_mode = source_mode
        self.force = force
        self._fh = open(path, "a", encoding="utf-8")

    def _write(self, rec: dict) -> None:
        rec.setdefault("model", self.model)
        rec.setdefault("at", time.strftime("%Y-%m-%dT%H:%M:%S%z"))
        self._fh.write(json.dumps(rec, separators=(",", ":")) + "\n")
        self._fh.flush()
        os.fsync(self._fh.fileno())

    def record_run(self, planned: int, **extra) -> None:
        """The header record. This is where 'which model made these vectors'
        lives, since `embeddings` has no `model` column yet."""
        rec = {"type": "run", "source_mode": self.source_mode,
               "force": self.force, "planned": planned}
        rec.update(extra)
        self._write(rec)

    def record_done(self, photo_id: int, source: str, dim: int) -> None:
        self._write({"type": "done", "photo_id": photo_id, "source": source, "dim": dim})

    def record_fail(self, photo_id: int, source: Optional[str], reason: str) -> None:
        self._write({"type": "fail", "photo_id": photo_id, "source": source,
                     "reason": str(reason)[:300]})

    def record_abort(self, reason: str) -> None:
        self._write({"type": "abort", "reason": reason})

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:
            pass


# ----------------------------------------------------------------------
# DISK-HEALTH GUARD
# ----------------------------------------------------------------------
def read_disk_errors(kern_log: str = KERN_LOG, ioerr_path: str = IOERR_PATH) -> Dict[str, Optional[int]]:
    """Sample the two error counters.

    A source that cannot be read comes back as `None`, never 0 — a zero would
    look like a perfectly healthy disk and silently disable the guard.
    `kern.log` is mode 0640 root:adm, so this needs group `adm`.
    """
    counts: Dict[str, Optional[int]] = {"kern": None, "ioerr": None}

    try:
        n = 0
        with open(kern_log, "r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if KERN_PATTERN in line:
                    n += 1
        counts["kern"] = n
    except OSError:
        pass

    try:
        with open(ioerr_path, "r") as fh:
            counts["ioerr"] = int(fh.read().strip(), 0)  # the sysfs value is hex
    except (OSError, ValueError):
        pass

    return counts


class DiskGuard:
    """Aborts the pass if the disk's error counters climb during it.

    Both counters are watched and either one can trip the guard: `ioerr_cnt` is
    cheap and always present, while the kernel log distinguishes *unrecovered
    medium* errors specifically. A counter that reads `None` is ignored rather
    than treated as zero.
    """

    def __init__(self, threshold: int = DEFAULT_MAX_NEW_ERRORS,
                 reader: Callable[[], Dict[str, Optional[int]]] = read_disk_errors):
        self.threshold = threshold
        self.reader = reader
        self.baseline: Dict[str, Optional[int]] = {}
        self.latest: Dict[str, Optional[int]] = {}

    def start(self) -> Dict[str, Optional[int]]:
        self.baseline = self.reader()
        self.latest = dict(self.baseline)
        return self.baseline

    def deltas(self) -> Dict[str, Optional[int]]:
        out: Dict[str, Optional[int]] = {}
        for key, base in self.baseline.items():
            now = self.latest.get(key)
            out[key] = None if (base is None or now is None) else now - base
        return out

    def check(self) -> Optional[str]:
        """Sample again. Returns an abort reason, or None to carry on."""
        self.latest = self.reader()
        for key, delta in self.deltas().items():
            if delta is not None and delta > self.threshold:
                return (
                    f"{key} error count climbed by {delta} during this run "
                    f"(baseline {self.baseline.get(key)} -> {self.latest.get(key)}, "
                    f"threshold {self.threshold})"
                )
        return None


# ----------------------------------------------------------------------
# EMBEDDER BACKENDS
# ----------------------------------------------------------------------
class SidecarEmbedder:
    """Talks to the resident CLIP in `embed_service.py` over loopback.

    `POST /embed/image` was built for exactly this and had no caller. Using it
    avoids the 1.1 GB per-process model load *and* keeps this script's RSS in
    the tens of MB — which matters, because the box has been cgroup-OOM-killed
    six times and the sidecar's 1.7 GB is already resident.

    `rank_labels` is reproduced client-side: `ClipEmbedder.rank_labels` is just
    a dot product between the (already normalised) image vector and the
    normalised label vectors, and the sidecar hands out both. The 17 label
    vectors are embedded once and cached, so tagging costs one extra HTTP hop
    for the whole run rather than 17 per photo.
    """

    def __init__(self, base_url: Optional[str] = None, timeout: float = 300.0,
                 token: Optional[str] = None):
        import httpx

        self.base_url = (base_url or os.environ.get("CHITRA_EMBED_URL")
                         or "http://127.0.0.1:5101").rstrip("/")
        self.timeout = timeout
        self.token = token if token is not None else (os.environ.get("CHITRA_EMBED_TOKEN") or None)
        self._http = httpx.Client(timeout=timeout)
        self._label_cache: Dict[str, Any] = {}

        health = self._http.get(f"{self.base_url}/health", headers=self._headers()).json()
        if health.get("status") != "ok":
            raise RuntimeError(f"sidecar at {self.base_url} is not ready: {health}")
        self.name = health.get("model") or DEFAULT_MODEL
        # Probe for the real dimension rather than assuming 512: a model swap
        # is the main reason to run this script at all.
        self.dim = int(self._post_text("dimension probe")["dim"])

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self.token}"} if self.token else {}

    @staticmethod
    def _decode(payload: dict):
        import base64

        import numpy as np

        vec = np.frombuffer(base64.b64decode(payload["vector_b64"]), dtype="float32")
        return vec / (np.linalg.norm(vec) + 1e-9)

    def _post_text(self, text: str) -> dict:
        r = self._http.post(f"{self.base_url}/embed/text", json={"text": text},
                            headers=self._headers())
        r.raise_for_status()
        return r.json()

    def image_embedding(self, filename: str, data: bytes):
        # The filename matters: the sidecar keeps its suffix on the temp file
        # and `load_image` dispatches RAW vs PIL on the extension.
        r = self._http.post(
            f"{self.base_url}/embed/image",
            files={"file": (filename, data, "application/octet-stream")},
            headers=self._headers(),
        )
        r.raise_for_status()
        return self._decode(r.json())

    def _label_vectors(self, labels: Sequence[str]):
        import numpy as np

        for lab in labels:
            if lab not in self._label_cache:
                self._label_cache[lab] = self._decode(self._post_text(lab))
        return np.stack([self._label_cache[lab] for lab in labels])

    def rank_labels(self, filename: str, data: bytes, labels: Sequence[str], top_k: int = 6):
        image_vec = self.image_embedding(filename, data)
        return self.rank_labels_for_vector(image_vec, labels, top_k)

    def rank_labels_for_vector(self, image_vec, labels: Sequence[str], top_k: int = 6):
        """Same maths as `ClipEmbedder.rank_labels`, reusing a vector we already
        have so tagging does not cost a second forward pass over the image."""
        if not labels:
            return []
        sims = self._label_vectors(labels) @ image_vec
        order = sims.argsort()[::-1][:top_k]
        return [(labels[i], float(sims[i])) for i in order]

    def close(self) -> None:
        self._http.close()


class InProcessEmbedder:
    """Fallback when the sidecar is down: one CLIP load for the whole run.

    Still a single load — not the per-job reload the RQ path pays — but it
    costs ~1.1 GB of RSS on a box with a history of OOM kills, so the sidecar
    is preferred whenever it answers.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL, threads: int = DEFAULT_THREADS):
        import torch

        torch.set_num_threads(threads)
        from core.embedder import ClipEmbedder

        self.name = model_name
        self._embedder = ClipEmbedder(model_name)
        self.dim = int(self._embedder.text_embedding("dimension probe").shape[0])

    def _with_temp(self, filename: str, data: bytes, fn):
        suffix = os.path.splitext(filename or "")[1] or ".bin"
        fd, path = tempfile.mkstemp(suffix=suffix, prefix="chitra-reembed-")
        try:
            with os.fdopen(fd, "wb") as fh:
                fh.write(data)
            return fn(path)
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass

    def image_embedding(self, filename: str, data: bytes):
        return self._with_temp(filename, data, self._embedder.image_embedding)

    def rank_labels(self, filename: str, data: bytes, labels: Sequence[str], top_k: int = 6):
        return self._with_temp(
            filename, data,
            lambda p: self._embedder.rank_labels(p, list(labels), top_k=top_k),
        )

    def rank_labels_for_vector(self, image_vec, labels: Sequence[str], top_k: int = 6):
        return None  # no cheap path; run() falls back to rank_labels

    def close(self) -> None:
        self._embedder = None


def build_embedder(backend: str, model_name: str = DEFAULT_MODEL, threads: int = DEFAULT_THREADS):
    """Pick a backend. `auto` prefers the sidecar and says why if it cannot."""
    if backend in (BACKEND_AUTO, BACKEND_SIDECAR):
        try:
            return SidecarEmbedder()
        except Exception as exc:
            if backend == BACKEND_SIDECAR:
                raise
            print(f"[reembed] sidecar unavailable ({exc}); loading CLIP in-process", flush=True)
    return InProcessEmbedder(model_name, threads)


# ----------------------------------------------------------------------
# EXECUTION
# ----------------------------------------------------------------------
class Result(NamedTuple):
    planned: int
    embedded: int
    skipped: int
    failed: List[Tuple[int, str]]
    aborted: bool
    abort_reason: Optional[str]
    decisions: List[Decision]


def _iter_fetched(todo: Sequence[Decision], storage: Any, concurrency: int):
    """Yield `(decision, data_or_exception)` in plan order.

    At `--concurrency 1` — the default, and the right setting for a disk with
    2,300+ unrecovered read errors — this is a plain serial download. Above 1
    it becomes a bounded read-ahead: downloads overlap, embeds stay serial.
    That is the only parallelism worth having here, because a MinIO GET off the
    bad disk measured 3-17 s while a thumbnail embed is ~0.10 s, and the
    sidecar serialises forward passes anyway.
    """
    if concurrency <= 1:
        for d in todo:
            try:
                yield d, storage.download_file(d.key)
            except BaseException as exc:  # noqa: BLE001 - surfaced per photo by the caller
                yield d, exc
        return

    from concurrent.futures import ThreadPoolExecutor

    def fetch(d):
        try:
            return storage.download_file(d.key)
        except BaseException as exc:  # noqa: BLE001
            return exc

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        pending = []
        items = list(todo)
        cursor = 0
        while cursor < len(items) and len(pending) < concurrency:
            pending.append((items[cursor], pool.submit(fetch, items[cursor])))
            cursor += 1
        while pending:
            d, fut = pending.pop(0)
            yield d, fut.result()
            if cursor < len(items):
                pending.append((items[cursor], pool.submit(fetch, items[cursor])))
                cursor += 1


def _write_photo(conn, photo_id: int, vec, tag_pairs, model: str) -> int:
    """Persist one photo's vector (and tags) as an upsert, never an append.

    `model` is written into `embeddings.model`, which is half the unique key —
    so a SigLIP row lands *alongside* the CLIP row search is still using rather
    than replacing it.
    """
    from core import db

    vec_bytes = vec.astype("float32").tobytes()
    dim = int(vec.shape[0])
    db.put_embedding(conn, photo_id, vec_bytes, dim, model=model)
    if tag_pairs is not None:
        db.replace_tags(conn, photo_id, tag_pairs)
    return dim


def run(
    conn: sqlite3.Connection,
    storage: Any,
    embedder: Any,
    *,
    apply: bool = False,
    source_mode: str = SOURCE_AUTO,
    force: bool = False,
    tags: bool = False,
    delay: float = 0.0,
    limit: Optional[int] = None,
    ids: Optional[Sequence[int]] = None,
    journal_path: str = DEFAULT_JOURNAL,
    guard: Optional[DiskGuard] = None,
    check_every: int = DEFAULT_CHECK_EVERY,
    retry_failed: bool = False,
    reset_state: bool = False,
    concurrency: int = 1,
    model: Optional[str] = None,
    progress: Optional[Callable[[str], None]] = None,
) -> Result:
    """Run one pass. Reads and writes nothing unless `apply` is true."""
    say = progress or (lambda msg: print(msg, flush=True))
    # The identifier written to `embeddings.model` and the key everything else
    # is scoped by: the journal, the skip check, and the row itself. Defaults to
    # whatever the backend says it is loading.
    model = model or getattr(embedder, "name", DEFAULT_MODEL)

    if reset_state and os.path.exists(journal_path):
        if apply:
            os.unlink(journal_path)
        done_ids, failed_map = set(), {}
    else:
        done_ids, failed_map = load_journal(journal_path, model)
    failed_ids = set() if retry_failed else set(failed_map)

    decisions = plan(
        conn, source_mode=source_mode, force=force, done_ids=done_ids,
        failed_ids=failed_ids, model=model, ids=ids, limit=limit,
    )
    todo = [d for d in decisions if d.action == ACTION_EMBED]
    skipped = len(decisions) - len(todo)

    if not apply:
        return Result(len(todo), 0, skipped, [], False, None, decisions)

    # Additive and idempotent, and the same statements `init_db` runs at
    # startup. Done here because the running API and workers were launched
    # before the columns existed and cannot be restarted from this script — so
    # the first writer to actually need `embeddings.model` has to be the one
    # that puts it there. Deliberately inside the `apply` branch: a dry run
    # must not touch the schema.
    from core import db as _db

    _db.migrate_embeddings_and_tags(conn)

    guard = guard or DiskGuard()
    guard.start()

    journal = Journal(journal_path, model=model, source_mode=source_mode, force=force)
    journal.record_run(planned=len(todo), disk_baseline=guard.baseline,
                       tags=tags, limit=limit)

    embedded = 0
    failures: List[Tuple[int, str]] = []
    aborted = False
    abort_reason: Optional[str] = None

    try:
        for index, (d, fetched) in enumerate(_iter_fetched(todo, storage, concurrency), start=1):
            try:
                if isinstance(fetched, BaseException):
                    raise fetched
                data = fetched
                filename = os.path.basename(d.key)
                vec = embedder.image_embedding(filename, data)

                tag_pairs = None
                if tags:
                    # Imported here, not at module scope: `core.tagger` does a
                    # module-level `from core.embedder import ClipEmbedder`
                    # purely for a type annotation, which drags ~450 MB of
                    # torch/transformers into any process that touches it. With
                    # the sidecar backend this script needs none of that, so
                    # `--no-tags` keeps it entirely ML-free.
                    from core.tagger import DEFAULT_LABELS

                    ranked = None
                    ranker = getattr(embedder, "rank_labels_for_vector", None)
                    if ranker is not None:
                        ranked = ranker(vec, DEFAULT_LABELS, 6)
                    if ranked is None:
                        ranked = embedder.rank_labels(filename, data, DEFAULT_LABELS, 6)
                    tag_pairs = [(t, float(s)) for t, s in ranked]

                dim = _write_photo(conn, d.photo_id, vec, tag_pairs, model)
                journal.record_done(d.photo_id, source=d.source or "", dim=dim)
                embedded += 1
                say(f"[{index}/{len(todo)}] photo {d.photo_id} <- {d.source} "
                    f"{d.key} ({len(data) / 1024:.0f} KB) dim={dim}")
            except KeyboardInterrupt:
                raise
            except BaseException as exc:  # noqa: BLE001 - per-photo failure is non-fatal
                reason = f"{type(exc).__name__}: {exc}"
                failures.append((d.photo_id, reason))
                journal.record_fail(d.photo_id, d.source, reason)
                say(f"[{index}/{len(todo)}] photo {d.photo_id} FAILED ({d.source} "
                    f"{d.key}): {reason[:160]}")

            if check_every and index % check_every == 0:
                abort_reason = guard.check()
                if abort_reason:
                    aborted = True
                    journal.record_abort(abort_reason)
                    say(f"[reembed] ABORTING: {abort_reason}")
                    break

            if delay:
                time.sleep(delay)
    except KeyboardInterrupt:
        aborted = True
        abort_reason = "interrupted by the operator (Ctrl-C)"
        journal.record_abort(abort_reason)
        say("\n[reembed] interrupted; the journal is consistent — re-run to resume")
    finally:
        journal.close()

    return Result(len(todo), embedded, skipped, failures, aborted, abort_reason, decisions)


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def _positive_int(value: str) -> int:
    n = int(value)
    if n < 1:
        raise argparse.ArgumentTypeError(f"must be >= 1, got {value}")
    return n


def _id_list(value: str) -> List[int]:
    return [int(p) for p in value.replace(",", " ").split() if p.strip()]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="reembed",
        description="Re-embed photos from thumbnails, resumably, with a disk-health guard.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--db", default=DB_DEFAULT_PATH, help=f"SQLite path (default {DB_DEFAULT_PATH})")
    p.add_argument("--apply", action="store_true",
                   help="actually read, embed and write. Without it nothing happens at all.")
    p.add_argument("--force", action="store_true",
                   help="re-embed photos that already have a vector (what a model change needs)")
    p.add_argument("--source", choices=SOURCE_MODES, default=SOURCE_AUTO,
                   help="thumb: 512px JPEG only. original: always the full file. "
                        "auto (default): thumbnail when there is one, original otherwise.")
    p.add_argument("--tags", dest="tags", action="store_true", default=True,
                   help="also refresh the CLIP auto-tags (default)")
    p.add_argument("--no-tags", dest="tags", action="store_false",
                   help="embeddings only; leave the tags table alone")
    p.add_argument("--backend", choices=(BACKEND_AUTO, BACKEND_SIDECAR, BACKEND_INPROCESS),
                   default=BACKEND_AUTO,
                   help="where CLIP runs. auto (default) prefers the resident sidecar.")
    p.add_argument("--model", default=None,
                   help="identifier written to embeddings.model (half its unique key) and "
                        "used for the skip check and the journal. Defaults to whatever the "
                        "backend reports. Also selects the weights for --backend inprocess.")
    p.add_argument("--threads", type=_positive_int, default=DEFAULT_THREADS,
                   help=f"torch threads for the in-process backend (default {DEFAULT_THREADS}, measured optimum)")
    p.add_argument("--concurrency", type=_positive_int, default=1,
                   help="photos in flight. 1 (default) is deliberate: the disk is failing.")
    p.add_argument("--delay", type=float, default=0.25,
                   help="seconds to pause between photos (default 0.25)")
    p.add_argument("--limit", type=_positive_int, default=None,
                   help="stop after this many photos actually read")
    p.add_argument("--ids", type=_id_list, default=None, help="only these photo ids")
    p.add_argument("--state", default=DEFAULT_JOURNAL,
                   help=f"progress journal (default {DEFAULT_JOURNAL})")
    p.add_argument("--reset-state", action="store_true",
                   help="ignore and delete the journal; start the pass over")
    p.add_argument("--retry-failed", action="store_true",
                   help="re-read photos a previous run could not read (bad sectors: usually pointless)")
    p.add_argument("--max-new-errors", type=int, default=DEFAULT_MAX_NEW_ERRORS,
                   help=f"abort if a disk error counter climbs by more than this (default {DEFAULT_MAX_NEW_ERRORS})")
    p.add_argument("--check-every", type=int, default=DEFAULT_CHECK_EVERY,
                   help=f"sample the disk counters every N photos (default {DEFAULT_CHECK_EVERY}; 0 disables)")
    p.add_argument("--kern-log", default=KERN_LOG)
    p.add_argument("--ioerr-path", default=IOERR_PATH)
    return p


def _summarise_plan(decisions: List[Decision]) -> None:
    from collections import Counter

    todo = [d for d in decisions if d.action == ACTION_EMBED]
    skips = Counter(d.reason for d in decisions if d.action == ACTION_SKIP)
    by_source = Counter(d.source for d in todo)

    print("\nWould embed %d photo(s):" % len(todo))
    for source, n in by_source.most_common():
        print(f"    {n:5d} from {source}")
    thumb_bytes = sum(1 for d in todo if d.source == SOURCE_THUMB) * 237 * 1024
    orig_bytes = sum((d.size or 0) for d in todo if d.source == SOURCE_ORIGINAL)
    print(f"    est. read volume: {(thumb_bytes + orig_bytes) / 1024 / 1024:.0f} MB "
          f"({thumb_bytes / 1024 / 1024:.0f} MB thumbnails at the measured 237 KB mean, "
          f"{orig_bytes / 1024 / 1024:.0f} MB originals)")
    print("Would skip %d:" % sum(skips.values()))
    for reason, n in skips.most_common():
        print(f"    {n:5d} {reason}")
    if todo:
        preview = ", ".join(str(d.photo_id) for d in todo[:15])
        print(f"First ids: {preview}{' ...' if len(todo) > 15 else ''}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    from core import db

    conn = db.connect(args.db)
    embedder = None
    try:
        embedder = build_embedder(args.backend, args.model or DEFAULT_MODEL, args.threads)
        backend_model = getattr(embedder, "name", DEFAULT_MODEL)
        model = args.model or backend_model
        if args.model and args.model != backend_model:
            # Not fatal — staging a new model can legitimately want a different
            # label — but writing vectors under a name the backend did not
            # produce is exactly how a mixed table becomes unqueryable.
            print(f"[reembed] WARNING: --model {args.model!r} but the backend reports "
                  f"{backend_model!r}; rows will be recorded as {args.model!r}")
        print(f"[reembed] db={args.db} model={model} "
              f"dim={getattr(embedder, 'dim', '?')} backend={type(embedder).__name__} "
              f"source={args.source} force={args.force} tags={args.tags} "
              f"apply={args.apply}")

        guard = DiskGuard(
            threshold=args.max_new_errors,
            reader=lambda: read_disk_errors(args.kern_log, args.ioerr_path),
        )
        before = read_disk_errors(args.kern_log, args.ioerr_path)
        print(f"[reembed] disk errors before: {before}")

        result = run(
            conn, None if not args.apply else _storage(), embedder,
            apply=args.apply, source_mode=args.source, force=args.force,
            tags=args.tags, delay=args.delay, limit=args.limit, ids=args.ids,
            journal_path=args.state, guard=guard, check_every=args.check_every,
            retry_failed=args.retry_failed, reset_state=args.reset_state,
            concurrency=args.concurrency, model=model,
        )

        if not args.apply:
            _summarise_plan(result.decisions)
            print("\nDRY RUN — nothing was read, embedded or written. Re-run with --apply.")
            return 0

        after = read_disk_errors(args.kern_log, args.ioerr_path)
        print("\n--- summary ---")
        print(f"planned  {result.planned}")
        print(f"embedded {result.embedded}")
        print(f"skipped  {result.skipped}")
        print(f"failed   {len(result.failed)}")
        print(f"disk errors before {before} -> after {after}")
        if result.aborted:
            print(f"ABORTED: {result.abort_reason}")
        if result.failed:
            print("\nUnreadable / unembeddable photos (likely on bad sectors):")
            for pid, reason in result.failed:
                print(f"    {pid}: {reason[:180]}")
            print("\n  Re-run with --retry-failed to attempt these again.")
        return 1 if result.aborted else 0
    finally:
        if embedder is not None and hasattr(embedder, "close"):
            embedder.close()
        conn.close()


def _storage():
    from core.storage_client import MinIOStorageClient

    return MinIOStorageClient()


if __name__ == "__main__":
    sys.exit(main())
