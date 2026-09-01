"""
FAISS index management with HNSW and persistence.

Three properties this module is responsible for, each of which used to be
missing and each of which failed silently rather than loudly:

**Location.** ``index_dir`` was ``"faiss_indexes"`` — relative, resolved
against the process CWD. It worked only because ``start_workers.sh`` cds into
the repo first. A systemd unit without a matching ``WorkingDirectory`` created
a second, empty index directory and face matching stopped working with no
error at all. The directory now comes from ``CHITRA_FAISS_INDEX_DIR`` or from
the module's own location, is always absolute, and is logged once per process
so a misconfiguration is visible.

**Atomicity.** ``save_index`` called ``faiss.write_index`` straight onto the
final path. A 2.4 MB index takes long enough to write that a concurrent reader
routinely got a truncated file: the regression test measured 387 failed reads
during 12 rewrites. Writes now go to a temp file in the same directory,
followed by ``os.replace`` — atomic within a filesystem, so a reader sees
either the whole old index or the whole new one.

**Serialisation.** ``update_index`` is a read-modify-write reachable from four
concurrent default-queue workers, with no lock: four processes adding 40
vectors each left 80 of 160 in the file. Every mutation now runs under an
``fcntl.flock`` on a sidecar lock file (plus a per-key ``RLock`` so threads in
one process serialise too). Reads deliberately take no lock — that is what the
atomic replace buys, and the per-face match path depends on cheap reads.

**Metric.** ``build_hnsw_index`` L2-normalises its input, so inner product *is*
cosine similarity. Indexes are therefore built with
``METRIC_INNER_PRODUCT`` and ``search`` returns cosine similarity directly,
comparable against ``FACE_MATCH_THRESHOLD`` with no conversion. The previous
default (``METRIC_L2``) returned *squared* L2 distances that callers tried to
convert back with a batch-wide "if the max looks big, it must be L2" guess —
which silently mis-scored exactly the good matches it was meant to find.
"""
import fcntl
import logging
import os
import tempfile
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Optional, List, Sequence, Tuple

import faiss
import numpy as np

logger = logging.getLogger(__name__)

#: Operator override for where indexes live. Absolute or relative; a relative
#: value is resolved against the CWD once, at construction.
ENV_INDEX_DIR = "CHITRA_FAISS_INDEX_DIR"

#: Default: beside the repo that contains this module, never the CWD.
DEFAULT_INDEX_DIR = Path(__file__).resolve().parent.parent / "faiss_indexes"

# Directories already announced by this process, so the resolved path is
# logged once and not once per detected face.
_logged_dirs = set()
_logged_dirs_guard = threading.Lock()


def _reset_logged_dirs_for_tests():
    """Test hook: forget which directories have been logged."""
    with _logged_dirs_guard:
        _logged_dirs.clear()


def resolve_index_dir(index_dir: Optional[str] = None) -> Path:
    """
    Resolve the index directory to an absolute path.

    Precedence: explicit argument, then ``CHITRA_FAISS_INDEX_DIR``, then a
    path derived from this module's location. Never the bare CWD.
    """
    if index_dir is None:
        index_dir = os.environ.get(ENV_INDEX_DIR) or DEFAULT_INDEX_DIR
    return Path(index_dir).expanduser().resolve()


class _IndexLock:
    """
    Cross-process (flock) and cross-thread (RLock) lock for one index name.

    Reentrant within a thread: ``update_index`` holds it across a
    load-modify-save, and the ``save_index`` nested inside must not deadlock on
    a second ``flock`` of the same file (flock is per open file description, so
    a naive second ``open`` + ``LOCK_EX`` in the same process would hang).
    """

    def __init__(self, path: str):
        self.path = path
        self._thread_lock = threading.RLock()
        self._depth = 0
        self._fh = None

    def __enter__(self):
        self._thread_lock.acquire()
        try:
            if self._depth == 0:
                self._fh = open(self.path, "a+")
                fcntl.flock(self._fh.fileno(), fcntl.LOCK_EX)
            self._depth += 1
        except BaseException:
            self._thread_lock.release()
            raise
        return self

    def __exit__(self, *exc_info):
        try:
            self._depth -= 1
            if self._depth == 0 and self._fh is not None:
                try:
                    fcntl.flock(self._fh.fileno(), fcntl.LOCK_UN)
                finally:
                    self._fh.close()
                    self._fh = None
        finally:
            self._thread_lock.release()
        return False


_locks = {}
_locks_guard = threading.Lock()


def _lock_for(path) -> _IndexLock:
    key = str(path)
    with _locks_guard:
        lock = _locks.get(key)
        if lock is None:
            lock = _IndexLock(key)
            _locks[key] = lock
        return lock


def is_id_mapped(index: Optional[faiss.Index]) -> bool:
    """
    True if the index carries its own vector-position -> caller-id mapping.

    Without one, callers have to guess that index position *i* corresponds to
    row *i* of some later database query. That guess is what broke face
    matching: positions drift as soon as anything appends to the index, and the
    live index was found mis-attributing 715 of 1,044 positions (68.5%) to the
    wrong person while still passing a count check.
    """
    if index is None:
        return False
    return hasattr(index, "id_map")


def index_ids(index: faiss.Index) -> List[int]:
    """The stored ids of an ID-mapped index, in position order."""
    if not is_id_mapped(index):
        return []
    return faiss.vector_to_array(index.id_map).tolist()


class FAISSIndexManager:
    """Manages FAISS indexes with HNSW and persistence."""

    def __init__(self, index_dir: Optional[str] = None):
        """
        Initialize index manager.

        Args:
            index_dir: Directory to store persistent indexes. Defaults to
                       ``CHITRA_FAISS_INDEX_DIR``, else a path derived from
                       this module's location. Always resolved to an absolute
                       path — see the module docstring for why.
        """
        self.index_dir = resolve_index_dir(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)

        # Log once per process per directory: a wrong directory is otherwise
        # completely silent, and this path is the whole ballgame for matching.
        key = str(self.index_dir)
        with _logged_dirs_guard:
            first_time = key not in _logged_dirs
            if first_time:
                _logged_dirs.add(key)
        if first_time:
            source = (
                f"{ENV_INDEX_DIR}"
                if os.environ.get(ENV_INDEX_DIR) and index_dir is None
                else ("argument" if index_dir is not None else "module default")
            )
            logger.info("FAISS index directory: %s (from %s)", key, source)

    def get_index_path(self, index_name: str) -> Path:
        """Get path for index file."""
        return self.index_dir / f"{index_name}.index"

    def _get_lock_path(self, index_name: str) -> Path:
        return self.index_dir / f".{index_name}.lock"

    @contextmanager
    def index_lock(self, index_name: str):
        """
        Serialise mutations of one index across processes and threads.

        Public because a caller doing its own read-modify-write (load, append,
        save) needs to hold the lock across the whole sequence, not just the
        save.
        """
        with _lock_for(self._get_lock_path(index_name)):
            yield

    # ------------------------------------------------------------------
    # BUILD / ADD
    # ------------------------------------------------------------------

    def build_hnsw_index(
        self,
        vectors: np.ndarray,
        index_name: str,
        m: int = 32,
        ef_construction: int = 200,
        ids: Optional[Sequence[int]] = None,
        metric: int = faiss.METRIC_INNER_PRODUCT,
    ) -> faiss.Index:
        """
        Build HNSW index from vectors and persist it.

        Args:
            vectors: Numpy array of shape (n, dim) with float32 vectors.
                     L2-normalised in place.
            index_name: Name for the index (used for persistence)
            m: Number of connections per node (higher = more accurate, slower)
            ef_construction: Size of dynamic candidate list
            ids: Optional caller ids, one per vector. When given, the index is
                 wrapped in ``IndexIDMap2`` so ``search`` returns these ids
                 instead of positions. Strongly preferred: an index without
                 ids cannot be appended to safely, because position *i* stops
                 meaning row *i* of any query the moment anything appends.
            metric: Defaults to inner product, which on the normalised vectors
                    this builds *is* cosine similarity.

        Returns:
            FAISS index (HNSW, or IndexFlat fallback if HNSW is unavailable)
        """
        if vectors.shape[0] == 0:
            raise ValueError("Cannot build index with empty vectors")
        if ids is not None and len(ids) != vectors.shape[0]:
            raise ValueError(
                f"ids/vectors length mismatch: {len(ids)} != {vectors.shape[0]}"
            )

        dim = vectors.shape[1]

        # Normalize vectors for cosine similarity
        faiss.normalize_L2(vectors)

        try:
            base = faiss.IndexHNSWFlat(dim, m, metric)
            if hasattr(base, "hnsw") and hasattr(base.hnsw, "efConstruction"):
                base.hnsw.efConstruction = ef_construction
        except (AttributeError, TypeError, RuntimeError) as exc:
            logger.warning(
                "HNSW unavailable, falling back to exact flat index: %s", exc
            )
            base = (
                faiss.IndexFlatIP(dim)
                if metric == faiss.METRIC_INNER_PRODUCT
                else faiss.IndexFlatL2(dim)
            )

        if ids is None:
            index = base
            index.add(vectors)
        else:
            index = faiss.IndexIDMap2(base)
            index.add_with_ids(vectors, np.asarray(ids, dtype="int64"))

        with self.index_lock(index_name):
            self._write_atomic(index, index_name)

        return index

    def add_to_index(
        self,
        index: faiss.Index,
        vectors: np.ndarray,
        ids: Optional[Sequence[int]] = None,
    ) -> faiss.Index:
        """
        Append normalised vectors to an in-memory index.

        Raises if the index is ID-mapped and no ids are supplied — silently
        assigning sequential ids there is precisely how the position mapping
        rotted last time.
        """
        vectors = np.ascontiguousarray(vectors, dtype="float32")
        faiss.normalize_L2(vectors)

        if is_id_mapped(index):
            if ids is None:
                raise ValueError(
                    "index is ID-mapped: ids are required when appending"
                )
            index.add_with_ids(vectors, np.asarray(ids, dtype="int64"))
        else:
            if ids is not None:
                raise ValueError(
                    "index is not ID-mapped: it cannot store ids. Rebuild it "
                    "with build_hnsw_index(..., ids=...)"
                )
            index.add(vectors)
        return index

    # ------------------------------------------------------------------
    # LOAD / SAVE
    # ------------------------------------------------------------------

    def load_index(self, index_name: str) -> Optional[faiss.Index]:
        """
        Load index from disk.

        Takes no lock on purpose: ``save_index`` publishes with ``os.replace``,
        so a reader always sees a complete file, and the per-face match path
        reads far more often than it writes.

        Returns:
            FAISS index or None if not found / unreadable
        """
        index_path = self.get_index_path(index_name)
        if not index_path.exists():
            return None

        try:
            return faiss.read_index(str(index_path))
        except Exception as exc:
            logger.error("Error loading index %s from %s: %s",
                         index_name, index_path, exc)
            return None

    def _write_atomic(self, index: faiss.Index, index_name: str) -> None:
        """
        Write ``index`` to its final path atomically.

        Temp file in the same directory (``os.replace`` is only atomic within
        one filesystem), fsync, rename, then fsync the directory so the rename
        itself survives a power loss. Caller must hold ``index_lock``.
        """
        final_path = self.get_index_path(index_name)
        fd, tmp_path = tempfile.mkstemp(
            dir=str(self.index_dir), prefix=f".{index_name}.", suffix=".tmp"
        )
        os.close(fd)
        try:
            faiss.write_index(index, tmp_path)
            with open(tmp_path, "rb") as fh:
                os.fsync(fh.fileno())
            os.replace(tmp_path, final_path)
            dir_fd = os.open(str(self.index_dir), os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        except BaseException:
            # Leave the previous index alone and take the partial file with us.
            try:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
            except OSError:
                pass
            raise

    def save_index(self, index: faiss.Index, index_name: str) -> bool:
        """
        Save index to disk atomically, under the index lock.

        Returns:
            True if successful. On failure the previously persisted index is
            left untouched.
        """
        try:
            with self.index_lock(index_name):
                self._write_atomic(index, index_name)
            return True
        except Exception as exc:
            logger.error("Error saving index %s: %s", index_name, exc)
            print(f"Error saving index {index_name}: {exc}")
            return False

    # ------------------------------------------------------------------
    # SEARCH
    # ------------------------------------------------------------------

    def search(
        self,
        index: faiss.Index,
        query_vectors: np.ndarray,
        k: int = 10,
        ef_search: int = 50
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search in the index.

        Args:
            index: FAISS index
            query_vectors: Query vectors (n, dim). Copied, then normalised.
            k: Number of nearest neighbours to return
            ef_search: Size of dynamic candidate list for HNSW

        Returns:
            ``(scores, ids)``. For an inner-product index — everything
            ``build_hnsw_index`` produces — ``scores`` are cosine similarities,
            directly comparable against ``FACE_MATCH_THRESHOLD``. ``ids`` are
            the caller's ids for an ID-mapped index, otherwise positions.
            Use :func:`scores_to_cosine` if the index metric is unknown.
        """
        # Normalize query vectors (use copy to avoid modifying input)
        query_vectors = np.ascontiguousarray(query_vectors, dtype="float32").copy()
        faiss.normalize_L2(query_vectors)

        # Set ef_search for HNSW (on the base index if this is an ID map)
        base = index
        if hasattr(base, "index") and base.index is not None:
            try:
                base = faiss.downcast_index(base.index)
            except Exception:
                base = base.index
        if hasattr(base, "hnsw") and hasattr(base.hnsw, "efSearch"):
            base.hnsw.efSearch = ef_search

        return index.search(query_vectors, k)

    def metric_of(self, index: faiss.Index) -> int:
        """The metric of an index, unwrapping an ID map if present."""
        base = index
        if hasattr(base, "index") and base.index is not None:
            try:
                base = faiss.downcast_index(base.index)
            except Exception:
                base = base.index
        return getattr(base, "metric_type", faiss.METRIC_INNER_PRODUCT)

    # ------------------------------------------------------------------
    # UPDATE
    # ------------------------------------------------------------------

    def update_index(
        self,
        index_name: str,
        new_vectors: np.ndarray,
        new_ids: Optional[List[int]] = None
    ) -> faiss.Index:
        """
        Update existing index with new vectors.

        The whole load-append-save runs under ``index_lock``, so concurrent
        workers serialise instead of clobbering each other. ``new_ids`` used to
        be accepted and silently ignored; it is honoured now.

        Args:
            index_name: Name of the index
            new_vectors: New vectors to add
            new_ids: Caller ids for the new vectors. Required if the existing
                     index is ID-mapped.

        Returns:
            Updated index
        """
        with self.index_lock(index_name):
            index = self.load_index(index_name)

            if index is None:
                return self.build_hnsw_index(
                    new_vectors, index_name, ids=new_ids
                )

            self.add_to_index(index, new_vectors, new_ids)
            self._write_atomic(index, index_name)
            return index

    def index_exists(self, index_name: str) -> bool:
        """Check if index exists on disk."""
        return self.get_index_path(index_name).exists()

    def delete_index(self, index_name: str) -> bool:
        """Delete index from disk."""
        index_path = self.get_index_path(index_name)
        if index_path.exists():
            with self.index_lock(index_name):
                if index_path.exists():
                    index_path.unlink()
                    return True
        return False


def scores_to_cosine(scores: np.ndarray, metric: int) -> np.ndarray:
    """
    Normalise raw FAISS scores to cosine similarity for unit-norm vectors.

    ``METRIC_INNER_PRODUCT`` scores already *are* cosine. ``METRIC_L2`` scores
    are **squared** L2 distances, and for unit vectors
    ``d^2 = 2(1 - cos)``, so ``cos = 1 - d^2 / 2``.

    This replaces a batch-wide guess (``if scores.max() > 1.0: ...``) that
    inspected the data instead of asking the index. That guess dropped good
    matches: a single-query search whose one hit was a *close* match produced a
    max below 1.0, no conversion ran, and a squared distance of ~0.3 was
    compared against a cosine threshold of 0.60 and rejected.
    """
    if metric == faiss.METRIC_INNER_PRODUCT:
        return np.clip(scores, -1.0, 1.0)
    return np.clip(1.0 - (np.asarray(scores, dtype="float64") / 2.0), -1.0, 1.0)
