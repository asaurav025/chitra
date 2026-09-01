"""Synchronous client for the CLIP embedding sidecar — for the RQ jobs.

Why this module exists
----------------------
RQ forks a work-horse per job, so `core/jobs.py`'s module-level `_EMBEDDER` is
built in the child and thrown away when it exits. Every embedding job therefore
paid a full CLIP load: ~90% of the measured 58.5 s job, and 1.67 GB of peak RSS
in a cgroup that has been OOM-killed before.

`embed_service.py` already holds one resident CLIP for the whole box and has
exposed `POST /embed/image` since the day it was written, with no caller. This
is that caller. The job now uploads over loopback the same bytes it already had
in hand, and the 58.5 s becomes ~113 ms of forward pass in a process that never
unloads the model.

`core/embed_client.py` is the *async* client, because its caller is a uvicorn
handler. This one is separate rather than a mode of that one for two reasons:
jobs are plain synchronous functions and `asyncio.run()` per call would build
and tear down an event loop for one round trip; and the two need different
timeouts (below). The wire format is not duplicated — `_decode` is imported
from the async client, so a change to the payload contract cannot leave one of
them behind.

The timeout is deliberately not the API's
-----------------------------------------
`CHITRA_EMBED_TIMEOUT` is 5 s because a user is waiting on a search. A cold
sidecar takes ~10 s to load CLIP, so a worker that gave up at 5 s would fail
every job in the window after a restart while the sidecar was perfectly
healthy. Jobs are queued work and can afford to wait: `CHITRA_EMBED_JOB_TIMEOUT`
defaults to 60 s.

There is deliberately no in-process fallback
--------------------------------------------
Every failure raises `EmbeddingUnavailable` and the job re-raises it, so RQ
marks the job failed and it is visible. Quietly constructing a `ClipEmbedder`
here would restore both the 58 s path and the 1.67 GB of residency invisibly —
the pipeline would simply become slow again with nothing in the logs to say so.
Failing loudly is the feature. `tests/test_embed_client_sync.py` pins that this
module names neither `ClipEmbedder` nor `core.embedder`.

Wire format
-----------
    POST /embed/image  multipart file=<bytes>, filename carries the suffix
    POST /embed/text   {"text": "..."}
    200                {"dim": 512, "dtype": "float32", "vector_b64": "..."}

Configuration
-------------
    CHITRA_EMBED_URL          default http://127.0.0.1:5101
    CHITRA_EMBED_JOB_TIMEOUT  seconds, default 60.0
    CHITRA_EMBED_TOKEN        optional shared secret; sent as a bearer token
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence, Tuple

import httpx
import numpy as np

from core.embed_client import DEFAULT_URL
from core.embed_client import EmbeddingClient as _AsyncClient

#: A job can afford to wait for a cold sidecar; a search request cannot. See the
#: module docstring.
DEFAULT_JOB_TIMEOUT = 60.0

#: One decode implementation for both clients. It rejects a truncated payload
#: rather than returning a silently shorter vector, which would still rank —
#: just wrongly.
_decode = _AsyncClient._decode


class EmbeddingUnavailable(RuntimeError):
    """The sidecar could not answer. Jobs re-raise this; nothing falls back."""


class SyncEmbeddingClient:
    """Blocking client for the embedding sidecar, for use inside RQ jobs."""

    def __init__(
        self,
        http: Optional[httpx.Client] = None,
        base_url: Optional[str] = None,
        timeout: Optional[float] = None,
        token: Optional[str] = None,
    ):
        self.base_url = (base_url or os.environ.get("CHITRA_EMBED_URL") or DEFAULT_URL).rstrip("/")
        if timeout is None:
            try:
                timeout = float(os.environ.get("CHITRA_EMBED_JOB_TIMEOUT", DEFAULT_JOB_TIMEOUT))
            except (TypeError, ValueError):
                timeout = DEFAULT_JOB_TIMEOUT
        self.timeout = timeout
        self.token = token if token is not None else (os.environ.get("CHITRA_EMBED_TOKEN") or None)

        self._owns_http = http is None
        self._http = http if http is not None else httpx.Client(timeout=self.timeout)

        # The label vectors are identical for every photo, so they are embedded
        # once and reused. In a single-photo job that is one round trip per
        # label; in `index_embeddings_batch_job`, which shares one client across
        # the whole batch, it is one round trip per label for the entire run.
        self._label_cache: Dict[str, np.ndarray] = {}

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------
    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self.token}"} if self.token else {}

    def _unavailable(self, reason: str) -> EmbeddingUnavailable:
        return EmbeddingUnavailable(f"embedding service at {self.base_url} {reason}")

    def _request(self, path: str, **kwargs) -> np.ndarray:
        try:
            resp = self._http.post(
                f"{self.base_url}{path}",
                headers=self._headers(),
                timeout=self.timeout,
                **kwargs,
            )
        except httpx.TimeoutException as exc:
            raise self._unavailable(f"timed out after {self.timeout}s ({exc})") from exc
        except httpx.HTTPError as exc:
            raise self._unavailable(f"is unreachable ({exc})") from exc

        if resp.status_code != 200:
            raise self._unavailable(f"returned HTTP {resp.status_code}: {resp.text[:200]}")

        try:
            vec = _decode(resp.json())
        except Exception as exc:
            raise self._unavailable(f"returned an unusable payload ({exc})") from exc

        # The sidecar normalises already; downstream scoring is a bare dot
        # product and assumes a unit vector, so make that unconditional.
        return vec / (np.linalg.norm(vec) + 1e-9)

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def image_embedding(self, filename: str, data: bytes) -> np.ndarray:
        """Embed image `data`, returning an L2-normalised float32 vector.

        `filename` is passed through because the sidecar keeps its suffix on the
        temp file it writes. `core.extractor.load_image` dispatches on the
        file's bytes rather than its name, so a wrong suffix is no longer fatal
        — but an honest name still costs nothing.
        """
        return self._request(
            "/embed/image",
            files={"file": (filename, data, "application/octet-stream")},
        )

    def text_embedding(self, text: str) -> np.ndarray:
        """Embed `text`, returning an L2-normalised float32 vector."""
        return self._request("/embed/text", json={"text": text})

    def label_vectors(self, labels: Sequence[str]) -> np.ndarray:
        """`(len(labels), dim)` of normalised label vectors, cached per client."""
        for label in labels:
            if label not in self._label_cache:
                self._label_cache[label] = self.text_embedding(label)
        return np.stack([self._label_cache[label] for label in labels])

    def rank_labels_for_vector(
        self,
        image_vec: np.ndarray,
        labels: Sequence[str],
        top_k: int = 6,
    ) -> List[Tuple[str, float]]:
        """`ClipEmbedder.rank_labels`'s arithmetic, without its second image pass.

        `rank_labels` calls `image_embedding` again for every call — so the old
        embedding job ran two CLIP forward passes over the same photo, one for
        the stored vector and one thrown away inside `auto_tags`. Both vectors
        are normalised, so the score is a plain dot product and the image vector
        already in hand is all this needs.
        """
        if not labels:
            return []
        sims = self.label_vectors(labels) @ np.asarray(image_vec, dtype="float32")
        order = sims.argsort()[::-1][:top_k]
        return [(labels[i], float(sims[i])) for i in order]

    def health(self) -> str:
        """Report the sidecar's status. Never raises."""
        try:
            resp = self._http.get(
                f"{self.base_url}/health", headers=self._headers(), timeout=self.timeout
            )
        except httpx.HTTPError as exc:
            return f"unavailable: {exc}"
        if resp.status_code != 200:
            return f"unavailable: HTTP {resp.status_code}"
        try:
            return str(resp.json().get("status", "unknown"))
        except Exception as exc:
            return f"unavailable: unusable payload ({exc})"

    def close(self) -> None:
        """Close the underlying transport if this client created it."""
        if self._owns_http:
            self._http.close()
