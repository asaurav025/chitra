"""Client for the CLIP embedding sidecar.

Why this module exists
----------------------
`chitra-api.service` was cgroup-OOM-killed six times in two days. The measured
cause was that every uvicorn worker imported `core.embedder`, so each of the
four workers carried ~450 MB of torch/transformers import weight before loading
a single model, and up to ~1.14 GB once search touched CLIP — against a 4 GB
cap. The entire compute cost being paid for was 12.7 ms of text embedding per
query.

So CLIP moved into one long-lived sidecar process (`embed_service.py`) and the
API asks it over loopback HTTP. One model per box instead of one per worker.
The measured hop is ~2 ms on top of a ~13 ms embed, which is not a regression
worth discussing next to the memory it returns.

**There is deliberately no in-process fallback.** If the sidecar is unreachable
the caller gets a 503. A fallback that quietly built a `ClipEmbedder` here
would re-create the original OOM under exactly the load that triggers it, and
would do so invisibly — the service would look healthy right up to the kill.
Failing loudly is the feature.

Nothing in this module may import torch, transformers, or `core.embedder`; a
test asserts it. The whole point is that the API stays ML-free.

Wire format
-----------
    POST /embed/text  {"text": "..."}
    200               {"dim": 512, "dtype": "float32", "vector_b64": "..."}

Base64-encoded float32 rather than a JSON array of floats: it is lossless (no
float-repr round-trip), about 3x more compact, and decodes with a single
`np.frombuffer`.

Configuration
-------------
    CHITRA_EMBED_URL      default http://127.0.0.1:5101
    CHITRA_EMBED_TIMEOUT  seconds, default 5.0
    CHITRA_EMBED_TOKEN    optional shared secret; sent as a bearer token
"""
from __future__ import annotations

import base64
import os
from typing import Optional

import httpx
import numpy as np
from fastapi import HTTPException

DEFAULT_URL = "http://127.0.0.1:5101"
DEFAULT_TIMEOUT = 5.0

# The detail string the clients see. Kept stable so a dashboard can key on it.
UNAVAILABLE_DETAIL = "search_unavailable"


class EmbeddingClient:
    """Async client for the embedding sidecar.

    Shape-compatible with the `ClipEmbedder.text_embedding` it replaces, except
    that it is a coroutine — which is the point. The old call blocked the
    uvicorn event loop for the duration of the model forward pass.
    """

    def __init__(
        self,
        http: Optional[httpx.AsyncClient] = None,
        base_url: Optional[str] = None,
        timeout: Optional[float] = None,
        token: Optional[str] = None,
    ):
        self.base_url = (base_url or os.environ.get("CHITRA_EMBED_URL") or DEFAULT_URL).rstrip("/")
        if timeout is None:
            try:
                timeout = float(os.environ.get("CHITRA_EMBED_TIMEOUT", DEFAULT_TIMEOUT))
            except (TypeError, ValueError):
                timeout = DEFAULT_TIMEOUT
        self.timeout = timeout
        self.token = token if token is not None else (os.environ.get("CHITRA_EMBED_TOKEN") or None)

        # A caller-supplied client is shared across requests and owned by the
        # caller (the API builds one in its lifespan). Otherwise we own ours.
        self._owns_http = http is None
        self._http = http if http is not None else httpx.AsyncClient(timeout=self.timeout)

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------
    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self.token}"} if self.token else {}

    @staticmethod
    def _decode(payload: dict) -> np.ndarray:
        """Decode the wire payload into a 1-D float32 vector."""
        dtype = payload.get("dtype", "float32")
        if dtype != "float32":
            raise ValueError(f"unexpected dtype {dtype!r}")
        raw = base64.b64decode(payload["vector_b64"])
        vec = np.frombuffer(raw, dtype="float32")
        dim = int(payload.get("dim", vec.shape[0]))
        if vec.shape[0] != dim:
            # A truncated response must not become a silently shorter vector —
            # it would still rank, just wrongly.
            raise ValueError(f"payload declares dim={dim} but carries {vec.shape[0]} floats")
        return vec

    def _unavailable(self, reason: str) -> HTTPException:
        return HTTPException(
            status_code=503,
            detail=f"{UNAVAILABLE_DETAIL}: embedding service at {self.base_url} {reason}",
        )

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    async def text_embedding(self, text: str) -> np.ndarray:
        """Embed `text`, returning an L2-normalised float32 vector.

        Raises `HTTPException(503)` for every failure mode. Never falls back to
        an in-process model.
        """
        try:
            resp = await self._http.post(
                f"{self.base_url}/embed/text",
                json={"text": text},
                headers=self._headers(),
                timeout=self.timeout,
            )
        except httpx.TimeoutException as exc:
            raise self._unavailable(f"timed out after {self.timeout}s ({exc})") from exc
        except httpx.HTTPError as exc:
            raise self._unavailable(f"is unreachable ({exc})") from exc

        if resp.status_code != 200:
            body = resp.text[:200]
            raise self._unavailable(f"returned HTTP {resp.status_code}: {body}")

        try:
            vec = self._decode(resp.json())
        except Exception as exc:
            raise self._unavailable(f"returned an unusable payload ({exc})") from exc

        # Defensive re-normalisation: the sidecar normalises already, but the
        # ranking downstream is a bare dot product and assumes a unit vector.
        return vec / (np.linalg.norm(vec) + 1e-9)

    async def health(self) -> str:
        """Report the sidecar's status for `/api/health`. Never raises."""
        try:
            resp = await self._http.get(
                f"{self.base_url}/health",
                headers=self._headers(),
                timeout=self.timeout,
            )
        except httpx.HTTPError as exc:
            return f"unavailable: {exc}"
        if resp.status_code != 200:
            return f"unavailable: HTTP {resp.status_code}"
        try:
            return str(resp.json().get("status", "unknown"))
        except Exception as exc:
            return f"unavailable: unusable payload ({exc})"

    async def aclose(self) -> None:
        """Close the underlying transport if this client created it."""
        if self._owns_http:
            await self._http.aclose()
