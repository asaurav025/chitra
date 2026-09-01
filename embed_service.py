"""CLIP embedding sidecar — one model per box instead of one per uvicorn worker.

Why this process exists
-----------------------
`chitra-api.service` was cgroup-OOM-killed six times in two days against a 4 GB
cap. Each of the four uvicorn workers imported `core.embedder` independently
(uvicorn does not fork after import, so nothing is shared): ~450 MB of torch
import weight per worker before any model loaded, and ~1.14 GB once search
touched CLIP. All of it to compute a 13 ms text embedding per query.

This service holds exactly one `ClipEmbedder` for the whole box. The API talks
to it over loopback via `core/embed_client.py` and carries no ML runtime at
all.

Operational notes
-----------------
* **`--workers 1` is mandatory.** Every worker would load its own 1.14 GB copy,
  which is precisely the bug this exists to fix. `start_workers.sh` hard-codes
  it; do not make it configurable.
* **Loopback only.** There is no reason for this port to be reachable off the
  box. `CHITRA_EMBED_TOKEN` adds a shared secret if it ever is.
* **Thread count is set explicitly at startup**, before the model loads.
  Measured on this 6-core box: a CLIP image embed costs 113 ms at 3 threads and
  194-220 ms at the 6-thread default. See `thread_limits.sh`.
* Embeds run in a **single-threaded executor**, so the event loop stays free to
  answer `/health` and so two concurrent requests cannot fight over the same
  three threads.

Run:
    uvicorn embed_service:app --host 127.0.0.1 --port 5101 --workers 1
"""
from __future__ import annotations

import asyncio
import logging
import os
import resource
import tempfile
from base64 import b64encode
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from pydantic import BaseModel, Field

logger = logging.getLogger("chitra.embed")

DEFAULT_MODEL = "openai/clip-vit-base-patch32"
# 3 is the measured optimum for CLIP on this 6-core box, and the text curve is
# flat above it — see docs/plans/api-oom-fix.md, Task 6.4.
DEFAULT_THREADS = 3


class TextRequest(BaseModel):
    # min_length=1 makes an empty string a 422 from the framework rather than a
    # zero vector that would silently rank every photo identically.
    text: str = Field(..., min_length=1)


def configure_threads() -> int:
    """Set the torch thread count from `CHITRA_ML_THREADS`. Returns the count.

    torch is imported here, not at module scope, so `import embed_service`
    stays cheap and testable. The BLAS/OpenMP variables are set even earlier by
    `thread_limits.sh`; this covers torch's own intra-op pool, which it sizes
    from core count regardless.
    """
    raw = os.environ.get("CHITRA_ML_THREADS", "")
    try:
        threads = int(raw)
        if threads < 1:
            raise ValueError(raw)
    except (TypeError, ValueError):
        if raw:
            logger.warning("CHITRA_ML_THREADS=%r is not a positive int; using %d", raw, DEFAULT_THREADS)
        threads = DEFAULT_THREADS

    import torch

    torch.set_num_threads(threads)
    return threads


def _default_embedder():
    """Build the real CLIP embedder. Imported lazily — see `configure_threads`."""
    from core.embedder import ClipEmbedder

    return ClipEmbedder(os.environ.get("CHITRA_CLIP_MODEL", DEFAULT_MODEL))


def _rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _encode(vec: np.ndarray) -> dict:
    """The wire format: base64 float32, lossless and ~3x smaller than JSON floats."""
    arr = np.asarray(vec, dtype="float32").reshape(-1)
    arr = arr / (np.linalg.norm(arr) + 1e-9)
    return {
        "dim": int(arr.shape[0]),
        "dtype": "float32",
        "vector_b64": b64encode(arr.tobytes()).decode("ascii"),
    }


def require_token(authorization: Optional[str] = Header(None)) -> None:
    """Enforce `CHITRA_EMBED_TOKEN` when one is configured. Read per request so
    the sidecar picks the setting up without a code change."""
    expected = os.environ.get("CHITRA_EMBED_TOKEN") or None
    if not expected:
        return
    if authorization != f"Bearer {expected}":
        raise HTTPException(status_code=401, detail="invalid_token")


def create_app(embedder_factory=_default_embedder) -> FastAPI:
    """Build the sidecar app. `embedder_factory` is the seam the tests use."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.threads = configure_threads()
        # max_workers=1: serialise forward passes so the configured thread
        # count is the real degree of parallelism, not a per-request multiple.
        app.state.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="embed")
        before = _rss_mb()
        app.state.embedder = embedder_factory()
        app.state.model = os.environ.get("CHITRA_CLIP_MODEL", DEFAULT_MODEL)
        logger.info(
            "CLIP embedder ready: model=%s threads=%d rss=%.0f MB (+%.0f MB to load)",
            app.state.model, app.state.threads, _rss_mb(), _rss_mb() - before,
        )
        print(
            f"embed_service ready: model={app.state.model} threads={app.state.threads} "
            f"rss={_rss_mb():.0f} MB",
            flush=True,
        )
        try:
            yield
        finally:
            app.state.executor.shutdown(wait=True)
            app.state.embedder = None

    app = FastAPI(title="Chitra embedding sidecar", version="1.0.0", lifespan=lifespan)

    async def run_embed(fn, *args):
        """Run a blocking model call off the event loop."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(app.state.executor, fn, *args)

    @app.get("/health")
    async def health():
        loaded = getattr(app.state, "embedder", None) is not None
        return {
            "status": "ok" if loaded else "loading",
            "model": getattr(app.state, "model", None),
            "threads": getattr(app.state, "threads", 0),
            "rss_mb": round(_rss_mb(), 1),
        }

    @app.post("/embed/text")
    async def embed_text(req: TextRequest, _: None = Depends(require_token)):
        text = req.text.strip()
        if not text:
            # Whitespace-only survives min_length but is just as meaningless.
            raise HTTPException(status_code=422, detail="empty_text")
        vec = await run_embed(app.state.embedder.text_embedding, text)
        return _encode(vec)

    @app.post("/embed/image")
    async def embed_image(file: UploadFile = File(...), _: None = Depends(require_token)):
        """Embed an uploaded image.

        Exposed from day one so routing the RQ workers' embedding jobs here
        later — which would cut a ~58 s job to ~1 s by skipping the per-job
        model load — is a client change only. Nothing calls it yet.
        """
        suffix = os.path.splitext(file.filename or "")[1] or ".bin"
        fd, path = tempfile.mkstemp(suffix=suffix, prefix="chitra-embed-")
        try:
            with os.fdopen(fd, "wb") as fh:
                while chunk := await file.read(1024 * 1024):
                    fh.write(chunk)
            vec = await run_embed(app.state.embedder.image_embedding, path)
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass
        return _encode(vec)

    return app


app = create_app()
