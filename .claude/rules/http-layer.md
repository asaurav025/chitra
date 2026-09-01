---
paths:
  - "app_fastapi.py"
  - "core/schemas.py"
  - "core/auth.py"
---

# HTTP layer rules

**Every route needs an explicit auth dependency.** Only `/`, `/api/health`,
`/api/auth/register`, `/api/auth/login` and `/api/auth/logout` are intentionally
public. Add `Depends(get_current_active_user)` — or `require_admin` for
destructive and administrative routes — to everything else. A missing dependency
is silent: the route simply serves anyone who can reach the tunnel.

**`app_fastapi.py` is a 2,509-line god-file** holding middleware, DI, DTO
mappers, thumbnail helpers and all 40 routes. Lines 1524-1913 are unreachable
dead code. Prefer adding new endpoint groups as routers over growing this file.

**No table has an owner column.** Photos, faces and persons are global — every
approved user sees everything. That is the current model; if you add
per-user scoping, it is a schema change, not a filter.

**Client-supplied filenames flow into MinIO object keys unsanitized**
(`generate_photo_path`). Anything accepting a filename must normalize with
`Path(filename).name` plus a character allowlist before it reaches a key.

**Uploads have no type or per-file size validation.** The only guard is a
request-level `Content-Length` check that chunked encoding bypasses. Validate
against `IMG_EXTS`/`RAW_EXTS`/`VIDEO_EXTS` in the streaming loop.

**The health check returns 200 while degraded**, so monitors see a broken
instance as healthy. If you touch it, return 503 when `status == "degraded"`.

**Never call a model or a blocking encode directly in an async handler.** CLIP
text embedding at the search endpoint blocks the uvicorn event loop today —
new work goes through `run_in_executor` or a job.
