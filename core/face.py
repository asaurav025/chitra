from __future__ import annotations
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List

import numpy as np

from core.extractor import load_image

HAS_INSIGHTFACE = False
_FACE_APP = None

#: Intra-op threads for every ONNX session buffalo_l builds.
#:
#: Measured on this box (i5-8500, 6 cores / 6 threads, no HT), buffalo_l detect
#: at ctx_id=-1 and det_size=(640, 640), median of 15 on an idle machine:
#:
#:     uncapped                   +25 ORT threads    249 ms
#:     intra_op_num_threads=2      +5 ORT threads    143 ms
#:     intra_op_num_threads=3     +10 ORT threads    104 ms   <- 2.4x faster
#:     intra_op_num_threads=4     +15 ORT threads    302 ms
#:
#: The curve is sharply non-monotonic — 4 is worse than uncapped-minus-a-bit and
#: nearly 3x worse than 3 — so this is not a "lower is safer" knob to nudge.
#: Re-measure before changing it.
#:
#: `OMP_NUM_THREADS` cannot do this: the onnxruntime CPU wheel links no OpenMP,
#: and a session adds its threads regardless of what that variable says. That was
#: measured too, in docs/plans/api-oom-fix.md. `FaceAnalysis` forwards only
#: `providers`/`provider_options` and there is no `sess_options` anywhere in the
#: chain, which is why this is done by wrapping the session class rather than by
#: passing an argument.
INSIGHTFACE_INTRA_OP_THREADS = 3


def _capped_options(args, kwargs, threads: int, ort):
    """Force `intra_op_num_threads` onto whichever slot the options live in.

    Call sites in insightface differ: `model_zoo.get_model` passes only
    `providers`/`provider_options` as keywords, while
    `arcface_onnx.py`/`retinaface.py`/`landmark.py` call
    `InferenceSession(model_file, None)` with sess_options *positional*. The
    options object is replaced in place rather than moved between args and
    kwargs, so a call passing `providers` positionally as well is not silently
    reinterpreted.
    """
    args = list(args)
    if len(args) >= 2:
        options = args[1] or ort.SessionOptions()
        options.intra_op_num_threads = threads
        args[1] = options
    else:
        options = kwargs.get("sess_options") or ort.SessionOptions()
        options.intra_op_num_threads = threads
        kwargs["sess_options"] = options
    return tuple(args), kwargs


@contextmanager
def _capped_onnx_sessions(threads: int):
    """Cap every ONNX session built inside this block, and *only* inside it.

    The cap has to be scoped. A module-global monkeypatch would outlive the
    load and silently reconfigure every other ONNX session the process ever
    builds — a 2.4x win on face detection traded for an invisible change to
    everything else. `tests/test_face_threads.py` pins the restoration on the
    success path and the failure path.
    """
    import onnxruntime as ort

    original = ort.InferenceSession

    class _CappedInferenceSession(original):
        def __init__(self, *args, **kwargs):
            args, kwargs = _capped_options(args, kwargs, threads, ort)
            super().__init__(*args, **kwargs)

    ort.InferenceSession = _CappedInferenceSession
    restore_pickable = _cap_preimported_insightface(_CappedInferenceSession, threads, ort)
    try:
        yield _CappedInferenceSession
    finally:
        ort.InferenceSession = original
        if restore_pickable is not None:
            restore_pickable()


def _cap_preimported_insightface(capped, threads: int, ort):
    """Cap `PickableInferenceSession` when insightface was imported too early.

    `insightface.model_zoo.model_zoo.PickableInferenceSession` subclasses
    `onnxruntime.InferenceSession` at *import* time. Patching the module
    attribute therefore only reaches it when insightface is imported **inside**
    the scope — which is why `_lazy_init_insightface` imports it there. If
    something imported insightface first (the sidecar will, once buffalo_l goes
    co-resident with CLIP) the cap would otherwise be a silent no-op: no error,
    just 249 ms per detect again.

    Returns a callable that puts the original class back, or None if there was
    nothing to do.
    """
    module = sys.modules.get("insightface.model_zoo.model_zoo")
    pickable = getattr(module, "PickableInferenceSession", None)
    if pickable is None or issubclass(pickable, capped):
        return None

    class _CappedPickableSession(pickable):
        def __init__(self, *args, **kwargs):
            args, kwargs = _capped_options(args, kwargs, threads, ort)
            super().__init__(*args, **kwargs)

    module.PickableInferenceSession = _CappedPickableSession
    return lambda: setattr(module, "PickableInferenceSession", pickable)


def _lazy_init_insightface():
    global HAS_INSIGHTFACE, _FACE_APP
    if _FACE_APP is not None:
        return _FACE_APP

    try:
        import onnxruntime  # noqa: F401

        with _capped_onnx_sessions(INSIGHTFACE_INTRA_OP_THREADS):
            # Imported *inside* the block on purpose: insightface binds its own
            # session subclass to `onnxruntime.InferenceSession` at import time,
            # so importing it here is what makes that subclass inherit the cap.
            from insightface.app import FaceAnalysis

            # Use CPU by default; you can tweak providers later.
            _FACE_APP = FaceAnalysis(name="buffalo_l")
            # ctx_id=-1 => CPU only
            _FACE_APP.prepare(ctx_id=-1, det_size=(640, 640))
        HAS_INSIGHTFACE = True
    except Exception as e:
        print(f"[yellow]InsightFace not available:[/yellow] {e}")
        HAS_INSIGHTFACE = False
        _FACE_APP = None

    return _FACE_APP


def face_encodings(file_path: str) -> List[Dict]:
    """
    Return a list of faces for the given image.

    Each item is:
      {
        "bbox": (x, y, w, h),
        "embedding": np.ndarray (float32),
        "score": float,
      }
    """
    app = _lazy_init_insightface()
    if not HAS_INSIGHTFACE or app is None:
        return []

    img = load_image(Path(file_path))
    # InsightFace expects numpy array in BGR or RGB; PIL -> numpy in RGB
    img_np = np.array(img)

    faces = app.get(img_np)
    results: List[Dict] = []

    for f in faces:
        # f.bbox: [x1, y1, x2, y2]
        x1, y1, x2, y2 = f.bbox.astype(int).tolist()
        w = x2 - x1
        h = y2 - y1

        emb = f.normed_embedding.astype("float32")

        results.append(
            {
                "bbox": (int(x1), int(y1), int(w), int(h)),
                "embedding": emb,
                "score": float(getattr(f, "det_score", 1.0)),
            }
        )

    return results
