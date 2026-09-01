"""InsightFace's ONNX sessions run capped at 3 intra-op threads.

The measurement
---------------
buffalo_l detect, `ctx_id=-1`, `det_size=(640, 640)`, on this 6-core i5-8500,
median of 15 on an idle box:

    uncapped                    +25 ORT threads    249 ms
    intra_op_num_threads=2       +5 ORT threads    143 ms
    intra_op_num_threads=3      +10 ORT threads    104 ms   <- 2.4x
    intra_op_num_threads=4      +15 ORT threads    302 ms

`OMP_NUM_THREADS` does not reach ONNX Runtime — the CPU wheel links no OpenMP,
and a session adds its threads regardless of what that variable says. And
`FaceAnalysis` forwards only `providers`/`provider_options`; there is no
`sess_options` anywhere in the chain. Patching `onnxruntime.InferenceSession`
around the construction is the only lever there is.

Why the patch has to be scoped
------------------------------
A module-global monkeypatch would outlive the call and silently reconfigure
every other ONNX session in the process — including any this box grows later.
The wrapper is installed for the duration of the load and restored in a
`finally`, and the tests below pin the restoration on the success path, the
failure path, and the "insightface was already imported" path.
"""
import sys
import types
import unittest
from unittest.mock import patch

import onnxruntime as ort

from core import face


class RecordingSession:
    """Stands in for `onnxruntime.InferenceSession` so nothing loads a model."""

    instances = []

    def __init__(self, *args, **kwargs):
        RecordingSession.instances.append((args, kwargs))

    @classmethod
    def reset(cls):
        cls.instances = []

    @classmethod
    def options_of(cls, index=0):
        args, kwargs = cls.instances[index]
        if "sess_options" in kwargs:
            return kwargs["sess_options"]
        return args[1]


class TestTheCap(unittest.TestCase):
    """`_capped_onnx_sessions` — the scoped wrapper itself."""

    def setUp(self):
        RecordingSession.reset()
        self.original = ort.InferenceSession
        ort.InferenceSession = RecordingSession
        self.addCleanup(lambda: setattr(ort, "InferenceSession", self.original))

    def test_the_measured_thread_count_is_three(self):
        self.assertEqual(3, face.INSIGHTFACE_INTRA_OP_THREADS)

    def test_it_caps_a_session_built_with_keyword_providers(self):
        with face._capped_onnx_sessions(3) as capped:
            capped("buffalo_l.onnx", providers=["CPUExecutionProvider"])

        self.assertEqual(1, len(RecordingSession.instances))
        self.assertEqual(3, RecordingSession.options_of().intra_op_num_threads)

    def test_it_caps_a_session_given_a_positional_none(self):
        """`insightface/model_zoo/arcface_onnx.py:46` calls
        `onnxruntime.InferenceSession(self.model_file, None)` — sess_options
        positional. That has to be capped too, not skipped."""
        with face._capped_onnx_sessions(3) as capped:
            capped("buffalo_l.onnx", None)

        self.assertEqual(3, RecordingSession.options_of().intra_op_num_threads)

    def test_it_caps_an_options_object_the_caller_supplied(self):
        supplied = ort.SessionOptions()
        supplied.intra_op_num_threads = 6

        with face._capped_onnx_sessions(3) as capped:
            capped("buffalo_l.onnx", sess_options=supplied)

        self.assertEqual(3, RecordingSession.options_of().intra_op_num_threads)

    def test_it_restores_inference_session_afterwards(self):
        with face._capped_onnx_sessions(3):
            self.assertIsNot(RecordingSession, ort.InferenceSession)
        self.assertIs(RecordingSession, ort.InferenceSession)

    def test_it_restores_inference_session_when_the_body_raises(self):
        with self.assertRaises(RuntimeError):
            with face._capped_onnx_sessions(3):
                raise RuntimeError("model file is missing")
        self.assertIs(RecordingSession, ort.InferenceSession)


class FakeFaceAnalysis:
    """Records what `onnxruntime.InferenceSession` was while it was built."""

    seen_during_init = None
    prepared_with = None
    raise_on_init = False

    def __init__(self, name=None, **kwargs):
        FakeFaceAnalysis.seen_during_init = ort.InferenceSession
        self.name = name
        if FakeFaceAnalysis.raise_on_init:
            raise RuntimeError("buffalo_l is not downloaded")

    def prepare(self, ctx_id, det_size=None):
        FakeFaceAnalysis.prepared_with = (ctx_id, det_size)


def stub_insightface():
    """`insightface` and `insightface.app`, with no model behind them."""
    package = types.ModuleType("insightface")
    app = types.ModuleType("insightface.app")
    app.FaceAnalysis = FakeFaceAnalysis
    package.app = app
    return {"insightface": package, "insightface.app": app}


class TestLazyInit(unittest.TestCase):
    """`_lazy_init_insightface` — the caller, with no real model anywhere."""

    def setUp(self):
        FakeFaceAnalysis.seen_during_init = None
        FakeFaceAnalysis.prepared_with = None
        FakeFaceAnalysis.raise_on_init = False
        self._saved_app, self._saved_flag = face._FACE_APP, face.HAS_INSIGHTFACE
        face._FACE_APP, face.HAS_INSIGHTFACE = None, False

        def restore():
            face._FACE_APP, face.HAS_INSIGHTFACE = self._saved_app, self._saved_flag

        self.addCleanup(restore)
        self.original = ort.InferenceSession
        self.addCleanup(lambda: setattr(ort, "InferenceSession", self.original))

    def test_face_analysis_is_built_while_the_cap_is_installed(self):
        with patch.dict(sys.modules, stub_insightface()):
            face._lazy_init_insightface()

        self.assertIsNotNone(FakeFaceAnalysis.seen_during_init)
        self.assertIsNot(self.original, FakeFaceAnalysis.seen_during_init)
        self.assertTrue(issubclass(FakeFaceAnalysis.seen_during_init, self.original))

    def test_it_still_prepares_on_cpu_at_640(self):
        with patch.dict(sys.modules, stub_insightface()):
            face._lazy_init_insightface()

        self.assertEqual((-1, (640, 640)), FakeFaceAnalysis.prepared_with)
        self.assertTrue(face.HAS_INSIGHTFACE)

    def test_inference_session_is_restored_afterwards(self):
        """A wrapper that outlived the call would silently reconfigure every
        other ONNX session in the process."""
        with patch.dict(sys.modules, stub_insightface()):
            face._lazy_init_insightface()

        self.assertIs(self.original, ort.InferenceSession)

    def test_inference_session_is_restored_when_the_load_fails(self):
        FakeFaceAnalysis.raise_on_init = True

        with patch.dict(sys.modules, stub_insightface()):
            self.assertIsNone(face._lazy_init_insightface())

        self.assertIs(self.original, ort.InferenceSession)
        self.assertFalse(face.HAS_INSIGHTFACE)


class TestAlreadyImportedInsightface(unittest.TestCase):
    """`PickableInferenceSession` subclasses `InferenceSession` at *import*
    time, so patching the module attribute only reaches it if insightface is
    imported inside the scope. When something imported insightface first — the
    sidecar will, once buffalo_l goes co-resident with CLIP — the subclass has
    to be capped directly, or the cap is silently a no-op."""

    def setUp(self):
        RecordingSession.reset()
        self._saved_app, self._saved_flag = face._FACE_APP, face.HAS_INSIGHTFACE
        face._FACE_APP, face.HAS_INSIGHTFACE = None, False

        def restore():
            face._FACE_APP, face.HAS_INSIGHTFACE = self._saved_app, self._saved_flag

        self.addCleanup(restore)
        self.original = ort.InferenceSession
        self.addCleanup(lambda: setattr(ort, "InferenceSession", self.original))

    def test_a_preimported_pickable_session_is_capped_and_restored(self):
        # Bound to the *unpatched* class, exactly as a real early import would be.
        class PickableInferenceSession(RecordingSession):
            pass

        model_zoo = types.ModuleType("insightface.model_zoo.model_zoo")
        model_zoo.PickableInferenceSession = PickableInferenceSession

        modules = stub_insightface()
        modules["insightface.model_zoo.model_zoo"] = model_zoo

        with patch.dict(sys.modules, modules):
            with face._capped_onnx_sessions(3):
                model_zoo.PickableInferenceSession(
                    "buffalo_l.onnx", providers=["CPUExecutionProvider"]
                )

            self.assertEqual(3, RecordingSession.options_of().intra_op_num_threads)
            self.assertIs(PickableInferenceSession, model_zoo.PickableInferenceSession)


if __name__ == "__main__":
    unittest.main()
