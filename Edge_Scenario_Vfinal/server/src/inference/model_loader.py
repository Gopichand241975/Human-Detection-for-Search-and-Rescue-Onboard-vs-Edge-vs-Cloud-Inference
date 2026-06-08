"""
model_loader.py
───────────────
Loads best.onnx via ONNX Runtime.

Execution provider: CPUExecutionProvider (hardcoded — CPU only).

The session is created once and cached for the process lifetime.
MODEL_PATH is read exclusively from server/config/config.py.
"""

import os
import onnxruntime as ort

from server.config.config import MODEL_PATH

_session = None


def _build_session(model_path: str) -> ort.InferenceSession:
    """Create an ONNX Runtime InferenceSession on CPU only."""
    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    print(f"[MODEL] Loaded '{os.path.basename(model_path)}' on CPU")
    return session


def load_model() -> ort.InferenceSession:
    """Return the singleton ONNX Runtime session, creating it on first call."""
    global _session

    if _session is None:
        model_path = os.path.abspath(MODEL_PATH)
        if not os.path.isfile(model_path):
            raise FileNotFoundError(
                f"[MODEL] ONNX model not found at: {model_path}\n"
                "Export best.pt with: yolo export model=best.pt format=onnx"
            )
        _session = _build_session(model_path)

    return _session
