"""
model_loader.py
───────────────
Loads best.onnx via ONNX Runtime with GPU-first, CPU-fallback logic.

Provider priority
─────────────────
When USE_GPU=true (the default):
  1. CUDAExecutionProvider   — NVIDIA GPU via CUDA
  2. CPUExecutionProvider    — fallback if CUDA is unavailable

When USE_GPU=false:
  1. CPUExecutionProvider    — forced CPU (e.g. testing, no driver)

Fallback is automatic: if onnxruntime-gpu is installed but no CUDA
device is found at runtime, ONNX Runtime silently drops CUDA and runs
on CPU.  We detect which provider was actually selected and log it so
the operator knows what is running.

The session is created once and cached for the process lifetime.
MODEL_PATH and USE_GPU are read exclusively from server/config/config.py.
"""

import os
import onnxruntime as ort

from server.config.config import MODEL_PATH, USE_GPU

_session: ort.InferenceSession | None = None


# ──────────────────────────────────────────────────────────────────────
# PROVIDER SELECTION
# ──────────────────────────────────────────────────────────────────────

def _resolve_providers() -> list[str]:
    """
    Return an ordered provider list based on USE_GPU and what is
    actually available in the current onnxruntime build.

    ONNX Runtime tries providers in order and silently skips any that
    are unavailable, so putting CUDA first is always safe — it will
    fall back to CPU automatically if CUDA is not present.
    """
    available = ort.get_available_providers()

    if not USE_GPU:
        return ["CPUExecutionProvider"]

    # Build preference list: GPU providers first, CPU as final fallback.
    # DmlExecutionProvider covers DirectML (Windows GPU / NPU).
    gpu_providers = [
        p for p in ("CUDAExecutionProvider", "DmlExecutionProvider")
        if p in available
    ]

    return gpu_providers + ["CPUExecutionProvider"]


def _active_provider(session: ort.InferenceSession) -> str:
    """Return the first provider that the session is actually using."""
    providers = session.get_providers()
    return providers[0] if providers else "unknown"


# ──────────────────────────────────────────────────────────────────────
# SESSION CREATION
# ──────────────────────────────────────────────────────────────────────

def _build_session(model_path: str) -> ort.InferenceSession:
    """
    Create an ONNX Runtime InferenceSession, preferring GPU when
    USE_GPU=true, with automatic CPU fallback.
    """
    providers = _resolve_providers()

    try:
        session = ort.InferenceSession(model_path, providers=providers)
    except Exception as gpu_err:
        # If the preferred provider list itself causes an init error
        # (rare — usually only happens with misconfigured CUDA installs),
        # retry on CPU only before giving up.
        print(
            f"[MODEL] WARNING: session init with {providers} failed "
            f"({gpu_err}). Retrying on CPU only."
        )
        session = ort.InferenceSession(
            model_path, providers=["CPUExecutionProvider"]
        )

    active = _active_provider(session)
    device = "GPU" if "CPU" not in active else "CPU"
    print(
        f"[MODEL] Loaded '{os.path.basename(model_path)}' "
        f"on {device} ({active})"
    )

    if USE_GPU and device == "CPU":
        print(
            "[MODEL] NOTE: GPU was requested but is unavailable — "
            "running on CPU. Install onnxruntime-gpu and a CUDA toolkit "
            "to enable GPU inference."
        )

    return session


# ──────────────────────────────────────────────────────────────────────
# PUBLIC API
# ──────────────────────────────────────────────────────────────────────

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
