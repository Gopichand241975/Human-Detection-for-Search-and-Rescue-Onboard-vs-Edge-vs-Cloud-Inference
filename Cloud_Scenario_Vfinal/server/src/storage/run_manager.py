"""
run_manager.py
──────────────
Creates a datetime-stamped run and returns a context dict with every
relevant path used throughout the server.

Final directory layout
──────────────────────
runs/
  run_20260424_153012/
    images/           ← raw images received from client
    logs/             ← per-run events (connections, images, errors)

data/
  metrics/
    run_20260424_153012_latency.csv
  predictions/
    run_20260424_153012/
      <image_id>.json
  inference_output/
    run_20260424_153012/
      <image_id>.jpg  ← annotated images with bounding boxes

server/logs/          ← global logs (startup, security, fatal errors)
  server.log
  security.log
  errors.log
"""

import os
from datetime import datetime

# ── Run storage ────────────────────────────────────────────────────────
BASE_RUN_DIR = "runs"

# ── Shared data dirs (one sub-folder per run inside each) ──────────────
DATA_DIR               = "data"
DATA_METRICS_DIR       = os.path.join(DATA_DIR, "metrics")
DATA_PREDICTIONS_DIR   = os.path.join(DATA_DIR, "predictions")
DATA_INFERENCE_OUT_DIR = os.path.join(DATA_DIR, "inference_output")


def create_new_run() -> dict:
    """
    Creates all required directories for a new run and returns a context
    dict that is registered via set_run_context() in run_server.py.

    Keys returned
    run_name         str   e.g. "run_20260424_153012"
    run_path         str   runs/run_20260424_153012/
    images           str   runs/run_20260424_153012/images/
    logs             str   runs/run_20260424_153012/logs/
    metrics          str   data/metrics/
    predictions      str   data/predictions/run_20260424_.../
    inference_output str   data/inference_output/run_20260424_.../
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name  = f"run_{timestamp}"

    # ── Per-run folder (inside runs/) ──────────────────────────────────
    run_path    = os.path.join(BASE_RUN_DIR, run_name)
    images_path = os.path.join(run_path, "images")
    logs_path   = os.path.join(run_path, "logs")

    # ── Per-run sub-folders inside shared data/ dirs ───────────────────
    metrics_path       = DATA_METRICS_DIR
    predictions_path   = os.path.join(DATA_PREDICTIONS_DIR,   run_name)
    inference_out_path = os.path.join(DATA_INFERENCE_OUT_DIR, run_name)

    for path in (
        images_path,
        logs_path,
        metrics_path,
        predictions_path,
        inference_out_path,
    ):
        os.makedirs(path, exist_ok=True)

    return {
        "run_name":         run_name,
        "run_path":         run_path,
        "images":           images_path,
        "logs":             logs_path,
        "metrics":          metrics_path,
        "predictions":      predictions_path,
        "inference_output": inference_out_path,
    }
