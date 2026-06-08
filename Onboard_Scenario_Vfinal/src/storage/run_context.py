"""
run_context.py
──────────────
Creates a unique run ID at startup and registers it in data/runs.json.
Provides paths for all run-scoped output directories.

Usage (from project root):
    from src.storage.run_context import RunContext
    ctx = RunContext()
    ctx.run_id                   # "run_20260426_103000"
    ctx.predictions_dir          # "data/predictions/run_20260426_103000/"
    ctx.inference_output_dir     # "data/inference_output/run_20260426_103000/"
    ctx.latency_path             # "data/metrics/run_20260426_103000_latency.csv"
"""

import os
import json
from datetime import datetime

from src.utils.config import DATA_DIR

RUNS_FILE = os.path.join(DATA_DIR, "runs.json")


class RunContext:
    def __init__(self):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_id  = f"run_{ts}"
        self.created = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        self.predictions_dir       = os.path.join(DATA_DIR, "predictions",      self.run_id)
        self.inference_output_dir  = os.path.join(DATA_DIR, "inference_output", self.run_id)
        self.latency_path          = os.path.join(DATA_DIR, "metrics", f"{self.run_id}_latency.csv")

        os.makedirs(self.predictions_dir,      exist_ok=True)
        os.makedirs(self.inference_output_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.latency_path), exist_ok=True)

        self._register()

    def _register(self):
        os.makedirs(DATA_DIR, exist_ok=True)

        if os.path.exists(RUNS_FILE):
            with open(RUNS_FILE, "r") as f:
                data = json.load(f)
        else:
            data = {"runs": []}

        data["runs"].append({
            "run_id":          self.run_id,
            "created":         self.created,
            "has_predictions": False,
            "has_latency":     False,
        })

        with open(RUNS_FILE, "w") as f:
            json.dump(data, f, indent=2)

    def mark_complete(self, has_predictions: bool, has_latency: bool):
        """Update runs.json flags once the run finishes."""
        if not os.path.exists(RUNS_FILE):
            return

        with open(RUNS_FILE, "r") as f:
            data = json.load(f)

        for entry in data["runs"]:
            if entry["run_id"] == self.run_id:
                entry["has_predictions"] = has_predictions
                entry["has_latency"]     = has_latency
                break

        with open(RUNS_FILE, "w") as f:
            json.dump(data, f, indent=2)
