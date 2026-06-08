"""
latency_storage.py
──────────────────
Saves per-frame latency rows to:
    data/metrics/<run_id>_latency.csv

CSV columns:
    timestamp, image_id, preprocess_ms, inference_ms, total_ms
"""

import csv
import os
from datetime import datetime


class LatencyStorage:
    def __init__(self, latency_path: str):
        """
        Parameters
        ----------
        latency_path : full path from RunContext.latency_path
        """
        os.makedirs(os.path.dirname(latency_path), exist_ok=True)

        self._file   = open(latency_path, mode="w", newline="")
        self._writer = csv.writer(self._file)
        self._writer.writerow([
            "timestamp",
            "image_id",
            "preprocess_ms",
            "inference_ms",
            "total_ms",
        ])

    def log(self, image_id: str, preprocess_ms: float, inference_ms: float):
        self._writer.writerow([
            datetime.now().isoformat(),
            image_id,
            round(preprocess_ms, 3),
            round(inference_ms, 3),
            round(preprocess_ms + inference_ms, 3),
        ])
        self._file.flush()

    def close(self):
        self._file.close()
