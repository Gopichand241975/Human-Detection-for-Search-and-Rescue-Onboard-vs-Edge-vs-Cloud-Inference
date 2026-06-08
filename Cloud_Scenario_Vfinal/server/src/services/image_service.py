"""
image_service.py
────────────────
Handles one incoming image per call:
  1. Decodes the raw bytes → numpy BGR image
  2. Generates a run-scoped, sequential image_id
     e.g.  run_20260424_153012_img_0003
  3. Saves the raw image to runs/<run>/images/
  4. Submits (image, filename, image_id) to the inference engine

image_id is generated HERE so that the latency CSV, prediction JSON,
and label .txt all share the exact same identifier for every image.

Bug fixed: previously, image_id was taken verbatim from the client
header (e.g. "run1_img_0001"), which never changed across server runs,
causing all runs to produce the same IDs in the latency CSV.
"""

import os
import time
import threading

import numpy as np
import cv2

from server.config.config import INFERENCE_ENABLED
from server.src.inference import engine as inference_engine
from server.src.core.run_context import get_run_context


class ImageService:

    def __init__(self, logger):
        self.logger   = logger
        self._lock    = threading.Lock()    # protect sequence counter
        self._seq     = 0                   # per-run image sequence number

        if INFERENCE_ENABLED:
            inference_engine.start()

    # ──────────────────────────────────────────────
    # ID GENERATION
    # ──────────────────────────────────────────────
    def _next_image_id(self, run_name: str) -> str:
        """
        Thread-safe sequential ID tied to the current run.
        Format: run_20260424_153012_img_0001
        """
        with self._lock:
            self._seq += 1
            return f"{run_name}_img_{self._seq:04d}"

    # ──────────────────────────────────────────────
    # MAIN
    # ──────────────────────────────────────────────
    def process_and_save(self, request_id: str, header: dict, payload: bytes) -> str:
        start_time = time.time()

        run_ctx  = get_run_context()
        run_name = run_ctx["run_name"]
        image_dir = run_ctx["images"]

        # ── Generate stable run-scoped ID ──────────
        image_id = self._next_image_id(run_name)
        filename = f"{image_id}.jpg"
        save_path = os.path.join(image_dir, filename)

        try:
            np_arr = np.frombuffer(payload, np.uint8)
            image  = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if image is None:
                raise ValueError("Invalid image payload — cv2.imdecode returned None")

            # Save raw image to run folder
            cv2.imwrite(save_path, image)

            mode = "RAW"

            if INFERENCE_ENABLED:
                submitted = inference_engine.submit(image, filename, image_id)
                mode = "INFERENCE" if submitted else "DROPPED_QUEUE"

            duration = (time.time() - start_time) * 1000

            self.logger.log_event(
                "IMAGE_PROCESS_SUCCESS",
                "Image stored successfully",
                {
                    "request_id": request_id,
                    "image_id":   image_id,
                    "filename":   filename,
                    "path":       save_path,
                    "mode":       mode,
                    "size":       len(payload),
                    "duration_ms": round(duration, 2),
                }
            )

            return save_path

        except Exception as e:
            self.logger.log_error(
                "IMAGE_PROCESS_ERROR",
                str(e),
                {"request_id": request_id, "image_id": image_id}
            )
            raise
