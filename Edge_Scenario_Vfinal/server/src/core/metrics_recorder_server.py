import time
import csv
import os


class ServerMetricsRecorder:
    """
    Writes one latency CSV per run to data/metrics/<run_name>_latency.csv

    CSV columns:
        image_id, inference_start, inference_end, latency, queue_wait_ms
    """

    def __init__(self):
        self.log_file = None
        self.records  = {}      # image_id → {start, end, queue_wait_ms}

    def _ensure_init(self):
        if self.log_file is not None:
            return

        from server.src.core.run_context import get_run_context

        run_ctx     = get_run_context()
        run_name    = run_ctx["run_name"]
        metrics_dir = run_ctx["metrics"]

        self.log_file = os.path.join(metrics_dir, f"{run_name}_latency.csv")

        os.makedirs(metrics_dir, exist_ok=True)

        if not os.path.exists(self.log_file):
            with open(self.log_file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "image_id",
                    "inference_start",
                    "inference_end",
                    "latency",
                    "queue_wait_ms",
                ])

    def mark_inference_start(self, image_id: str, queue_wait_ms: float = 0.0):
        self._ensure_init()
        self.records[image_id] = {
            "start":         time.time(),
            "end":           None,
            "queue_wait_ms": round(queue_wait_ms, 3),
        }

    def mark_inference_end(self, image_id: str):
        self._ensure_init()

        if image_id not in self.records:
            return

        self.records[image_id]["end"] = time.time()
        self._finalize(image_id)

    def _finalize(self, image_id: str):
        rec   = self.records.pop(image_id)
        start = rec["start"]
        end   = rec["end"]

        if end is None:
            return

        latency = end - start
        self._write(image_id, start, end, latency, rec["queue_wait_ms"])

    def _write(self, image_id, start, end, latency, queue_wait_ms):
        with open(self.log_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                image_id,
                start,
                end,
                latency,
                queue_wait_ms,
            ])
            f.flush()
