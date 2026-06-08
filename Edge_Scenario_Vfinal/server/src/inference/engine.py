"""
engine.py
─────────
Background inference queue worker.

Each item in the queue carries:
    (image, filename, image_id, enqueue_time)

enqueue_time is captured at submit() so that queue_wait_ms
(time the image spent waiting before the worker picked it up)
can be recorded in the latency CSV alongside inference latency.
"""

import time
import threading
from queue import Queue, Full

from .inference import process_image
from server.config.config import INFERENCE_QUEUE_SIZE
from server.src.core.metrics_singleton import metrics as _metrics


MAX_QUEUE_SIZE  = INFERENCE_QUEUE_SIZE or 50

_queue          = Queue(maxsize=MAX_QUEUE_SIZE)
_worker_started = False


def start() -> None:
    global _worker_started

    if _worker_started:
        return

    thread = threading.Thread(target=_worker_loop, daemon=True)
    thread.start()
    _worker_started = True


def submit(image, filename: str, image_id: str) -> bool:
    """
    Enqueue one image for inference.
    Stamps enqueue_time so queue_wait_ms can be calculated by the worker.
    Returns False if the queue is full (image is dropped).
    """
    try:
        _queue.put((image, filename, image_id, time.time()), block=False)
        return True
    except Full:
        return False


def _worker_loop() -> None:
    while True:
        try:
            image, filename, image_id, enqueue_time = _queue.get()

            # How long this image waited in the queue before being picked up
            queue_wait_ms = (time.time() - enqueue_time) * 1000.0

            _metrics.mark_inference_start(image_id, queue_wait_ms)

            process_image(image, filename, image_id)

            _metrics.mark_inference_end(image_id)

        except Exception as e:
            print(f"[INFERENCE ERROR] {e}")
