"""
prediction_storage.py
─────────────────────
Saves one JSON prediction file per image under:
    data/predictions/<run_name>/<image_id>.json

JSON structure (absolute top-left bbox):
    {
        "image_id":          "run_20260424_153012_img_0001",
        "model":             "yolov8n",
        "inference_time_ms": 34.5,
        "detections": [
            {
                "class_id":   2,
                "confidence": 0.91,
                "bbox": {"x": 120, "y": 80, "width": 60, "height": 40}
            }
        ]
    }

bbox is stored as absolute top-left (x, y, width, height),
converted from the internal xyxy format used by YOLO.
"""

import os
import json

from server.config.config import MODEL_PATH

# Derive a short model name from the model file path (e.g. "best")
_MODEL_NAME = os.path.splitext(os.path.basename(MODEL_PATH))[0]

# Set at server startup via init_predictions_dir()
_PREDICTIONS_DIR: str | None = None


# ──────────────────────────────────────────────
# INIT
# ──────────────────────────────────────────────
def init_predictions_dir(path: str) -> None:
    """Call once during server startup with run_context['predictions']."""
    global _PREDICTIONS_DIR
    _PREDICTIONS_DIR = path
    os.makedirs(_PREDICTIONS_DIR, exist_ok=True)


# ──────────────────────────────────────────────
# INTERNAL HELPERS
# ──────────────────────────────────────────────
def _xyxy_to_xywh(x1: float, y1: float, x2: float, y2: float) -> dict:
    """Convert YOLO xyxy → absolute top-left x, y, w, h."""
    return {
        "x":      round(x1, 2),
        "y":      round(y1, 2),
        "width":  round(x2 - x1, 2),
        "height": round(y2 - y1, 2),
    }


# ──────────────────────────────────────────────
# PUBLIC API
# ──────────────────────────────────────────────
def save_prediction(image_id: str, detections: list, inference_time_ms: float) -> None:
    """
    Persist prediction for one image.

    Parameters
    ----------
    image_id          : e.g. "run_20260424_153012_img_0001"
    detections        : list of dicts with keys class_id, confidence, bbox [x1,y1,x2,y2]
    inference_time_ms : wall-clock latency in milliseconds
    """
    if _PREDICTIONS_DIR is None:
        raise RuntimeError("Predictions dir not initialized — call init_predictions_dir() first")

    formatted_detections = []
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        formatted_detections.append({
            "class_id":   det["class_id"],
            "confidence": round(det["confidence"], 4),
            "bbox":       _xyxy_to_xywh(x1, y1, x2, y2),
        })

    payload = {
        "image_id":          image_id,
        "model":             _MODEL_NAME,
        "inference_time_ms": round(inference_time_ms, 4),
        "detections":        formatted_detections,
    }

    out_path = os.path.join(_PREDICTIONS_DIR, f"{image_id}.json")

    # Atomic-style write: build full string first, then write in one call
    json_str = json.dumps(payload, indent=2)
    with open(out_path, "w") as f:
        f.write(json_str)
        f.flush()
