"""
prediction_storage.py
─────────────────────
Saves one JSON prediction file per image under:
    data/predictions/<run_id>/<image_id>.json

JSON structure:
    {
        "image_id":          "run_20260426_103000_img_0001",
        "model":             "best",
        "inference_time_ms": 34.5,
        "detections": [
            {
                "class_id":   0,
                "confidence": 0.91,
                "bbox": {"x": 120.0, "y": 80.0, "width": 60.0, "height": 40.0}
            }
        ]
    }

bbox is stored as absolute top-left (x, y, width, height)
in the original (pre-resize) image coordinate space.
"""

import os
import json

from src.utils.config import MODEL_PATH

_MODEL_NAME = os.path.splitext(os.path.basename(MODEL_PATH))[0]


def save_prediction(
    predictions_dir: str,
    image_id: str,
    detections: list,
    inference_time_ms: float,
) -> None:
    """
    Persist prediction for one image.

    Parameters
    ----------
    predictions_dir   : run-scoped folder from RunContext.predictions_dir
    image_id          : e.g. "run_20260426_103000_img_0001"
    detections        : list of [x1, y1, x2, y2, conf] in original image space
    inference_time_ms : inference wall-clock time in milliseconds
    """
    formatted = []
    for det in detections:
        x1, y1, x2, y2, conf = det
        formatted.append({
            "class_id":   0,
            "confidence": round(float(conf), 4),
            "bbox": {
                "x":      round(float(x1), 2),
                "y":      round(float(y1), 2),
                "width":  round(float(x2 - x1), 2),
                "height": round(float(y2 - y1), 2),
            },
        })

    payload = {
        "image_id":          image_id,
        "model":             _MODEL_NAME,
        "inference_time_ms": round(float(inference_time_ms), 4),
        "detections":        formatted,
    }

    out_path = os.path.join(predictions_dir, f"{image_id}.json")
    with open(out_path, "w") as f:
        f.write(json.dumps(payload, indent=2))
        f.flush()
