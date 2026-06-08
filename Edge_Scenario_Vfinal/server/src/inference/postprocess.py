"""
postprocess.py
──────────────
Parses raw ONNX Runtime output into filtered detection dicts.

Model output format: (1, 300, 6) — post-NMS detections
──────────────────────────────────────────────────────
Each row is one detection:
    col 0-3 : x1, y1, x2, y2  (absolute pixels at INFERENCE_IMG_WIDTH × INFERENCE_IMG_HEIGHT)
    col 4   : confidence score  (already final — no objectness × class multiplication needed)
    col 5   : class id

No transpose is needed — the model already outputs one detection per row.

CONF_THRESHOLD and CLASS_MAP are read from server/config/config.py (single source of truth).
"""

import numpy as np

from server.config.config import (
    CONF_THRESHOLD,
    CLASS_MAP,
)


def filter_detections(onnx_outputs: list, conf_threshold: float = None) -> list:
    """
    Parse raw ONNX outputs and return a list of detection dicts.

    Each dict contains:
        class_id   : int
        class_name : str  (from CLASS_MAP)
        confidence : float
        bbox       : [x1, y1, x2, y2]  in inference-image pixel coords

    Parameters
    ----------
    onnx_outputs   : list of np.ndarray returned by predictor.run_inference
    conf_threshold : override; uses CONF_THRESHOLD from config if None
    """
    if conf_threshold is None:
        conf_threshold = CONF_THRESHOLD

    valid_classes = set(CLASS_MAP.keys())
    detections    = []

    # outputs[0]: (1, 300, 6) — squeeze batch dim → (300, 6)
    # Each row: [x1, y1, x2, y2, conf, class_id]
    raw = np.squeeze(onnx_outputs[0], axis=0)  # (300, 6)

    for row in raw:
        conf     = float(row[4])
        class_id = int(row[5])

        if conf < conf_threshold:
            continue
        if class_id not in valid_classes:
            continue

        x1, y1, x2, y2 = float(row[0]), float(row[1]), float(row[2]), float(row[3])

        detections.append({
            "class_id":   class_id,
            "class_name": CLASS_MAP[class_id],
            "confidence": round(conf, 4),
            "bbox":       [x1, y1, x2, y2],
        })

    return detections


# Backward-compatibility alias
def filter_person_detections(onnx_outputs: list, conf_threshold: float = None) -> list:
    return filter_detections(onnx_outputs, conf_threshold)
