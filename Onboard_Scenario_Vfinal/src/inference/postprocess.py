"""
postprocess.py
──────────────
Decodes raw ONNX output into a list of [x1, y1, x2, y2, conf] bounding boxes
in original (post-resize) image pixel coordinates.

Model output format: (1, 300, 6) — detections per image
  Each row: [x1, y1, x2, y2, conf, class_id]
  Coordinates are absolute pixels at IMG_SIZE (640×640) resolution.

Suppression strategy (two-pass):
  1. Standard IOU-based NMS  — removes overlapping boxes of the same object.
  2. Center-distance NMS     — removes near-duplicate boxes whose centers are
                               within CENTER_DIST_PX pixels of a higher-scoring
                               box.  Needed because tiny detections (small
                               pedestrians) have very low IOU even when they
                               are clearly the same detection shifted by 1-3 px.
"""

import numpy as np
import cv2

from src.utils.config import IMG_SIZE, CONF_THRES, IOU_THRES

# Maximum distance (pixels, at IMG_SIZE resolution) between two box centres
# for the lower-scoring box to be suppressed.
CENTER_DIST_PX = 10


def _iou_nms(boxes_xyxy: list, scores: list) -> list:
    """Standard IOU-based NMS.  Returns indices to keep."""
    if not boxes_xyxy:
        return []
    # cv2.dnn.NMSBoxes wants [x, y, w, h]
    boxes_xywh = [
        [x1, y1, x2 - x1, y2 - y1]
        for x1, y1, x2, y2 in boxes_xyxy
    ]
    indices = cv2.dnn.NMSBoxes(boxes_xywh, scores, CONF_THRES, IOU_THRES)
    if len(indices) == 0:
        return []
    return indices.flatten().tolist()


def _center_dist_nms(detections: list) -> list:
    """
    Suppress near-duplicate boxes by centre proximity.

    detections : list of [x1, y1, x2, y2, conf], sorted high→low conf.
    Returns    : filtered list (same format).
    """
    kept = []
    for det in detections:
        x1, y1, x2, y2, conf = det
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0

        duplicate = False
        for kx1, ky1, kx2, ky2, _ in kept:
            kcx = (kx1 + kx2) / 2.0
            kcy = (ky1 + ky2) / 2.0
            dist = ((cx - kcx) ** 2 + (cy - kcy) ** 2) ** 0.5
            if dist < CENTER_DIST_PX:
                duplicate = True
                break

        if not duplicate:
            kept.append(det)

    return kept


def decode(outputs, frame_shape: tuple) -> list:
    """
    Decode raw ONNX output into filtered, de-duplicated detections.

    Parameters
    ----------
    outputs     : raw output from onnxruntime session.run()
    frame_shape : (H, W[, C]) of the frame that was fed to the model

    Returns
    -------
    list of [x1, y1, x2, y2, conf] in frame pixel coordinates
    """
    preds = np.squeeze(outputs[0])

    # Ensure 2D shape
    if preds.ndim == 1:
        preds = preds[np.newaxis, :]

    h, w = frame_shape[:2]
    scale_x = w / IMG_SIZE
    scale_y = h / IMG_SIZE

    # ── Step 1: confidence filter ─────────────────────────────────────────────
    boxes_xyxy = []
    scores     = []

    for pred in preds:
        if len(pred) < 5:
            continue
        conf = float(pred[4])
        if conf < CONF_THRES:
            continue
        boxes_xyxy.append(pred[:4].tolist())
        scores.append(conf)

    if not boxes_xyxy:
        return []

    # ── Step 2: IOU-based NMS ─────────────────────────────────────────────────
    keep_idx = _iou_nms(boxes_xyxy, scores)

    # ── Step 3: scale to frame coordinates, sort by confidence ───────────────
    detections = []
    for i in keep_idx:
        x1, y1, x2, y2 = boxes_xyxy[i]
        conf = scores[i]

        x1 = int(x1 * scale_x)
        y1 = int(y1 * scale_y)
        x2 = int(x2 * scale_x)
        y2 = int(y2 * scale_y)

        if x2 <= x1 or y2 <= y1:
            continue

        detections.append([x1, y1, x2, y2, conf])

    detections.sort(key=lambda d: d[4], reverse=True)

    # ── Step 4: centre-distance NMS (removes near-pixel-duplicate boxes) ──────
    detections = _center_dist_nms(detections)

    return detections
