import os
import threading

_lock = threading.Lock()
LABELS_DIR = None


def set_labels_dir(path):
    global LABELS_DIR
    with _lock:
        LABELS_DIR = path
        os.makedirs(LABELS_DIR, exist_ok=True)


def _xyxy_to_yolo(bbox, img_w, img_h):
    x1, y1, x2, y2 = bbox

    x_center = ((x1 + x2) / 2) / img_w
    y_center = ((y1 + y2) / 2) / img_h
    width = (x2 - x1) / img_w
    height = (y2 - y1) / img_h

    return x_center, y_center, width, height


def _clamp(v):
    return max(0.0, min(1.0, v))


def save_labels(image_name, detections, image_shape):
    with _lock:
        current_dir = LABELS_DIR

    if current_dir is None:
        raise RuntimeError("LABELS_DIR not initialized — call set_labels_dir() first")

    label_name = image_name.rsplit(".", 1)[0] + ".txt"
    label_path = os.path.join(current_dir, label_name)

    img_h, img_w = image_shape[:2]

    with open(label_path, "w") as f:
        if not detections:
            return

        for det in detections:
            cls = det["class_id"]
            x1, y1, x2, y2 = det["bbox"]

            x, y, w, h = _xyxy_to_yolo([x1, y1, x2, y2], img_w, img_h)

            # noise filter — skip implausibly small boxes
            if w < 0.02 or h < 0.02:
                continue

            x = _clamp(x)
            y = _clamp(y)
            w = _clamp(w)
            h = _clamp(h)

            f.write(f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")