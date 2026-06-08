"""
inference.py
────────────
Full inference pipeline for one image:

  1. Run ONNX model forward pass (timed)
  2. Filter detections by confidence + class  (postprocess.py)
  3. Save prediction JSON  → data/predictions/<run>/<image_id>.json
  4. Draw bounding boxes on a copy of the original image and save it
       → data/inference_output/<run>/<image_id>.jpg

Raw images are saved by image_service.py before this runs.

Note: detections are already plain dicts after postprocess — no YOLO
      box objects exist in this pipeline anymore.
"""

import os
import time
import cv2

from .model_loader import load_model
from .predictor    import run_inference
from .postprocess  import filter_detections

from server.src.storage.prediction_storage import save_prediction

# Directories set once by init_run_dirs()
_INFERENCE_OUT_DIR: str | None = None


# ──────────────────────────────────────────────────────────────────────
# INIT  (called once from run_server.py after run context is ready)
# ──────────────────────────────────────────────────────────────────────
def init_run_dirs(run_context: dict) -> None:
    global _INFERENCE_OUT_DIR

    _INFERENCE_OUT_DIR = run_context["inference_output"]

    from server.src.storage.prediction_storage import init_predictions_dir
    init_predictions_dir(run_context["predictions"])

    os.makedirs(_INFERENCE_OUT_DIR, exist_ok=True)


# ──────────────────────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ──────────────────────────────────────────────────────────────────────
def process_image(image, filename: str, image_id: str) -> None:
    """
    Run inference on one decoded image.

    Parameters
    ----------
    image     : BGR numpy array decoded from client payload
    filename  : used as the output filename for the annotated image
    image_id  : stable run-scoped ID  e.g. run_20260424_153012_img_0001
                flows into: latency CSV · prediction JSON · annotated filename
    """
    if _INFERENCE_OUT_DIR is None:
        raise RuntimeError("Inference not initialized — call init_run_dirs() first")

    if image is None:
        print(f"[ERROR] process_image received None image for {image_id}")
        return

    if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
        filename += ".jpg"

    # ── Model inference (timed) ────────────────────────────────────────
    session = load_model()

    t_start           = time.time()
    onnx_outputs      = run_inference(session, image)
    t_end             = time.time()
    inference_time_ms = (t_end - t_start) * 1000.0

    # ── Post-process: returns list of detection dicts ──────────────────
    detections = filter_detections(onnx_outputs)

    # ── Save prediction JSON ───────────────────────────────────────────
    try:
        save_prediction(image_id, detections, inference_time_ms)
    except Exception as e:
        print(f"[PREDICTION ERROR] {image_id}: {e}")

    # ── Draw bounding boxes on original-resolution image ──────────────
    # bbox coords are in inference-image space; scale back to original dims
    orig_h, orig_w = image.shape[:2]

    from server.config.config import INFERENCE_IMG_WIDTH, INFERENCE_IMG_HEIGHT
    scale_x = orig_w / INFERENCE_IMG_WIDTH
    scale_y = orig_h / INFERENCE_IMG_HEIGHT

    annotated = image.copy()

    for det in detections:
        try:
            x1, y1, x2, y2 = det["bbox"]
            x1 = int(x1 * scale_x)
            y1 = int(y1 * scale_y)
            x2 = int(x2 * scale_x)
            y2 = int(y2 * scale_y)
            label = f"{det['class_name']} {det['confidence']:.2f}"
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                annotated, label, (x1, max(y1 - 5, 0)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1,
            )
        except Exception:
            continue

    # ── Save annotated image ───────────────────────────────────────────
    out_path = os.path.join(_INFERENCE_OUT_DIR, filename)
    success  = cv2.imwrite(out_path, annotated)

    if success:
        print(f"[INFERENCE] {image_id} → {out_path}  "
              f"({len(detections)} det, {inference_time_ms:.1f} ms)")
    else:
        print(f"[INFERENCE ERROR] Could not write annotated image: {out_path}")
