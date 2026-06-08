"""
run_onboard.py
──────────────
Entry point for the onboard inference pipeline.

Pipeline per frame
──────────────────
    preprocess  →  ONNX inference  →  decode  →  save JSON + CSV  →  save annotated image
"""

import sys
import os
import cv2
import glob

sys.path.insert(0, os.path.dirname(__file__))

from src.processing.preprocess       import preprocess
from src.inference.onnx_infer        import ONNXInfer
from src.inference.postprocess       import decode
from src.metrics.latency             import LatencyTracker
from src.storage.run_context         import RunContext
from src.storage.prediction_storage  import save_prediction
from src.storage.latency_storage     import LatencyStorage
from src.utils.config                import IMAGE_DIR, MAX_IMAGES


def draw_detections(frame, detections: list):
    """Draw bounding boxes and confidence scores onto frame in-place."""
    for x1, y1, x2, y2, conf in detections:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"{conf:.2f}",
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )


def collect_image_paths() -> list:
    all_paths = sorted(glob.glob(os.path.join(IMAGE_DIR, "*.*")))
    paths     = all_paths[:MAX_IMAGES] if MAX_IMAGES is not None else all_paths

    print(f"  Images found    : {len(all_paths)}")
    cap_msg = (
        f" (capped at MAX_IMAGES={MAX_IMAGES})"
        if MAX_IMAGES is not None and len(all_paths) > MAX_IMAGES
        else ""
    )
    print(f"  Images to run   : {len(paths)}{cap_msg}")
    return paths


def main():
    # ── Run context ──────────────────────────────────────────────────────────
    ctx = RunContext()
    print(f"[START] {ctx.run_id}")
    print(f"  Predictions     : {ctx.predictions_dir}")
    print(f"  Inference output: {ctx.inference_output_dir}")
    print(f"  Latency         : {ctx.latency_path}")

    # ── Initialise ───────────────────────────────────────────────────────────
    infer           = ONNXInfer()
    latency         = LatencyTracker()
    latency_storage = LatencyStorage(ctx.latency_path)

    # ── Image queue ──────────────────────────────────────────────────────────
    image_paths = collect_image_paths()
    if not image_paths:
        print("No images found in folder:", IMAGE_DIR)
        latency_storage.close()
        return

    # ── Prime the model generator ────────────────────────────────────────────
    runner = infer.run_generator()
    next(runner)

    frame_id     = 1
    frames_saved = 0

    for img_path in image_paths:
        image_id = f"{ctx.run_id}_img_{frame_id:04d}"

        # ── Preprocess ───────────────────────────────────────────────────────
        start = latency.start()
        input_tensor, frame = preprocess(img_path)
        preprocess_ms = latency.stop(start)

        frame_id += 1

        if input_tensor is None:
            # Too blurry — skip
            continue

        # ── Inference ────────────────────────────────────────────────────────
        start        = latency.start()
        outputs      = runner.send(input_tensor)
        inference_ms = latency.stop(start)

        # ── Decode ───────────────────────────────────────────────────────────
        detections = decode(outputs, frame.shape)

        # ── Persist results ──────────────────────────────────────────────────
        save_prediction(
            predictions_dir=ctx.predictions_dir,
            image_id=image_id,
            detections=detections,
            inference_time_ms=inference_ms,
        )
        latency_storage.log(image_id, preprocess_ms, inference_ms)

        # ── Annotate and save image ───────────────────────────────────────────
        draw_detections(frame, detections)
        out_path = os.path.join(ctx.inference_output_dir, os.path.basename(img_path))
        cv2.imwrite(out_path, frame)

        print(
            f"[{frame_id - 1:04d}] {os.path.basename(img_path)} | "
            f"Pre: {preprocess_ms:.1f} ms | "
            f"Inf: {inference_ms:.1f} ms | "
            f"Det: {len(detections)} | "
            f"Saved: {out_path}"
        )
        frames_saved += 1

    # ── Cleanup ──────────────────────────────────────────────────────────────
    runner.close()
    latency_storage.close()
    ctx.mark_complete(
        has_predictions=frames_saved > 0,
        has_latency=frames_saved > 0,
    )
    print(f"\n[DONE] {ctx.run_id} — {frames_saved} frames processed")


if __name__ == "__main__":
    main()
