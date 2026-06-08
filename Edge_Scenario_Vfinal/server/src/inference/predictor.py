"""
predictor.py
────────────
Runs a single image through the ONNX Runtime session.

Pre-processing (all sizes from server/config/config.py — single source of truth):
  1. Resize to (INFERENCE_IMG_WIDTH, INFERENCE_IMG_HEIGHT)
  2. BGR → RGB
  3. Normalize pixel values to [0.0, 1.0]  (divide by 255)
  4. HWC → NCHW float32 tensor

Returns raw ONNX output arrays for postprocess.py to parse.
"""

import numpy as np
import cv2

from server.config.config import INFERENCE_IMG_WIDTH, INFERENCE_IMG_HEIGHT


def _preprocess(image: np.ndarray) -> np.ndarray:
    """
    Convert a BGR numpy image to a normalised NCHW float32 tensor.

    Resize target comes exclusively from server/config/config.py so there is
    a single source of truth for inference image dimensions on the server side.

    Returns
    -------
    np.ndarray  shape (1, 3, H, W), dtype float32, values in [0.0, 1.0]
    """
    # 1. Resize — dimensions from server config only
    resized = cv2.resize(
        image,
        (INFERENCE_IMG_WIDTH, INFERENCE_IMG_HEIGHT),
        interpolation=cv2.INTER_LINEAR,
    )

    # 2. BGR → RGB
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    # 3. Normalize [0, 255] → [0.0, 1.0]
    normalized = rgb.astype(np.float32) / 255.0

    # 4. HWC → CHW → NCHW batch dimension
    tensor = np.transpose(normalized, (2, 0, 1))[np.newaxis, ...]  # (1, 3, H, W)
    return tensor


def run_inference(session, image: np.ndarray) -> list:
    """
    Run ONNX Runtime inference on a single BGR image.

    Parameters
    ----------
    session : ort.InferenceSession   (from model_loader.load_model)
    image   : BGR numpy array (H, W, 3)

    Returns
    -------
    list of np.ndarray — raw output arrays from the ONNX session.
    Typically outputs[0] has shape (1, num_detections, 6) for YOLOv8 ONNX.
    """
    tensor     = _preprocess(image)
    input_name = session.get_inputs()[0].name
    outputs    = session.run(None, {input_name: tensor})
    return outputs
