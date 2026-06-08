"""
preprocess.py
─────────────
Image preprocessing pipeline before ONNX inference:

    1. Load image from disk
    2. Resize ONCE to IMG_SIZE × IMG_SIZE  (the model's input resolution)
    3. Blur detection via Laplacian variance
    4. Conditional denoise — GaussianBlur if variance < BLUR_THRESHOLD
    5. Skip frame         — returns (None, None) if variance < BLUR_THRESHOLD × 0.5
    6. Normalise + transpose into ONNX input tensor  (1, 3, IMG_SIZE, IMG_SIZE)

NOTE: the image is resized exactly once, in step 2.  to_tensor() receives an
already-resized frame and must NOT resize it again.
"""

import cv2
import numpy as np

from src.utils.config import IMG_SIZE, BLUR_THRESHOLD


# ──────────────────────────────────────────────────────────────────────────────
# Step helpers
# ──────────────────────────────────────────────────────────────────────────────

def load_image(path: str):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {path}")
    return img


def resize_image(img):
    """Resize to the model's input resolution (IMG_SIZE × IMG_SIZE)."""
    return cv2.resize(img, (IMG_SIZE, IMG_SIZE))


def compute_blur_variance(img) -> float:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def conditional_denoise(img, variance: float):
    if variance < BLUR_THRESHOLD:
        return cv2.GaussianBlur(img, (3, 3), 0)
    return img


def to_tensor(img) -> np.ndarray:
    """
    Convert an already-resized HxWxC BGR uint8 image to a
    1×3×H×W float32 tensor for ONNX.

    The image MUST already be at IMG_SIZE × IMG_SIZE — no resize here.
    """
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_f   = img_rgb / 255.0
    img_chw = np.transpose(img_f, (2, 0, 1))
    return np.expand_dims(img_chw, axis=0).astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def preprocess(path: str):
    """
    Full preprocessing pipeline.

    Parameters
    ----------
    path : path to the source image file

    Returns
    -------
    (tensor, resized_frame)  on success  — tensor is ready for ONNX inference;
                                           resized_frame is the BGR uint8 image
                                           for drawing boxes later.
    (None, None)             if the frame is too blurry to process.
    """
    img = load_image(path)

    # Step 1: resize ONCE to model input resolution
    img = resize_image(img)

    # Step 2: blur detection (on the resized frame)
    variance = compute_blur_variance(img)

    # Step 3: skip extremely blurry frames
    if variance < BLUR_THRESHOLD * 0.5:
        print(f"[PREPROCESS] Skipping very blurry image (var={variance:.2f}): {path}")
        return None, None

    # Step 4: conditional denoise
    img = conditional_denoise(img, variance)

    print(f"[PREPROCESS] var={variance:.2f} | {path}")

    # Step 5: build ONNX input tensor (no resize inside)
    tensor = to_tensor(img)

    # Return tensor and the (single-)resized BGR frame for box drawing
    return tensor, img
