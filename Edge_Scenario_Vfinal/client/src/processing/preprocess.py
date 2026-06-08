import time
import cv2
from client.config.config import PREPROCESS_WIDTH, PREPROCESS_HEIGHT, IMG_QUALITY

try:
    from client.config.config import BLUR_THRESHOLD
except ImportError:
    BLUR_THRESHOLD = 100.0


def load_image(path):
    img = cv2.imread(path)
    if img is None:
        raise Exception("Image not found: " + path)
    return img


def resize_image(img, width, height):
    """Resize image once during preprocessing. No further resize should occur before the server
    receives the image. The server-side inference pipeline will resize independently for the model."""
    return cv2.resize(img, (width, height))


def compute_blur_variance(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return lap.var()


def conditional_denoise(img, variance, threshold):
    if variance < threshold:
        return cv2.GaussianBlur(img, (3, 3), 0)
    return img


def compress_image(img, quality):
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    success, encoded = cv2.imencode(".jpg", img, encode_param)
    if not success:
        raise Exception("Compression failed")
    return encoded


def image_to_bytes(encoded):
    return encoded.tobytes()


def preprocess_image(path, width=None, height=None, quality=None):
    """
    Resize → optional denoise → compress → bytes.

    Resizing happens exactly ONCE here (client side, before sending).
    No normalization is applied — that step belongs exclusively inside
    the server-side inference pipeline (server/src/inference/predictor.py)
    immediately before the model forward pass.

    Returns (payload_bytes, preprocess_ms) on success.
    Returns (None, 0.0) if the frame is too blurry to process.
    """
    width   = width   if width   is not None else PREPROCESS_WIDTH
    height  = height  if height  is not None else PREPROCESS_HEIGHT
    quality = quality if quality is not None else IMG_QUALITY

    t_start = time.perf_counter()

    img = load_image(path)
    img = resize_image(img, width, height)  # single resize — client side only

    variance = compute_blur_variance(img)
    img = conditional_denoise(img, variance, BLUR_THRESHOLD)

    if variance < (BLUR_THRESHOLD * 0.5):
        print(f"[PREPROCESS] Skipping very blurry image (var={variance:.2f})")
        return None, 0.0

    encoded = compress_image(img, quality)
    payload = image_to_bytes(encoded)

    preprocess_ms = (time.perf_counter() - t_start) * 1000.0

    print(f"[PREPROCESS] Variance: {variance:.2f} | Q={quality} | {preprocess_ms:.2f} ms")

    return payload, preprocess_ms