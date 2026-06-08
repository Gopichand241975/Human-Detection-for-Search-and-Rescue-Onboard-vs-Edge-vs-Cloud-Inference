MODEL_PATH = "best.onnx"

# ─── Inference ───────────────────────────────────────────────────────────────
# ONNX model input resolution.  Preprocessing resizes to this size ONCE;
# to_tensor() must NOT resize again.
IMG_SIZE   = 640
CONF_THRES = 0.30
IOU_THRES  = 0.45

# ─── Preprocessing ───────────────────────────────────────────────────────────
BLUR_THRESHOLD = 100.0   # Laplacian variance threshold for blur detection

# ─── Paths ───────────────────────────────────────────────────────────────────
IMAGE_DIR  = "images"
OUTPUT_DIR = "outputs"
DATA_DIR   = "data"

# ─── Run limiter ─────────────────────────────────────────────────────────────
# Maximum images to process per run.  None = process all images in IMAGE_DIR.
MAX_IMAGES = 50
