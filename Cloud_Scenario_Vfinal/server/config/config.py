import os
import sys
from dotenv import load_dotenv

load_dotenv()

# =========================
# SERVER CONFIG
# =========================
HOST = os.getenv("SERVER_HOST", "0.0.0.0")
PORT = int(os.getenv("SERVER_PORT", 5000))

TOKEN = os.getenv("SERVER_TOKEN")
if not TOKEN:
    TOKEN = "dev_secret"
    print(
        "[WARNING] SERVER_TOKEN is not set. Using insecure default.",
        file=sys.stderr
    )

# =========================
# LIMITS
# =========================
MAX_SIZE = int(os.getenv("MAX_SIZE", 5_000_000))
MAX_INVALID = int(os.getenv("MAX_INVALID", 5))
IDLE_TIMEOUT = int(os.getenv("IDLE_TIMEOUT", 15))

MAX_CONN_PER_IP = int(os.getenv("MAX_CONN_PER_IP", 3))
CONN_WINDOW = int(os.getenv("CONN_WINDOW", 60))
RECONNECT_COOLDOWN = int(os.getenv("RECONNECT_COOLDOWN", 10))

MAX_CONNECTIONS = int(os.getenv("MAX_CONNECTIONS", 1))
RATE_LIMIT = int(os.getenv("RATE_LIMIT", 10))

# =========================
# PATHS
# =========================
BASE_DIR = os.path.dirname(os.path.dirname(__file__))

SAVE_DIR = os.getenv(
    "SAVE_DIR",
    os.path.join(BASE_DIR, "src", "storage", "images")
)

LOG_DIR = os.getenv(
    "LOG_DIR",
    os.path.join(BASE_DIR, "logs")
)

# =========================
# INFERENCE CONFIG
# =========================
INFERENCE_ENABLED = os.getenv("INFERENCE_ENABLED", "false").lower() == "true"

MODEL_PATH = os.getenv(
    "MODEL_PATH",
    os.path.join(BASE_DIR, "src", "model", "best.onnx")
)

INFERENCE_QUEUE_SIZE = int(os.getenv("INFERENCE_QUEUE_SIZE", 50))

# Image size used server-side for inference — single source of truth.
# All resize operations inside the inference pipeline must read from here.
INFERENCE_IMG_WIDTH  = int(os.getenv("INFERENCE_IMG_WIDTH",  640))
INFERENCE_IMG_HEIGHT = int(os.getenv("INFERENCE_IMG_HEIGHT", 640))

# Prefer GPU for inference when available.
# Set USE_GPU=false to force CPU even when a GPU is present.
USE_GPU = os.getenv("USE_GPU", "true").lower() == "true"

# =========================
# DETECTION CONFIG (NEW)
# =========================
CONF_THRESHOLD = float(os.getenv("CONF_THRESHOLD", 0.45))

# Full class map from best.pt — all 10 classes
CLASS_MAP = {
    0: "person",
    1: "person"
}