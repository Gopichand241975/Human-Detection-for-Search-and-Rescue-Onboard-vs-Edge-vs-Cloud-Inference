# client/config/config.py

import os
from dotenv import load_dotenv

load_dotenv()

# =========================
# SERVER CONFIG
# =========================
SERVER_HOST = os.getenv("CLIENT_HOST", "127.0.0.1")
SERVER_PORT = int(os.getenv("SERVER_PORT", 5000))

# =========================
# AUTH
# =========================
TOKEN = os.getenv("SERVER_TOKEN", "my_secret")

# =========================
# CLIENT CONFIG
# =========================
IMAGE_PATH = os.getenv("IMAGE_PATH", "client/data")

# =========================
# PREPROCESSING SETTINGS
# Image is resized ONCE here at the client before compression/sending.
# The server never re-sizes received images arbitrarily — the inference
# pipeline reads INFERENCE_IMG_WIDTH/HEIGHT from server/config/config.py
# and resizes only inside the model pre-processing step (predictor.py).
#
# Default 640×640 matches the YOLO input expected by the server model so
# no additional resize is needed inside the inference pre-processing step.
# Change via env vars PREPROCESS_WIDTH / PREPROCESS_HEIGHT.
# =========================
PREPROCESS_WIDTH  = int(os.getenv("PREPROCESS_WIDTH",  640))
PREPROCESS_HEIGHT = int(os.getenv("PREPROCESS_HEIGHT", 640))
IMG_QUALITY       = int(os.getenv("IMG_QUALITY", 80))

# Backward-compatible aliases so any existing code that imported the old
# names continues to work without changes.
IMG_WIDTH  = PREPROCESS_WIDTH
IMG_HEIGHT = PREPROCESS_HEIGHT