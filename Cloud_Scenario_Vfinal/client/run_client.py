"""
run_client.py
─────────────
Client entry point.  Uses ConnectionHandler for all network calls
so that every image send and CSV upload automatically retries on
socket failure and reconnects if the link drops mid-run.
"""

import os
import time

from client.src.network.connection_handler import ConnectionHandler
from client.src.processing.preprocess import preprocess_image
from client.src.core.metrics_recorder_client import ClientMetricsRecorder

from client.config.config import (
    IMAGE_PATH,
    IMG_WIDTH,
    IMG_HEIGHT,
    IMG_QUALITY,
)

MAX_FPS = 30


def _collect_images(path):
    if os.path.isdir(path):
        files = sorted(
            f for f in os.listdir(path)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        )
        return [os.path.join(path, f) for f in files]
    return [path]


def _send_images(handler, image_paths, metrics, fps_cap=None):
    period = (1.0 / fps_cap) if fps_cap else 0

    for idx, full_path in enumerate(image_paths):
        img_name = os.path.basename(full_path)

        try:
            payload, preprocess_ms = preprocess_image(
                full_path,
                width=IMG_WIDTH,
                height=IMG_HEIGHT,
                quality=IMG_QUALITY,
            )

            if payload is None:
                print(f"[CLIENT] Skipped (blurry): {img_name}")
                continue

            image_id = metrics.generate_image_id()
            print(f"[CLIENT] Sending {img_name} as {image_id} ({len(payload)} bytes)")

            frame_start = time.time()

            # Retry-aware send — reconnects transparently if socket drops
            status, send_time, ack_time = handler.send_image_with_retry(
                payload, image_id
            )

            metrics.record(image_id, send_time, ack_time, preprocess_ms)
            print(f"[CLIENT] STATUS: {status}")

            if period and idx < len(image_paths) - 1:
                elapsed = time.time() - frame_start
                wait = period - elapsed
                if wait > 0:
                    time.sleep(wait)

        except RuntimeError as e:
            # All retry attempts exhausted for this image — log and continue
            print(f"[CLIENT] DROPPED {img_name}: {e}")


def _upload_csv(handler, metrics):
    csv_path = metrics.log_file

    if not os.path.exists(csv_path):
        print("[CLIENT] No CSV to upload")
        return

    print(f"[CLIENT] Uploading metrics: {os.path.basename(csv_path)}")
    try:
        status = handler.send_csv_with_retry(csv_path, metrics.run_id)
        print(f"[CLIENT] CSV upload STATUS: {status}")
    except RuntimeError as e:
        print(f"[CLIENT] CSV upload failed permanently: {e}")


def main():
    handler = ConnectionHandler()

    try:
        handler.connect_with_retry()
    except ConnectionError as e:
        print(f"[CLIENT] Could not connect to server: {e}")
        return

    run_id = handler.get_run_id()
    print(f"[CLIENT] Synced run ID: {run_id}")

    metrics = ClientMetricsRecorder(run_id=run_id)
    print(f"[CLIENT] Metrics file: {metrics.log_file}")

    print("\n===== CONTINUOUS SESSION MODE =====")
    print("1 → Send one image")
    print("2 → Stream images (≤30 FPS)")
    print("exit → Close session")
    print("====================================")

    images_sent = False

    while True:
        choice = input("\nEnter command: ").strip()

        if choice.lower() == "exit":
            print("[CLIENT] Closing session...")
            break

        elif choice == "1":
            image_paths = _collect_images(IMAGE_PATH)
            if not image_paths:
                print("[CLIENT] No images found")
                continue
            _send_images(handler, image_paths[:30], metrics)
            images_sent = True

        elif choice == "2":
            image_paths = _collect_images(IMAGE_PATH)
            if not image_paths:
                print("[CLIENT] No images found")
                continue
            print(f"[CLIENT] Streaming {len(image_paths)} images...")
            _send_images(handler, image_paths, metrics, fps_cap=MAX_FPS)
            _upload_csv(handler, metrics)
            images_sent = False

        else:
            print("[CLIENT] Invalid command")

    if images_sent:
        _upload_csv(handler, metrics)

    handler.close()


if __name__ == "__main__":
    main()
