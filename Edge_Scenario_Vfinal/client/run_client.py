"""
run_client.py
─────────────
Client entry point.
"""

import os
import time

from client.src.network.client_socket import ClientSocket
from client.src.processing.preprocess import preprocess_image
from client.src.core.metrics_recorder_client import ClientMetricsRecorder

from client.config.config import (
    IMAGE_PATH,
    IMG_WIDTH,
    IMG_HEIGHT,
    IMG_QUALITY
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


def _send_images(client, sock, image_paths, metrics, fps_cap=None):
    period = (1.0 / fps_cap) if fps_cap else 0

    for idx, full_path in enumerate(image_paths):
        img_name = os.path.basename(full_path)

        try:
            # preprocess_image now returns (payload, preprocess_ms)
            payload, preprocess_ms = preprocess_image(
                full_path,
                width=IMG_WIDTH,
                height=IMG_HEIGHT,
                quality=IMG_QUALITY
            )

            if payload is None:
                print(f"[CLIENT] Skipped (blurry): {img_name}")
                continue

            image_id = metrics.generate_image_id()

            print(f"[CLIENT] Sending {img_name} as {image_id} ({len(payload)} bytes)")

            frame_start = time.time()

            status, send_time, ack_time = client.send_image(
                sock, payload, image_id
            )

            metrics.record(image_id, send_time, ack_time, preprocess_ms)
            print(f"[CLIENT] STATUS: {status}")

            if period and idx < len(image_paths) - 1:
                elapsed = time.time() - frame_start
                wait = period - elapsed
                if wait > 0:
                    time.sleep(wait)

        except Exception as e:
            print(f"[CLIENT] Failed: {img_name} → {e}")


def _upload_csv(client, sock, metrics):
    csv_path = metrics.log_file

    if not os.path.exists(csv_path):
        print("[CLIENT] No CSV to upload")
        return

    print(f"[CLIENT] Uploading metrics: {os.path.basename(csv_path)}")
    status = client.send_csv(sock, csv_path, metrics.run_id)
    print(f"[CLIENT] CSV upload STATUS: {status}")


def main():
    client = ClientSocket()

    sock = client.connect()

    if client.authenticate(sock) != "SUCCESS":
        print("[CLIENT] Authentication failed")
        sock.close()
        return

    run_id = client.request_run_id(sock)
    if not run_id:
        print("[CLIENT] Failed to get run_id")
        sock.close()
        return

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
            _send_images(client, sock, image_paths[:30], metrics)
            images_sent = True

        elif choice == "2":
            image_paths = _collect_images(IMAGE_PATH)
            if not image_paths:
                print("[CLIENT] No images found")
                continue
            print(f"[CLIENT] Streaming {len(image_paths)} images...")
            _send_images(client, sock, image_paths, metrics, fps_cap=MAX_FPS)
            _upload_csv(client, sock, metrics)
            images_sent = False

        else:
            print("[CLIENT] Invalid command")

    if images_sent:
        _upload_csv(client, sock, metrics)

    sock.close()
    print("[CLIENT] Session closed")


if __name__ == "__main__":
    main()
