from client.src.processing.preprocess import preprocess_image


class ImageHandler:

    def __init__(self, width, height, quality):
        self.width = width
        self.height = height
        self.quality = quality

    def prepare(self, path):
        """
        Returns (payload_bytes, preprocess_ms) or (None, 0.0) for blurry frames.
        preprocess_image now returns a tuple; we propagate it unchanged so
        callers can record timing if they want to.
        """
        payload, preprocess_ms = preprocess_image(
            path,
            width=self.width,
            height=self.height,
            quality=self.quality,
        )

        if payload is None:
            print("[HANDLER] Skipping frame")
            return None, 0.0

        return payload, preprocess_ms
