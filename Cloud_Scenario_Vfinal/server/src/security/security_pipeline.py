import hashlib

from .exceptions import SecurityError


class SecurityPipeline:
    def __init__(self, token, max_size):
        self.token = token
        self.max_size = int(max_size)

    def validate(self, header, payload):

        msg_type = header.get("type")

        # =========================
        # COMMON VALIDATIONS (ALL TYPES)
        # =========================
        self._validate_token(header.get("token"))

        # =========================
        # TYPE-SPECIFIC VALIDATION
        # =========================
        if msg_type == "IMAGE":
            self._validate_size(header.get("payload_size"))
            self._validate_checksum(header.get("checksum"), payload)
            self._validate_filename(header.get("filename"))

        elif msg_type == "TEXT":
            # TEXT messages: enforce a strict size cap (1KB) and checksum
            payload_size = header.get("payload_size", 0)
            if payload_size > 1024:
                raise SecurityError("SIZE_EXCEEDED")
            self._validate_checksum(header.get("checksum"), payload)

        else:
            # Unknown type: still validate size and checksum
            self._validate_size(header.get("payload_size"))
            self._validate_checksum(header.get("checksum"), payload)

    def _validate_token(self, token):
        if token != self.token:
            raise SecurityError("INVALID_TOKEN")

    def _validate_size(self, size):
        if size is None or size > self.max_size:
            raise SecurityError("SIZE_EXCEEDED")

    def _validate_filename(self, filename):
        if not filename or ".." in filename or "/" in filename or "\\" in filename:
            raise SecurityError("INVALID_FILENAME")

    def _validate_checksum(self, checksum, payload):
        # Allow empty payloads (e.g. AUTH messages with no payload)
        if not checksum:
            return
        if not isinstance(payload, (bytes, bytearray)):
            raise SecurityError("CORRUPTED")
        if hashlib.sha256(payload).hexdigest() != checksum:
            raise SecurityError("CORRUPTED")
