import json
from .exceptions import ProtocolError


def safe_parse_header(raw_header):
    try:
        header = json.loads(raw_header)

        required = ["type", "payload_size", "checksum", "token"]

        for field in required:
            if field not in header:
                raise ValueError(f"Missing {field}")

        # ✅ BACKWARD COMPATIBILITY
        if "filename" not in header:
            header["filename"] = ""

        # ✅ NEW: normalize image_id
        # If not provided, fallback to filename
        if "image_id" not in header:
            header["image_id"] = header.get("filename")

        return header

    except Exception:
        raise ProtocolError("PROTOCOL_ERROR")