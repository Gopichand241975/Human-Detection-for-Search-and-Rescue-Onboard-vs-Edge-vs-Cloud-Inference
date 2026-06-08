import json
import struct
import hashlib

# =========================
# MESSAGE TYPES
# =========================
TYPE_IMAGE    = "IMAGE"
TYPE_AUTH     = "AUTH"
TYPE_ACK      = "ACK"
TYPE_TEXT     = "TEXT"
TYPE_RUN_INIT = "RUN_INIT"   # client → server: request server run_id
TYPE_CSV      = "CSV"        # client → server: upload metrics CSV
TYPE_GOODBYE  = "GOODBYE"    # client → server: clean session close signal

# =========================
# ACK STATUS
# =========================
STATUS_SUCCESS = "SUCCESS"
STATUS_ERROR = "ERROR"


# =========================
# CREATE HEADER
# =========================
def create_header(msg_type, filename="", payload=b"", token="", image_id=None):
    """
    Creates a standardized header.
    Backward compatible.
    """

    if payload is None:
        payload = b""

    if not isinstance(payload, (bytes, bytearray)):
        raise TypeError("Payload must be bytes")

    checksum = hashlib.sha256(payload).hexdigest() if payload else ""

    header = {
        "type": msg_type,
        "filename": filename,   # keep for compatibility
        "payload_size": len(payload),
        "checksum": checksum,
        "token": token
    }

    # ✅ NEW FIELD (optional, safe)
    if image_id is not None:
        header["image_id"] = image_id

    return header

# =========================
# SERIALIZE MESSAGE
# =========================
def serialize_message(header, payload=b""):
    """
    Serializes header + payload into:
    [4 bytes header length][header][payload]
    """

    # Ensure payload consistency
    if payload is None:
        payload = b""

    if not isinstance(payload, (bytes, bytearray)):
        raise TypeError("Payload must be bytes")

    header_bytes = json.dumps(header).encode("utf-8")
    header_length = struct.pack("!I", len(header_bytes))

    return header_length + header_bytes + payload


# =========================
# SAFE RECEIVE FUNCTION
# =========================
def recv_exact(sock, size):
    """
    Ensures exact number of bytes are received.
    Prevents partial read issues.
    """

    if size < 0:
        raise ValueError("Invalid size requested")

    data = b""

    while len(data) < size:
        chunk = sock.recv(size - len(data))

        if not chunk:
            raise ConnectionError("Connection lost during recv")

        data += chunk

    return data


# =========================
# RECEIVE MESSAGE
# =========================
def receive_message(sock):
    """
    Receives message safely:
    1. Reads header length
    2. Reads header
    3. Reads payload (RAW BYTES)
    """

    # --- HEADER LENGTH ---
    header_len_bytes = recv_exact(sock, 4)
    header_len = struct.unpack("!I", header_len_bytes)[0]

    if header_len <= 0:
        raise ValueError("Invalid header length")

    # --- HEADER ---
    header_bytes = recv_exact(sock, header_len)

    try:
        header = json.loads(header_bytes.decode("utf-8"))
    except Exception:
        raise ValueError("Invalid header format")

    # --- PAYLOAD ---
    payload_size = header.get("payload_size", 0)

    if payload_size < 0:
        raise ValueError("Invalid payload size")

    payload = recv_exact(sock, payload_size) if payload_size > 0 else b""

    return header, payload


# =========================
# CHECKSUM VERIFICATION
# =========================
def verify_checksum(header, payload):
    """
    Verifies payload integrity.
    Should be called inside server security pipeline.
    """

    expected = header.get("checksum", "")

    if not expected:
        return True  # allow empty payloads

    if not isinstance(payload, (bytes, bytearray)):
        raise ValueError("Invalid payload type for checksum")

    actual = hashlib.sha256(payload).hexdigest()

    if actual != expected:
        raise ValueError("Checksum mismatch")

    return True