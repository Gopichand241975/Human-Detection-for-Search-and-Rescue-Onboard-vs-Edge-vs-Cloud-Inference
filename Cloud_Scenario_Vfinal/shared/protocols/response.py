import json

# =========================
# SERVER → CLIENT RESPONSE
# =========================
def create_response(status, message=""):
    return json.dumps({
        "status": status,
        "message": message
    }).encode()


# =========================
# CLIENT → PARSE RESPONSE
# =========================
def parse_response(data):
    try:
        return json.loads(data.decode())
    except Exception:
        return {
            "status": "ERROR",
            "message": "Invalid response format"
        }