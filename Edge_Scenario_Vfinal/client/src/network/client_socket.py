import time
import socket

from client.config.config import SERVER_HOST, SERVER_PORT, TOKEN

from shared.protocols.protocol import (
    create_header,
    serialize_message,
    receive_message,
    TYPE_AUTH,
    TYPE_IMAGE,
    TYPE_TEXT,
    TYPE_RUN_INIT,
    TYPE_CSV
)


class ClientSocket:
    def __init__(self):
        self.host = SERVER_HOST
        self.port = SERVER_PORT
        self.token = TOKEN

    # =========================
    # CONNECT
    # =========================
    def connect(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((self.host, self.port))
        sock.settimeout(10)

        print("[CLIENT] Connected to server")
        return sock

    # =========================
    # AUTH
    # =========================
    def authenticate(self, sock):
        header = create_header(
            msg_type=TYPE_AUTH,
            token=self.token
        )

        sock.sendall(serialize_message(header))

        response, _ = receive_message(sock)
        status = response.get("status")

        print(f"[CLIENT] AUTH STATUS: {status}")
        return status

    # =========================
    # SEND IMAGE
    # Returns (status, send_time, ack_time) so the caller
    # can record accurate RTT timestamps.
    # =========================
    def send_image(self, sock, image_bytes, image_id):
        header = create_header(
            msg_type=TYPE_IMAGE,
            filename=image_id,
            payload=image_bytes,
            token=self.token,
            image_id=image_id
        )

        message = serialize_message(header, image_bytes)

        send_time = time.time()          # record JUST before bytes go out
        sock.sendall(message)

        response, _ = receive_message(sock)
        ack_time = time.time()           # record IMMEDIATELY when ACK arrives

        status = response.get("status")

        return status, send_time, ack_time

    # =========================
    # SEND TEXT
    # =========================
    def send_text(self, sock, message):
        payload = message.encode("utf-8")

        # safety limit (temporary feature)
        if len(payload) > 1024:
            print("[CLIENT] Message too large (limit 1KB)")
            return "ERROR"

        header = create_header(
            msg_type=TYPE_TEXT,
            payload=payload,
            token=self.token
        )

        msg = serialize_message(header, payload)
        sock.sendall(msg)

        response, _ = receive_message(sock)
        status = response.get("status")

        return status

    # =========================
    # REQUEST RUN ID
    # Ask server for its authoritative run_id so client metrics
    # can be linked to the same run on the server side.
    # =========================
    def request_run_id(self, sock):
        header = create_header(
            msg_type=TYPE_RUN_INIT,
            token=self.token
        )
        sock.sendall(serialize_message(header))
        response, _ = receive_message(sock)
        return response.get("run_id")

    # =========================
    # SEND CSV
    # Upload the per-run network-delay CSV to the server.
    # =========================
    def send_csv(self, sock, csv_path, run_id):
        with open(csv_path, "rb") as f:
            csv_bytes = f.read()

        import os
        filename = os.path.basename(csv_path)

        header = create_header(
            msg_type=TYPE_CSV,
            filename=filename,
            payload=csv_bytes,
            token=self.token,
            image_id=run_id
        )
        sock.sendall(serialize_message(header, csv_bytes))
        response, _ = receive_message(sock)
        return response.get("status")

    # =========================
    # IS ALIVE
    # =========================
    def is_alive(self, sock):
        try:
            sock.send(b"")
            return True
        except Exception:
            return False
