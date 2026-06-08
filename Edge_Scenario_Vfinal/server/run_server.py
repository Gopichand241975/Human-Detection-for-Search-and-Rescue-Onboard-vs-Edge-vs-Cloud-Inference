"""
run_server.py
─────────────
Server entry point. Init order matters:

  1. Create global dirs (SAVE_DIR, LOG_DIR)
  2. Create Logger (only needs global log dir, no run context yet)
  3. Create run  → get RUN_CONTEXT
  4. Register run context globally
  5. Bind logger to run  → per-run events.log now available
  6. Init inference dirs (needs run context)
  7. Start socket and accept clients
"""

import os
import time
import atexit
import socket
import threading

from server.config.config import (
    HOST, PORT, TOKEN, MAX_SIZE, IDLE_TIMEOUT, SAVE_DIR, LOG_DIR,
    INFERENCE_ENABLED, MAX_CONNECTIONS, RATE_LIMIT
)

from server.src.network.server_socket import ServerSocket
from server.src.network.safe_socket   import SafeSocket

from server.src.security.security_pipeline  import SecurityPipeline
from server.src.security.connection_manager import ConnectionManager
from server.src.security.exceptions         import SecurityError

from shared.protocols.protocol import (
    receive_message,
    serialize_message,
    TYPE_IMAGE,
    TYPE_TEXT,
    TYPE_RUN_INIT,
    TYPE_CSV
)

from server.src.services.image_service import ImageService
from server.src.logging.logger         import Logger
from server.src.storage.run_manager    import create_new_run
from server.src.inference              import inference
from server.src.core.run_context       import set_run_context
from evaluation.sync_runs import rebuild_runs_json


# ══════════════════════════════════════════════════════════════════════
# 1. GLOBAL DIRS
# ══════════════════════════════════════════════════════════════════════
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(LOG_DIR,  exist_ok=True)

# Directory where client-uploaded network-delay CSVs are stored
CLIENT_METRICS_DIR = os.path.join("data", "client_metrics")
os.makedirs(CLIENT_METRICS_DIR, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════
# 2. LOGGER  (global logs only at this point)
# ══════════════════════════════════════════════════════════════════════
logger = Logger(LOG_DIR)

# ══════════════════════════════════════════════════════════════════════
# 3. CREATE RUN
# ══════════════════════════════════════════════════════════════════════
RUN_CONTEXT = create_new_run()

# ══════════════════════════════════════════════════════════════════════
# 4. REGISTER RUN CONTEXT  (makes get_run_context() work everywhere)
# ══════════════════════════════════════════════════════════════════════
set_run_context(RUN_CONTEXT)

# ══════════════════════════════════════════════════════════════════════
# 5. BIND LOGGER TO RUN  (enables per-run events.log)
# ══════════════════════════════════════════════════════════════════════
logger.bind_run(RUN_CONTEXT)

print(f"[RUN] Run name:         {RUN_CONTEXT['run_name']}")
print(f"[RUN] Raw images:       {RUN_CONTEXT['images']}")
print(f"[RUN] Run logs:         {RUN_CONTEXT['logs']}")
print(f"[RUN] Metrics:          {RUN_CONTEXT['metrics']}")
print(f"[RUN] Predictions:      {RUN_CONTEXT['predictions']}")
print(f"[RUN] Inference output: {RUN_CONTEXT['inference_output']}")

logger.log_server(
    "SERVER_START",
    "Server initialized",
    {"run": RUN_CONTEXT["run_name"], "host": HOST, "port": PORT}
)

# ══════════════════════════════════════════════════════════════════════
# 6.RUN SYNC (STARTUP)
# ══════════════════════════════════════════════════════════════════════
rebuild_runs_json()

# ══════════════════════════════════════════════════════════════════════
# 7. SECURITY + SERVICES
# ══════════════════════════════════════════════════════════════════════
security     = SecurityPipeline(TOKEN, MAX_SIZE)
conn_manager = ConnectionManager(MAX_CONNECTIONS, RATE_LIMIT)
image_service = ImageService(logger)

# ══════════════════════════════════════════════════════════════════════
# 8. INFERENCE DIRS
# ══════════════════════════════════════════════════════════════════════
if INFERENCE_ENABLED:
    inference.init_run_dirs(RUN_CONTEXT)

print(f"[SERVER] Listening on {HOST}:{PORT}...")
server = ServerSocket(HOST, PORT)


# ══════════════════════════════════════════════════════════════════════
# ACK HELPER
# ══════════════════════════════════════════════════════════════════════
def send_ack(sock, status, reason=""):
    ack = {"type": "ACK", "status": status, "reason": reason}
    sock.send(serialize_message(ack))


# ══════════════════════════════════════════════════════════════════════
# CLIENT HANDLER
# ══════════════════════════════════════════════════════════════════════
def handle_client(conn, addr):
    request_id      = None
    start_conn_time = time.time()

    conn_manager.add_connection()
    safe_sock  = SafeSocket(conn, timeout=IDLE_TIMEOUT)
    request_id = logger.new_request_id()

    logger.log_event(
        "CONNECTION_ACCEPTED",
        "Client connected",
        {"request_id": request_id, "addr": addr}
    )
    print(f"[SERVER] Accepted connection from {addr}  [{request_id}]")

    authenticated = False

    try:
        while True:

            # ── Receive ────────────────────────────────────────────────
            try:
                header, payload = receive_message(safe_sock)

            except socket.timeout:
                logger.log_security(
                    "TIMEOUT",
                    "Idle timeout triggered",
                    {"request_id": request_id}
                )
                break

            except Exception as recv_err:
                logger.log_event(
                    "DISCONNECT",
                    "Client disconnected or send invalid message",
                    {"request_id": request_id, "error": str(recv_err)}
                )
                print(f"[SERVER] Recv error: {recv_err}")
                break

            msg_type = header.get("type")

            # ── AUTH ───────────────────────────────────────────────────
            if msg_type == "AUTH":
                if header.get("token") != TOKEN:
                    logger.log_security(
                        "INVALID_TOKEN",
                        "Authentication failed",
                        {"request_id": request_id}
                    )
                    send_ack(safe_sock, "INVALID_TOKEN")
                    continue

                authenticated = True
                logger.log_event(
                    "AUTH_SUCCESS",
                    "Client authenticated",
                    {"request_id": request_id}
                )
                send_ack(safe_sock, "SUCCESS")
                continue

            if not authenticated:
                logger.log_security(
                    "UNAUTHORIZED_ACCESS",
                    "Client attempted action without authentication",
                    {"request_id": request_id}
                )
                send_ack(safe_sock, "NOT_AUTHENTICATED")
                continue

            # ── Security validation ────────────────────────────────────
            try:
                security.validate(header, payload)
            except SecurityError as e:
                logger.log_security(
                    "SECURITY_VALIDATION_FAILED",
                    e.code,
                    {"request_id": request_id}
                )
                send_ack(safe_sock, e.code)
                continue

            # ── RUN_INIT ───────────────────────────────────────────────
            # Client requests the server's run_id so both sides share
            # the same run identifier for metric correlation.
            if msg_type == TYPE_RUN_INIT:
                resp = {
                    "type":    "ACK",
                    "status":  "SUCCESS",
                    "run_id":  RUN_CONTEXT["run_name"],
                    "payload_size": 0
                }
                safe_sock.send(serialize_message(resp))
                logger.log_event(
                    "RUN_INIT",
                    "Client synced run_id",
                    {"request_id": request_id, "run_id": RUN_CONTEXT["run_name"]}
                )
                continue

            # ── CSV UPLOAD ─────────────────────────────────────────────
            # Client sends its per-run network-delay CSV after the last
            # image.  We store it under data/client_metrics/.
            if msg_type == TYPE_CSV:
                try:
                    filename = header.get("filename", "unknown.csv")
                    save_path = os.path.join(CLIENT_METRICS_DIR, filename)
                    with open(save_path, "wb") as f:
                        f.write(payload)
                    logger.log_event(
                        "CSV_RECEIVED",
                        "Client metrics CSV saved",
                        {"request_id": request_id, "file": save_path}
                    )
                    print(f"[SERVER] Client metrics saved: {save_path}")
                    send_ack(safe_sock, "SUCCESS")
                except Exception as e:
                    logger.log_error(
                        "CSV_SAVE_ERROR",
                        str(e),
                        {"request_id": request_id}
                    )
                    send_ack(safe_sock, "CSV_ERROR")
                continue

            # ── IMAGE ──────────────────────────────────────────────────
            if msg_type == TYPE_IMAGE:
                try:
                    t0 = time.time()

                    save_path = image_service.process_and_save(
                        request_id, header, payload
                    )

                    logger.record_processing_time(time.time() - t0)

                    if INFERENCE_ENABLED:
                        print(f"[SERVER] Queued for inference: {save_path}")
                    else:
                        print(f"[SERVER] Saved image: {save_path}")

                    send_ack(safe_sock, "SUCCESS")

                except SecurityError as e:
                    logger.log_security(
                        "IMAGE_PROCESS_FAILED",
                        e.code,
                        {"request_id": request_id}
                    )
                    send_ack(safe_sock, e.code)

                continue

            # ── TEXT ───────────────────────────────────────────────────
            if msg_type == TYPE_TEXT:
                try:
                    message = payload.decode(errors="replace")
                    print(f"[SERVER] TEXT: {repr(message)}")

                    logger.log_event(
                        "TEXT_RECEIVED",
                        "Client message received",
                        {"request_id": request_id, "message": message}
                    )
                    send_ack(safe_sock, "SUCCESS")

                except Exception as e:
                    logger.log_error(
                        "TEXT_PROCESS_ERROR",
                        str(e),
                        {"request_id": request_id}
                    )
                    send_ack(safe_sock, "TEXT_ERROR")

                continue

            # ── Unknown type ───────────────────────────────────────────
            logger.log_error(
                "INVALID_TYPE",
                "Unsupported message type",
                {"request_id": request_id, "type": msg_type}
            )
            send_ack(safe_sock, "INVALID_TYPE")

    except Exception as e:
        logger.log_error(
            "SERVER_ERROR",
            str(e),
            {"request_id": request_id or "unknown"}
        )

    finally:
        conn_manager.remove_connection()

        try:
            safe_sock.close()
        except Exception:
            pass

        duration = time.time() - start_conn_time
        logger.log_event(
            "CONNECTION_CLOSED",
            "Connection cleanup complete",
            {"request_id": request_id, "duration_sec": round(duration, 2)}
        )
        print("[SERVER] Connection closed\n")


# ══════════════════════════════════════════════════════════════════════
# MAIN LOOP
# ══════════════════════════════════════════════════════════════════════
while True:
    conn = server.accept_client()
    addr = conn.getpeername()[0]

    if not conn_manager.can_accept(addr):
        print("[SECURITY] Connection rejected:", addr)
        logger.log_security(
            "CONNECTION_REJECTED",
            "Rate limit / max connections exceeded",
            {"addr": addr}
        )
        try:
            send_ack(conn, "SERVER_BUSY")
        except Exception:
            pass
        conn.close()
        continue

    client_thread = threading.Thread(
        target=handle_client,
        args=(conn, addr),
        daemon=True
    )
    client_thread.start()
