"""
connection_handler.py
─────────────────────
Wraps ClientSocket with retry + reconnection logic.

Responsibilities
────────────────
- connect_with_retry()   : establish TCP connection + auth with exponential
                           backoff; raises after MAX_CONNECT_RETRIES attempts.
- send_image_with_retry(): send one image; if the socket dies mid-send,
                           reconnect + re-auth transparently and retry the
                           same image up to MAX_SEND_RETRIES times.
- send_csv_with_retry()  : same pattern for CSV upload.

Design notes
────────────
* run_id is obtained ONCE at the start of a run.  On reconnect we do NOT
  re-issue RUN_INIT — the run_id the caller already holds stays valid;
  the server stores images under run folders on disk so re-sends would
  create duplicates if we started a new run.
* Backoff is capped at MAX_BACKOFF_S to avoid very long waits on a Pi
  with a flaky Tailscale link.
* All retry counts are configurable via env vars so they can be tightened
  in tests without monkey-patching sleep.
"""

import os
import time
import socket

from client.src.network.client_socket import ClientSocket

# ── tunables (overridable via env for tests) ──────────────────────────
MAX_CONNECT_RETRIES = int(os.getenv("MAX_CONNECT_RETRIES", 5))
MAX_SEND_RETRIES    = int(os.getenv("MAX_SEND_RETRIES",    3))
BASE_BACKOFF_S      = float(os.getenv("BASE_BACKOFF_S",    1.0))
MAX_BACKOFF_S       = float(os.getenv("MAX_BACKOFF_S",     30.0))


def _backoff(attempt: int) -> float:
    """Exponential backoff capped at MAX_BACKOFF_S."""
    return min(BASE_BACKOFF_S * (2 ** attempt), MAX_BACKOFF_S)


class ConnectionHandler:
    """
    High-level client connection manager with retry and reconnect.

    Usage
    ─────
        handler = ConnectionHandler()
        handler.connect_with_retry()          # connect + auth
        run_id = handler.get_run_id()         # sync run_id once
        handler.send_image_with_retry(payload, image_id)
        handler.send_csv_with_retry(csv_path, run_id)
        handler.close()
    """

    def __init__(self):
        self._client = ClientSocket()
        self._sock   = None

    # ──────────────────────────────────────────────────────────────────
    # CONNECT + AUTH  (with retry)
    # ──────────────────────────────────────────────────────────────────

    def connect_with_retry(self) -> None:
        """
        Try to connect and authenticate up to MAX_CONNECT_RETRIES times.
        Raises ConnectionError if all attempts fail.
        """
        last_error = None

        for attempt in range(MAX_CONNECT_RETRIES):
            try:
                self._sock = self._client.connect()
                status = self._client.authenticate(self._sock)

                if status != "SUCCESS":
                    raise ConnectionError(f"Auth rejected: {status}")

                print(f"[RETRY] Connected (attempt {attempt + 1})")
                return

            except Exception as e:
                last_error = e
                wait = _backoff(attempt)
                print(
                    f"[RETRY] Connect attempt {attempt + 1}/{MAX_CONNECT_RETRIES} "
                    f"failed: {e}. Retrying in {wait:.1f}s..."
                )
                self._close_sock_silently()
                time.sleep(wait)

        raise ConnectionError(
            f"Could not connect after {MAX_CONNECT_RETRIES} attempts. "
            f"Last error: {last_error}"
        )

    # ──────────────────────────────────────────────────────────────────
    # RUN ID
    # ──────────────────────────────────────────────────────────────────

    def get_run_id(self) -> str:
        """Fetch run_id from server. Call once after connect_with_retry()."""
        run_id = self._client.request_run_id(self._sock)
        if not run_id:
            raise RuntimeError("Server did not return a run_id")
        return run_id

    # ──────────────────────────────────────────────────────────────────
    # SEND IMAGE  (with per-image retry + reconnect)
    # ──────────────────────────────────────────────────────────────────

    def send_image_with_retry(self, payload: bytes, image_id: str):
        """
        Send one image to the server.

        On socket failure: reconnect, re-auth, then retry the same image.
        Returns (status, send_time, ack_time) on success.
        Raises RuntimeError after MAX_SEND_RETRIES failed attempts.
        """
        last_error = None

        for attempt in range(MAX_SEND_RETRIES):
            try:
                return self._client.send_image(self._sock, payload, image_id)

            except (OSError, BrokenPipeError, ConnectionResetError,
                    ConnectionError, socket.timeout) as e:
                last_error = e
                wait = _backoff(attempt)
                print(
                    f"[RETRY] Image {image_id} send failed (attempt "
                    f"{attempt + 1}/{MAX_SEND_RETRIES}): {e}. "
                    f"Reconnecting in {wait:.1f}s..."
                )
                self._close_sock_silently()
                time.sleep(wait)
                self._reconnect()

        raise RuntimeError(
            f"Failed to send {image_id} after {MAX_SEND_RETRIES} attempts. "
            f"Last error: {last_error}"
        )

    # ──────────────────────────────────────────────────────────────────
    # SEND CSV  (with retry + reconnect)
    # ──────────────────────────────────────────────────────────────────

    def send_csv_with_retry(self, csv_path: str, run_id: str) -> str:
        """
        Upload the metrics CSV.  Same retry / reconnect pattern as images.
        Returns the server status string.
        """
        last_error = None

        for attempt in range(MAX_SEND_RETRIES):
            try:
                return self._client.send_csv(self._sock, csv_path, run_id)

            except (OSError, BrokenPipeError, ConnectionResetError,
                    ConnectionError, socket.timeout) as e:
                last_error = e
                wait = _backoff(attempt)
                print(
                    f"[RETRY] CSV upload failed (attempt "
                    f"{attempt + 1}/{MAX_SEND_RETRIES}): {e}. "
                    f"Reconnecting in {wait:.1f}s..."
                )
                self._close_sock_silently()
                time.sleep(wait)
                self._reconnect()

        raise RuntimeError(
            f"Failed to upload CSV after {MAX_SEND_RETRIES} attempts. "
            f"Last error: {last_error}"
        )

    # ──────────────────────────────────────────────────────────────────
    # CLOSE
    # ──────────────────────────────────────────────────────────────────

    def close(self) -> None:
        if self._sock:
            self._client.send_goodbye(self._sock)
        self._close_sock_silently()
        print("[CLIENT] Connection closed")

    # ──────────────────────────────────────────────────────────────────
    # INTERNALS
    # ──────────────────────────────────────────────────────────────────

    def _reconnect(self) -> None:
        """Re-connect and re-auth. Raises if it can't."""
        print("[RETRY] Attempting reconnect...")
        self.connect_with_retry()

    def _close_sock_silently(self) -> None:
        if self._sock:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None
