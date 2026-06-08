"""
logger.py — Hybrid log architecture
────────────────────────────────────

GLOBAL logs  (server/logs/)         ← written for every event, always
──────────────────────────────────────────────────────────────────────
  server.log    SERVER_START / SERVER_STOP / fatal errors
  security.log  AUTH failures, TIMEOUT, rate-limit rejections
  errors.log    every ERROR-level event regardless of source

PER-RUN logs  (runs/<run_name>/logs/)   ← written for current run only
──────────────────────────────────────────────────────────────────────
  events.log    CONNECTION, AUTH_SUCCESS, IMAGE_*, TEXT_*, PERF

Routing table
─────────────
log_event()     → events.log (run)
log_security()  → security.log (global)
log_error()     → errors.log (global) + events.log (run)
log_server()    → server.log (global)          internal use

All files share the same line format:
  TIMESTAMP | LEVEL | EVENT_TYPE | MESSAGE | DATA
"""

import os
import threading
from datetime import datetime, timezone


class Logger:

    def __init__(self, global_log_dir: str):
        """
        Parameters
        ----------
        global_log_dir : path to server/logs/  (created by run_server.py)
        """
        self._lock = threading.Lock()
        self._request_counter = 0

        # ── Global log paths ───────────────────────────────────────────
        self._global_log_dir = global_log_dir
        os.makedirs(global_log_dir, exist_ok=True)

        self._server_log   = os.path.join(global_log_dir, "server.log")
        self._security_log = os.path.join(global_log_dir, "security.log")
        self._errors_log   = os.path.join(global_log_dir, "errors.log")

        # ── Per-run log path (resolved lazily after run context is set) ─
        self._run_events_log: str | None = None

    # ──────────────────────────────────────────────────────────────────
    # RUN CONTEXT BINDING
    # Called from run_server.py right after set_run_context()
    # ──────────────────────────────────────────────────────────────────
    def bind_run(self, run_context: dict) -> None:
        """Attach the per-run events log. Call once after create_new_run()."""
        run_logs_dir = run_context["logs"]
        os.makedirs(run_logs_dir, exist_ok=True)
        self._run_events_log = os.path.join(run_logs_dir, "events.log")

    # ──────────────────────────────────────────────────────────────────
    # REQUEST ID
    # ──────────────────────────────────────────────────────────────────
    def new_request_id(self) -> str:
        with self._lock:
            self._request_counter += 1
            return f"req_{self._request_counter:04d}"

    # ──────────────────────────────────────────────────────────────────
    # PUBLIC API
    # ──────────────────────────────────────────────────────────────────
    def log_event(self, event_type: str, message: str, data: dict = None) -> None:
        """
        Per-run event: connections, image processing, text messages, perf.
        Written to:  runs/<run>/logs/events.log
        """
        line = self._format("INFO", event_type, message, data)
        self._write(self._run_events_log, line)

    def log_security(self, event_type: str, message: str, data: dict = None) -> None:
        """
        Security events: bad tokens, TIMEOUT, rate-limit, unauthorized access.
        Written to:  server/logs/security.log  (global)
        """
        line = self._format("SECURITY", event_type, message, data)
        self._write(self._security_log, line)

    def log_error(self, event_type: str, message: str, data: dict = None) -> None:
        """
        Errors: always go to global errors.log AND the current run events.log.
        Written to:  server/logs/errors.log  +  runs/<run>/logs/events.log
        """
        line = self._format("ERROR", event_type, message, data)
        self._write(self._errors_log, line)
        self._write(self._run_events_log, line)

    def log_server(self, event_type: str, message: str, data: dict = None) -> None:
        """
        Server lifecycle events: startup, shutdown, run creation.
        Written to:  server/logs/server.log  (global)
        """
        line = self._format("SERVER", event_type, message, data)
        self._write(self._server_log, line)

    def record_processing_time(self, duration_sec: float) -> None:
        """Convenience wrapper — logs image processing wall-clock time."""
        self.log_event(
            "PROCESSING_TIME",
            f"{round(duration_sec * 1000, 2)} ms",
            None
        )

    # ──────────────────────────────────────────────────────────────────
    # INTERNAL
    # ──────────────────────────────────────────────────────────────────
    @staticmethod
    def _format(level: str, event_type: str, message: str, data) -> str:
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
        return f"{ts} | {level:<8} | {event_type} | {message} | {data}\n"

    def _write(self, filepath: str | None, line: str) -> None:
        """Thread-safe append to a log file. Silently skips if path is None."""
        if not filepath:
            return

        with self._lock:
            with open(filepath, "a", encoding="utf-8") as f:
                f.write(line)
