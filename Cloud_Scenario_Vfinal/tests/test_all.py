"""
tests/test_all.py
─────────────────
Full regression suite for the cloud_scenario refactor.

Run from the repo root:
    python -m pytest tests/test_all.py -v

Tests are grouped by component:
  T1  preprocess.py            — new (payload, ms) return contract
  T2  ClientMetricsRecorder    — preprocess_ms column written + backward compat
  T3  client ImageHandler      — propagates tuple, handles blurry skip
  T4  ServerMetricsRecorder    — queue_wait_ms column written
  T5  engine.py                — enqueue_time stamped, queue_wait_ms flows through
  T6  latency_eval.py          — old CSVs (no new cols), new CSVs, merged output cols
  T7  protocol helpers         — serialize / receive round-trip, create_header
  T8  SecurityPipeline         — token, size, checksum, filename validation
  T9  prediction_storage       — xyxy→xywh, JSON written with correct schema
  T10 sync_runs                — runs.json rebuilt correctly
"""

import csv
import json
import os
import struct
import sys
import tempfile
import threading
import time
import types

import numpy as np
import pytest

# ── repo root on path ──────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


# ══════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════

def _csv_rows(path):
    with open(path) as f:
        return list(csv.reader(f))


def _write_csv(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)


# ══════════════════════════════════════════════════════════════════════
# T1 — preprocess.py
# ══════════════════════════════════════════════════════════════════════

class TestPreprocess:

    IMG = os.path.join(ROOT, "client", "data", "001.jpg")

    def test_returns_tuple(self):
        from client.src.processing.preprocess import preprocess_image
        result = preprocess_image(self.IMG)
        assert isinstance(result, tuple) and len(result) == 2, \
            "preprocess_image must return a 2-tuple"

    def test_payload_is_bytes(self):
        from client.src.processing.preprocess import preprocess_image
        payload, _ = preprocess_image(self.IMG)
        assert isinstance(payload, bytes) and len(payload) > 0

    def test_preprocess_ms_positive_float(self):
        from client.src.processing.preprocess import preprocess_image
        _, ms = preprocess_image(self.IMG)
        assert isinstance(ms, float) and ms > 0

    def test_blurry_returns_none_zero(self):
        """Patch blur variance to be below the skip threshold."""
        import client.src.processing.preprocess as pp
        orig = pp.compute_blur_variance
        pp.compute_blur_variance = lambda img: 0.0   # always below threshold
        try:
            payload, ms = pp.preprocess_image(self.IMG)
            assert payload is None
            assert ms == 0.0
        finally:
            pp.compute_blur_variance = orig

    def test_explicit_dimensions_used(self):
        from client.src.processing.preprocess import preprocess_image
        payload, ms = preprocess_image(self.IMG, width=320, height=240, quality=50)
        assert payload is not None and ms > 0


# ══════════════════════════════════════════════════════════════════════
# T2 — ClientMetricsRecorder
# ══════════════════════════════════════════════════════════════════════

class TestClientMetricsRecorder:

    def _make_recorder(self, tdir, run_id="run_test_001"):
        from client.src.core.metrics_recorder_client import ClientMetricsRecorder
        path = os.path.join(tdir, f"{run_id}_network-delay.csv")
        return ClientMetricsRecorder(run_id, log_file=path), path

    def test_header_has_preprocess_ms(self):
        with tempfile.TemporaryDirectory() as tdir:
            rec, path = self._make_recorder(tdir)
            rows = _csv_rows(path)
            assert rows[0][-1] == "preprocess_ms"

    def test_full_header(self):
        with tempfile.TemporaryDirectory() as tdir:
            rec, path = self._make_recorder(tdir)
            assert _csv_rows(path)[0] == [
                "image_id", "send_time", "ack_receive_time",
                "RTT_ack", "network_delay", "preprocess_ms",
            ]

    def test_record_writes_preprocess_ms(self):
        with tempfile.TemporaryDirectory() as tdir:
            rec, path = self._make_recorder(tdir)
            rec.record("run_test_001_img_0001", 1000.0, 1000.1, preprocess_ms=12.345)
            row = _csv_rows(path)[1]
            assert row[5] == "12.345"

    def test_record_rtt_and_delay_correct(self):
        with tempfile.TemporaryDirectory() as tdir:
            rec, path = self._make_recorder(tdir)
            rec.record("run_test_001_img_0001", 1000.0, 1000.1, preprocess_ms=0.0)
            row = _csv_rows(path)[1]
            rtt = float(row[3])
            delay = float(row[4])
            assert abs(rtt - 0.1) < 1e-9
            assert abs(delay - 0.05) < 1e-9

    def test_record_default_preprocess_ms_zero(self):
        """Calling record() without preprocess_ms should write 0.0."""
        with tempfile.TemporaryDirectory() as tdir:
            rec, path = self._make_recorder(tdir)
            rec.record("run_test_001_img_0001", 1000.0, 1000.05)
            row = _csv_rows(path)[1]
            assert float(row[5]) == 0.0

    def test_generate_image_id_sequential(self):
        with tempfile.TemporaryDirectory() as tdir:
            rec, _ = self._make_recorder(tdir, "run_seq")
            ids = [rec.generate_image_id() for _ in range(3)]
            assert ids == [
                "run_seq_img_0001",
                "run_seq_img_0002",
                "run_seq_img_0003",
            ]

    def test_multiple_rows_appended(self):
        with tempfile.TemporaryDirectory() as tdir:
            rec, path = self._make_recorder(tdir)
            for i in range(5):
                rec.record(f"img_{i:04d}", float(i), float(i) + 0.05, preprocess_ms=float(i))
            rows = _csv_rows(path)
            assert len(rows) == 6   # header + 5 data rows


# ══════════════════════════════════════════════════════════════════════
# T3 — client ImageHandler
# ══════════════════════════════════════════════════════════════════════

class TestClientImageHandler:

    IMG = os.path.join(ROOT, "client", "data", "001.jpg")

    def test_prepare_returns_tuple(self):
        from client.src.processing.image_handler import ImageHandler
        h = ImageHandler(640, 480, 90)
        result = h.prepare(self.IMG)
        assert isinstance(result, tuple) and len(result) == 2

    def test_prepare_payload_bytes(self):
        from client.src.processing.image_handler import ImageHandler
        payload, ms = ImageHandler(640, 480, 90).prepare(self.IMG)
        assert isinstance(payload, bytes) and len(payload) > 0

    def test_prepare_ms_positive(self):
        from client.src.processing.image_handler import ImageHandler
        _, ms = ImageHandler(640, 480, 90).prepare(self.IMG)
        assert isinstance(ms, float) and ms > 0

    def test_prepare_blurry_returns_none(self):
        import client.src.processing.preprocess as pp
        orig = pp.compute_blur_variance
        pp.compute_blur_variance = lambda img: 0.0
        try:
            from client.src.processing.image_handler import ImageHandler
            payload, ms = ImageHandler(640, 480, 90).prepare(self.IMG)
            assert payload is None
            assert ms == 0.0
        finally:
            pp.compute_blur_variance = orig


# ══════════════════════════════════════════════════════════════════════
# T4 — ServerMetricsRecorder
# ══════════════════════════════════════════════════════════════════════

class TestServerMetricsRecorder:

    def _make_recorder(self, tdir, run_name="run_test_srv"):
        import server.src.core.run_context as rc
        rc._RUN_CONTEXT = {"run_name": run_name, "metrics": tdir}
        # fresh instance (lazy init re-runs on first call)
        from server.src.core.metrics_recorder_server import ServerMetricsRecorder
        return ServerMetricsRecorder()

    def test_header_has_queue_wait_ms(self):
        with tempfile.TemporaryDirectory() as tdir:
            srv = self._make_recorder(tdir)
            srv.mark_inference_start("id_001", queue_wait_ms=0.0)
            time.sleep(0.005)
            srv.mark_inference_end("id_001")
            rows = _csv_rows(srv.log_file)
            assert rows[0][-1] == "queue_wait_ms"

    def test_full_header(self):
        with tempfile.TemporaryDirectory() as tdir:
            srv = self._make_recorder(tdir)
            srv.mark_inference_start("id_002", queue_wait_ms=0.0)
            time.sleep(0.005)
            srv.mark_inference_end("id_002")
            assert _csv_rows(srv.log_file)[0] == [
                "image_id", "inference_start", "inference_end",
                "latency", "queue_wait_ms",
            ]

    def test_queue_wait_ms_written(self):
        with tempfile.TemporaryDirectory() as tdir:
            srv = self._make_recorder(tdir)
            srv.mark_inference_start("id_003", queue_wait_ms=42.123)
            time.sleep(0.005)
            srv.mark_inference_end("id_003")
            row = _csv_rows(srv.log_file)[1]
            assert float(row[4]) == 42.123

    def test_latency_positive(self):
        with tempfile.TemporaryDirectory() as tdir:
            srv = self._make_recorder(tdir)
            srv.mark_inference_start("id_004", queue_wait_ms=0.0)
            time.sleep(0.02)
            srv.mark_inference_end("id_004")
            row = _csv_rows(srv.log_file)[1]
            assert float(row[3]) > 0

    def test_default_queue_wait_zero(self):
        with tempfile.TemporaryDirectory() as tdir:
            srv = self._make_recorder(tdir)
            srv.mark_inference_start("id_005")   # no queue_wait_ms arg
            time.sleep(0.005)
            srv.mark_inference_end("id_005")
            row = _csv_rows(srv.log_file)[1]
            assert float(row[4]) == 0.0

    def test_end_without_start_is_noop(self):
        """mark_inference_end for an unknown id must not raise and must not write a data row."""
        with tempfile.TemporaryDirectory() as tdir:
            srv = self._make_recorder(tdir)
            srv.mark_inference_end("nonexistent")   # must not raise
            # _ensure_init runs but no data row should be written
            if srv.log_file and os.path.exists(srv.log_file):
                rows = _csv_rows(srv.log_file)
                assert len(rows) == 1, "only header should be present"

    def test_multiple_images_appended(self):
        with tempfile.TemporaryDirectory() as tdir:
            srv = self._make_recorder(tdir)
            for i in range(4):
                srv.mark_inference_start(f"img_{i}", queue_wait_ms=float(i))
                time.sleep(0.005)
                srv.mark_inference_end(f"img_{i}")
            rows = _csv_rows(srv.log_file)
            assert len(rows) == 5   # header + 4


# ══════════════════════════════════════════════════════════════════════
# T5 — engine.py
# ══════════════════════════════════════════════════════════════════════

class TestEngine:
    """
    engine.py imports inference.py which imports ultralytics.
    We stub both at the sys.modules level before importing engine.
    """

    @pytest.fixture(autouse=True)
    def _stub_deps(self, monkeypatch):
        # Stub ultralytics
        ult = types.ModuleType("ultralytics")
        ult.YOLO = object
        monkeypatch.setitem(sys.modules, "ultralytics", ult)

        # Stub the inference sub-module so engine's relative import works
        fake_inf = types.ModuleType("server.src.inference.inference")
        fake_inf.process_image = lambda image, filename, image_id: None
        monkeypatch.setitem(sys.modules, "server.src.inference.inference", fake_inf)

        # Also stub model_loader / predictor / postprocess to be safe
        for mod in ("server.src.inference.model_loader",
                    "server.src.inference.predictor",
                    "server.src.inference.postprocess"):
            monkeypatch.setitem(sys.modules, mod, types.ModuleType(mod))

    def _fresh_engine(self, monkeypatch):
        """Return a freshly reloaded engine module with a fake metrics sink."""
        import importlib
        import server.src.inference.engine as eng
        importlib.reload(eng)

        recorded = {}

        class FakeMetrics:
            def mark_inference_start(self, image_id, queue_wait_ms=0.0):
                recorded["image_id"] = image_id
                recorded["queue_wait_ms"] = queue_wait_ms
            def mark_inference_end(self, image_id):
                recorded["ended"] = True

        eng._metrics = FakeMetrics()
        eng._worker_started = False
        return eng, recorded

    def test_queue_wait_ms_non_negative(self, monkeypatch):
        eng, recorded = self._fresh_engine(monkeypatch)
        eng.start()
        time.sleep(0.05)
        dummy = np.zeros((4, 4, 3), dtype="uint8")
        eng.submit(dummy, "test.jpg", "run_t_img_0001")
        time.sleep(0.3)
        assert recorded.get("queue_wait_ms", -1) >= 0

    def test_image_id_flows_through(self, monkeypatch):
        eng, recorded = self._fresh_engine(monkeypatch)
        eng.start()
        time.sleep(0.05)
        dummy = np.zeros((4, 4, 3), dtype="uint8")
        eng.submit(dummy, "test.jpg", "run_t_img_0042")
        time.sleep(0.3)
        assert recorded.get("image_id") == "run_t_img_0042"

    def test_submit_returns_true_when_space(self, monkeypatch):
        eng, _ = self._fresh_engine(monkeypatch)
        dummy = np.zeros((4, 4, 3), dtype="uint8")
        result = eng.submit(dummy, "test.jpg", "run_t_img_0001")
        assert result is True

    def test_submit_returns_false_when_full(self, monkeypatch):
        eng, _ = self._fresh_engine(monkeypatch)
        # Fill the queue without starting the worker
        dummy = np.zeros((4, 4, 3), dtype="uint8")
        for _ in range(eng.MAX_QUEUE_SIZE):
            eng._queue.put((dummy, "f.jpg", "id", time.time()))
        result = eng.submit(dummy, "test.jpg", "overflow")
        assert result is False

    def test_start_idempotent(self, monkeypatch):
        eng, _ = self._fresh_engine(monkeypatch)
        eng.start()
        eng.start()   # second call must not spawn a second thread
        assert eng._worker_started is True


# ══════════════════════════════════════════════════════════════════════
# T6 — latency_eval.py
# ══════════════════════════════════════════════════════════════════════

class TestLatencyEval:
    """Tests run against synthetic CSV files in a temp directory tree."""

    # ── helpers ────────────────────────────────────────────────────────

    def _make_tree(self, tdir):
        """Return (client_metrics_dir, server_metrics_dir, output_dir)."""
        cm  = os.path.join(tdir, "data", "client_metrics")
        sm  = os.path.join(tdir, "data", "metrics")
        out = os.path.join(tdir, "evaluation", "results", "latency")
        for d in (cm, sm, out):
            os.makedirs(d, exist_ok=True)
        return cm, sm, out

    def _patch_eval(self, monkeypatch, tdir, out):
        """Patch module-level paths so eval reads from tdir."""
        import evaluation.latency_eval as le
        monkeypatch.setattr(le, "OUTPUT_DIR", out)
        # Patch the file paths that load_network / load_inference build
        orig_load_network  = le.load_network
        orig_load_inference = le.load_inference

        def patched_load_network(run_id):
            file = os.path.join(tdir, f"data/client_metrics/run_{run_id}_network-delay.csv")
            import pandas as pd
            if not os.path.exists(file):
                return None
            df = pd.read_csv(file)
            if not {"image_id", "network_delay"}.issubset(df.columns):
                return None
            keep = ["image_id", "network_delay"]
            if "preprocess_ms" in df.columns:
                keep.append("preprocess_ms")
            df = df[keep].copy()
            df["image_id"] = df["image_id"].apply(le.normalize_image_id)
            df["network_delay"] = df["network_delay"] * 1000
            df = df.rename(columns={"network_delay": "network_delay_ms"})
            return df

        def patched_load_inference(run_id):
            file = os.path.join(tdir, f"data/metrics/run_{run_id}_latency.csv")
            import pandas as pd
            if not os.path.exists(file):
                return None
            df = pd.read_csv(file)
            if not {"image_id", "latency"}.issubset(df.columns):
                return None
            keep = ["image_id", "latency"]
            if "queue_wait_ms" in df.columns:
                keep.append("queue_wait_ms")
            df = df[keep].copy()
            df["image_id"] = df["image_id"].apply(le.normalize_image_id)
            df["latency"] = df["latency"] * 1000
            df = df.rename(columns={"latency": "inference_latency_ms"})
            return df

        monkeypatch.setattr(le, "load_network",   patched_load_network)
        monkeypatch.setattr(le, "load_inference", patched_load_inference)
        return le

    # ── old CSVs (no new columns) ──────────────────────────────────────

    def test_old_csvs_produce_output(self, monkeypatch):
        with tempfile.TemporaryDirectory() as tdir:
            cm, sm, out = self._make_tree(tdir)
            _write_csv(
                os.path.join(cm, "run_20260101_000000_network-delay.csv"),
                [["image_id", "send_time", "ack_receive_time", "RTT_ack", "network_delay"],
                 ["run_20260101_000000_img_0001", "1000", "1000.01", "0.01", "0.005"],
                 ["run_20260101_000000_img_0002", "1001", "1001.02", "0.02", "0.010"]],
            )
            _write_csv(
                os.path.join(sm, "run_20260101_000000_latency.csv"),
                [["image_id", "inference_start", "inference_end", "latency"],
                 ["run_20260101_000000_img_0001", "1000.01", "1000.41", "0.4"],
                 ["run_20260101_000000_img_0002", "1001.02", "1001.14", "0.12"]],
            )
            le = self._patch_eval(monkeypatch, tdir, out)
            le.evaluate_run("20260101_000000")
            out_file = os.path.join(out, "run_20260101_000000_latency.csv")
            assert os.path.exists(out_file)

    def test_old_csvs_mandatory_cols_present(self, monkeypatch):
        with tempfile.TemporaryDirectory() as tdir:
            cm, sm, out = self._make_tree(tdir)
            _write_csv(
                os.path.join(cm, "run_20260101_000001_network-delay.csv"),
                [["image_id", "send_time", "ack_receive_time", "RTT_ack", "network_delay"],
                 ["run_20260101_000001_img_0001", "1000", "1000.01", "0.01", "0.005"]],
            )
            _write_csv(
                os.path.join(sm, "run_20260101_000001_latency.csv"),
                [["image_id", "inference_start", "inference_end", "latency"],
                 ["run_20260101_000001_img_0001", "1000.01", "1000.41", "0.4"]],
            )
            le = self._patch_eval(monkeypatch, tdir, out)
            le.evaluate_run("20260101_000001")
            rows = _csv_rows(os.path.join(out, "run_20260101_000001_latency.csv"))
            header = rows[0]
            for col in ("image_id", "network_delay_ms", "inference_latency_ms", "total_latency_ms"):
                assert col in header, f"missing column: {col}"

    def test_old_csvs_no_optional_cols(self, monkeypatch):
        """Old CSVs must NOT produce preprocess_ms / queue_wait_ms columns."""
        with tempfile.TemporaryDirectory() as tdir:
            cm, sm, out = self._make_tree(tdir)
            _write_csv(
                os.path.join(cm, "run_20260101_000002_network-delay.csv"),
                [["image_id", "send_time", "ack_receive_time", "RTT_ack", "network_delay"],
                 ["run_20260101_000002_img_0001", "1000", "1000.01", "0.01", "0.005"]],
            )
            _write_csv(
                os.path.join(sm, "run_20260101_000002_latency.csv"),
                [["image_id", "inference_start", "inference_end", "latency"],
                 ["run_20260101_000002_img_0001", "1000.01", "1000.41", "0.4"]],
            )
            le = self._patch_eval(monkeypatch, tdir, out)
            le.evaluate_run("20260101_000002")
            rows = _csv_rows(os.path.join(out, "run_20260101_000002_latency.csv"))
            header = rows[0]
            assert "preprocess_ms" not in header
            assert "queue_wait_ms" not in header

    # ── new CSVs (with new columns) ────────────────────────────────────

    def test_new_csvs_optional_cols_present(self, monkeypatch):
        with tempfile.TemporaryDirectory() as tdir:
            cm, sm, out = self._make_tree(tdir)
            _write_csv(
                os.path.join(cm, "run_20260101_000003_network-delay.csv"),
                [["image_id", "send_time", "ack_receive_time", "RTT_ack", "network_delay", "preprocess_ms"],
                 ["run_20260101_000003_img_0001", "1000", "1000.01", "0.01", "0.005", "15.0"]],
            )
            _write_csv(
                os.path.join(sm, "run_20260101_000003_latency.csv"),
                [["image_id", "inference_start", "inference_end", "latency", "queue_wait_ms"],
                 ["run_20260101_000003_img_0001", "1000.01", "1000.41", "0.4", "2.5"]],
            )
            le = self._patch_eval(monkeypatch, tdir, out)
            le.evaluate_run("20260101_000003")
            rows = _csv_rows(os.path.join(out, "run_20260101_000003_latency.csv"))
            header = rows[0]
            assert "preprocess_ms" in header
            assert "queue_wait_ms" in header

    def test_new_csvs_total_latency_includes_all(self, monkeypatch):
        """total_latency_ms = preprocess + network + queue + inference."""
        with tempfile.TemporaryDirectory() as tdir:
            cm, sm, out = self._make_tree(tdir)
            # network_delay = 0.005 s → 5 ms; preprocess_ms = 10
            _write_csv(
                os.path.join(cm, "run_20260101_000004_network-delay.csv"),
                [["image_id", "send_time", "ack_receive_time", "RTT_ack", "network_delay", "preprocess_ms"],
                 ["run_20260101_000004_img_0001", "1000", "1000.005", "0.005", "0.005", "10.0"]],
            )
            # latency = 0.1 s → 100 ms; queue_wait_ms = 3
            _write_csv(
                os.path.join(sm, "run_20260101_000004_latency.csv"),
                [["image_id", "inference_start", "inference_end", "latency", "queue_wait_ms"],
                 ["run_20260101_000004_img_0001", "1000.005", "1000.105", "0.1", "3.0"]],
            )
            le = self._patch_eval(monkeypatch, tdir, out)
            le.evaluate_run("20260101_000004")
            rows = _csv_rows(os.path.join(out, "run_20260101_000004_latency.csv"))
            header = rows[0]
            data   = rows[1]
            total = float(data[header.index("total_latency_ms")])
            # 10 (preprocess) + 5 (network) + 3 (queue) + 100 (inference) = 118
            assert abs(total - 118.0) < 0.01, f"expected 118, got {total}"

    def test_missing_server_csv_skips(self, monkeypatch):
        with tempfile.TemporaryDirectory() as tdir:
            cm, sm, out = self._make_tree(tdir)
            _write_csv(
                os.path.join(cm, "run_20260101_000005_network-delay.csv"),
                [["image_id", "network_delay"],
                 ["run_20260101_000005_img_0001", "0.005"]],
            )
            # no server CSV
            le = self._patch_eval(monkeypatch, tdir, out)
            le.evaluate_run("20260101_000005")   # must not raise
            out_file = os.path.join(out, "run_20260101_000005_latency.csv")
            assert not os.path.exists(out_file)

    def test_no_matching_image_ids_skips(self, monkeypatch):
        with tempfile.TemporaryDirectory() as tdir:
            cm, sm, out = self._make_tree(tdir)
            _write_csv(
                os.path.join(cm, "run_20260101_000006_network-delay.csv"),
                [["image_id", "network_delay"],
                 ["run_20260101_000006_img_0001", "0.005"]],
            )
            _write_csv(
                os.path.join(sm, "run_20260101_000006_latency.csv"),
                [["image_id", "latency"],
                 ["run_20260101_000006_img_0099", "0.1"]],   # different id
            )
            le = self._patch_eval(monkeypatch, tdir, out)
            le.evaluate_run("20260101_000006")
            out_file = os.path.join(out, "run_20260101_000006_latency.csv")
            assert not os.path.exists(out_file)


# ══════════════════════════════════════════════════════════════════════
# T7 — protocol helpers
# ══════════════════════════════════════════════════════════════════════

class TestProtocol:

    def test_create_header_basic(self):
        from shared.protocols.protocol import create_header
        h = create_header("IMAGE", filename="test.jpg", payload=b"abc", token="tok")
        assert h["type"] == "IMAGE"
        assert h["filename"] == "test.jpg"
        assert h["payload_size"] == 3
        assert len(h["checksum"]) == 64   # sha256 hex

    def test_create_header_image_id(self):
        from shared.protocols.protocol import create_header
        h = create_header("IMAGE", image_id="run_x_img_0001", token="t")
        assert h["image_id"] == "run_x_img_0001"

    def test_create_header_no_image_id(self):
        from shared.protocols.protocol import create_header
        h = create_header("AUTH", token="t")
        assert "image_id" not in h

    def test_serialize_deserialize_roundtrip(self):
        import io
        from shared.protocols.protocol import create_header, serialize_message, receive_message

        payload = b"hello world"
        header  = create_header("IMAGE", filename="x.jpg", payload=payload, token="t")
        wire    = serialize_message(header, payload)

        # Wrap in a fake socket
        class FakeSock:
            def __init__(self, data):
                self._buf = io.BytesIO(data)
            def recv(self, n):
                return self._buf.read(n)

        h2, p2 = receive_message(FakeSock(wire))
        assert h2["type"] == "IMAGE"
        assert p2 == payload

    def test_empty_payload_roundtrip(self):
        import io
        from shared.protocols.protocol import create_header, serialize_message, receive_message

        header = create_header("AUTH", token="tok")
        wire   = serialize_message(header)

        class FakeSock:
            def __init__(self, data):
                self._buf = io.BytesIO(data)
            def recv(self, n):
                return self._buf.read(n)

        h2, p2 = receive_message(FakeSock(wire))
        assert h2["type"] == "AUTH"
        assert p2 == b""

    def test_checksum_of_empty_payload_is_empty(self):
        from shared.protocols.protocol import create_header
        h = create_header("AUTH", token="t")
        assert h["checksum"] == ""


# ══════════════════════════════════════════════════════════════════════
# T8 — SecurityPipeline
# ══════════════════════════════════════════════════════════════════════

class TestSecurityPipeline:

    def _pipeline(self):
        from server.src.security.security_pipeline import SecurityPipeline
        return SecurityPipeline(token="secret", max_size=1_000_000)

    def _img_header(self, payload, token="secret"):
        import hashlib
        return {
            "type": "IMAGE",
            "token": token,
            "payload_size": len(payload),
            "checksum": hashlib.sha256(payload).hexdigest(),
            "filename": "valid.jpg",
        }

    def test_valid_image_passes(self):
        p = self._pipeline()
        payload = b"x" * 100
        p.validate(self._img_header(payload), payload)   # must not raise

    def test_wrong_token_raises(self):
        from server.src.security.exceptions import SecurityError
        p = self._pipeline()
        payload = b"x"
        header  = self._img_header(payload, token="wrong")
        with pytest.raises(SecurityError) as exc:
            p.validate(header, payload)
        assert exc.value.code == "INVALID_TOKEN"

    def test_size_exceeded_raises(self):
        from server.src.security.exceptions import SecurityError
        p = self._pipeline()
        header = {
            "type": "IMAGE", "token": "secret",
            "payload_size": 2_000_000,
            "checksum": "", "filename": "f.jpg",
        }
        with pytest.raises(SecurityError) as exc:
            p.validate(header, b"x")
        assert exc.value.code == "SIZE_EXCEEDED"

    def test_bad_checksum_raises(self):
        from server.src.security.exceptions import SecurityError
        p = self._pipeline()
        payload = b"real data"
        header  = self._img_header(payload)
        header["checksum"] = "a" * 64   # wrong
        with pytest.raises(SecurityError) as exc:
            p.validate(header, payload)
        assert exc.value.code == "CORRUPTED"

    def test_path_traversal_filename_raises(self):
        from server.src.security.exceptions import SecurityError
        p = self._pipeline()
        payload = b"x"
        header  = self._img_header(payload)
        header["filename"] = "../../etc/passwd"
        with pytest.raises(SecurityError) as exc:
            p.validate(header, payload)
        assert exc.value.code == "INVALID_FILENAME"

    def test_text_size_limit(self):
        from server.src.security.exceptions import SecurityError
        p = self._pipeline()
        header = {
            "type": "TEXT", "token": "secret",
            "payload_size": 2000,
            "checksum": "", "filename": "",
        }
        with pytest.raises(SecurityError) as exc:
            p.validate(header, b"x" * 2000)
        assert exc.value.code == "SIZE_EXCEEDED"


# ══════════════════════════════════════════════════════════════════════
# T9 — prediction_storage
# ══════════════════════════════════════════════════════════════════════

class TestPredictionStorage:

    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as tdir:
            from server.src.storage import prediction_storage as ps
            ps._PREDICTIONS_DIR = None   # reset module state
            ps.init_predictions_dir(tdir)

            detections = [
                {"class_id": 0, "confidence": 0.95, "bbox": [10.0, 20.0, 110.0, 120.0]},
            ]
            ps.save_prediction("run_x_img_0001", detections, 34.56)

            out = os.path.join(tdir, "run_x_img_0001.json")
            assert os.path.exists(out)

            with open(out) as f:
                data = json.load(f)

            assert data["image_id"] == "run_x_img_0001"
            assert abs(data["inference_time_ms"] - 34.56) < 0.001
            assert len(data["detections"]) == 1

            bbox = data["detections"][0]["bbox"]
            # xyxy (10,20,110,120) → xywh x=10, y=20, w=100, h=100
            assert abs(bbox["x"]      - 10.0) < 0.01
            assert abs(bbox["y"]      - 20.0) < 0.01
            assert abs(bbox["width"]  - 100.0) < 0.01
            assert abs(bbox["height"] - 100.0) < 0.01

    def test_empty_detections(self):
        with tempfile.TemporaryDirectory() as tdir:
            from server.src.storage import prediction_storage as ps
            ps._PREDICTIONS_DIR = None
            ps.init_predictions_dir(tdir)
            ps.save_prediction("run_x_img_0002", [], 5.0)
            with open(os.path.join(tdir, "run_x_img_0002.json")) as f:
                data = json.load(f)
            assert data["detections"] == []

    def test_not_initialized_raises(self):
        from server.src.storage import prediction_storage as ps
        ps._PREDICTIONS_DIR = None
        with pytest.raises(RuntimeError):
            ps.save_prediction("id", [], 0.0)


# ══════════════════════════════════════════════════════════════════════
# T10 — sync_runs
# ══════════════════════════════════════════════════════════════════════

class TestSyncRuns:

    def _setup(self, tdir):
        pred_dir = os.path.join(tdir, "data", "predictions")
        lat_dir  = os.path.join(tdir, "data", "metrics")
        run_file = os.path.join(tdir, "data", "runs.json")
        os.makedirs(pred_dir, exist_ok=True)
        os.makedirs(lat_dir,  exist_ok=True)
        return pred_dir, lat_dir, run_file

    def _rebuild(self, monkeypatch, tdir, pred_dir, lat_dir, run_file):
        import evaluation.sync_runs as sr
        monkeypatch.setattr(sr, "PRED_DIR",    pred_dir)
        monkeypatch.setattr(sr, "LATENCY_DIR", lat_dir)
        monkeypatch.setattr(sr, "RUN_FILE",    run_file)
        sr.rebuild_runs_json()
        with open(run_file) as f:
            return json.load(f)

    def test_empty_pred_dir_produces_empty_runs(self, monkeypatch):
        with tempfile.TemporaryDirectory() as tdir:
            pred_dir, lat_dir, run_file = self._setup(tdir)
            data = self._rebuild(monkeypatch, tdir, pred_dir, lat_dir, run_file)
            assert data["runs"] == []

    def test_run_detected_with_predictions(self, monkeypatch):
        with tempfile.TemporaryDirectory() as tdir:
            pred_dir, lat_dir, run_file = self._setup(tdir)
            run_path = os.path.join(pred_dir, "run_20260101_120000")
            os.makedirs(run_path)
            open(os.path.join(run_path, "run_20260101_120000_img_0001.json"), "w").close()

            data = self._rebuild(monkeypatch, tdir, pred_dir, lat_dir, run_file)
            run_ids = [r["run_id"] for r in data["runs"]]
            assert "run_20260101_120000" in run_ids

    def test_has_latency_flag(self, monkeypatch):
        with tempfile.TemporaryDirectory() as tdir:
            pred_dir, lat_dir, run_file = self._setup(tdir)
            run_path = os.path.join(pred_dir, "run_20260101_130000")
            os.makedirs(run_path)
            open(os.path.join(run_path, "run_20260101_130000_img_0001.json"), "w").close()
            # write a latency CSV for this run
            open(os.path.join(lat_dir, "run_20260101_130000_latency.csv"), "w").close()

            data = self._rebuild(monkeypatch, tdir, pred_dir, lat_dir, run_file)
            run = next(r for r in data["runs"] if r["run_id"] == "run_20260101_130000")
            assert run["has_latency"] is True
            assert run["has_predictions"] is True

    def test_created_timestamp_parsed(self, monkeypatch):
        with tempfile.TemporaryDirectory() as tdir:
            pred_dir, lat_dir, run_file = self._setup(tdir)
            run_path = os.path.join(pred_dir, "run_20260215_093045")
            os.makedirs(run_path)
            open(os.path.join(run_path, "x.json"), "w").close()

            data = self._rebuild(monkeypatch, tdir, pred_dir, lat_dir, run_file)
            run = next(r for r in data["runs"] if r["run_id"] == "run_20260215_093045")
            assert run["created"] == "2026-02-15 09:30:45"


# ══════════════════════════════════════════════════════════════════════
# T10  ConnectionHandler — retry + reconnect logic
# ══════════════════════════════════════════════════════════════════════

class TestConnectionHandler:
    """
    Tests for client/src/network/connection_handler.py.
    All network I/O is replaced with simple MagicMock objects.
    time.sleep is patched to zero so tests run instantly.
    """

    def _make_handler(self, monkeypatch):
        """Return a ConnectionHandler with a mocked _client inside."""
        import types
        from unittest.mock import MagicMock, patch

        # Stub heavy client deps so the module can be imported cleanly
        for mod in ("client.config.config",):
            if mod not in sys.modules:
                fake = types.ModuleType(mod)
                fake.SERVER_HOST = "127.0.0.1"
                fake.SERVER_PORT = 5000
                fake.TOKEN       = "test"
                monkeypatch.setitem(sys.modules, mod, fake)

        # Stub client_socket so ClientSocket() never opens a real socket
        fake_cs_mod = types.ModuleType("client.src.network.client_socket")
        fake_cs_mod.ClientSocket = MagicMock
        monkeypatch.setitem(sys.modules, "client.src.network.client_socket", fake_cs_mod)

        # Force re-import of connection_handler to pick up stubs
        monkeypatch.delitem(sys.modules, "client.src.network.connection_handler", raising=False)

        from client.src.network.connection_handler import ConnectionHandler
        handler = ConnectionHandler()
        handler._client = MagicMock()
        return handler

    # ── connect_with_retry ────────────────────────────────────────────

    def test_connect_succeeds_first_attempt(self, monkeypatch):
        from unittest.mock import MagicMock, patch
        handler = self._make_handler(monkeypatch)
        handler._client.connect.return_value = MagicMock()
        handler._client.authenticate.return_value = "SUCCESS"

        with patch("client.src.network.connection_handler.time.sleep"):
            handler.connect_with_retry()

        assert handler._client.connect.call_count == 1
        assert handler._client.authenticate.call_count == 1

    def test_connect_retries_on_os_error(self, monkeypatch):
        from unittest.mock import MagicMock, patch
        handler = self._make_handler(monkeypatch)
        handler._client.connect.side_effect = [OSError("refused"), MagicMock()]
        handler._client.authenticate.return_value = "SUCCESS"

        with patch("client.src.network.connection_handler.time.sleep"):
            handler.connect_with_retry()

        assert handler._client.connect.call_count == 2

    def test_connect_raises_after_max_retries(self, monkeypatch):
        from unittest.mock import patch
        import pytest
        from client.src.network.connection_handler import MAX_CONNECT_RETRIES
        handler = self._make_handler(monkeypatch)
        handler._client.connect.side_effect = OSError("refused")

        with patch("client.src.network.connection_handler.time.sleep"):
            with pytest.raises(ConnectionError):
                handler.connect_with_retry()

        assert handler._client.connect.call_count == MAX_CONNECT_RETRIES

    def test_connect_raises_on_bad_auth(self, monkeypatch):
        from unittest.mock import MagicMock, patch
        import pytest
        handler = self._make_handler(monkeypatch)
        handler._client.connect.return_value = MagicMock()
        handler._client.authenticate.return_value = "INVALID_TOKEN"

        with patch("client.src.network.connection_handler.time.sleep"):
            with pytest.raises(ConnectionError):
                handler.connect_with_retry()

    # ── get_run_id ────────────────────────────────────────────────────

    def test_get_run_id_returns_server_value(self, monkeypatch):
        from unittest.mock import MagicMock
        handler = self._make_handler(monkeypatch)
        handler._sock = MagicMock()
        handler._client.request_run_id.return_value = "run_20260502_test"

        assert handler.get_run_id() == "run_20260502_test"

    def test_get_run_id_raises_when_none(self, monkeypatch):
        from unittest.mock import MagicMock
        import pytest
        handler = self._make_handler(monkeypatch)
        handler._sock = MagicMock()
        handler._client.request_run_id.return_value = None

        with pytest.raises(RuntimeError):
            handler.get_run_id()

    # ── send_image_with_retry ─────────────────────────────────────────

    def test_send_image_succeeds_first_attempt(self, monkeypatch):
        from unittest.mock import MagicMock, patch
        handler = self._make_handler(monkeypatch)
        handler._sock = MagicMock()
        handler._client.send_image.return_value = ("SUCCESS", 1.0, 1.1)
        handler._client.connect.return_value = MagicMock()
        handler._client.authenticate.return_value = "SUCCESS"

        with patch("client.src.network.connection_handler.time.sleep"):
            status, _, _ = handler.send_image_with_retry(b"bytes", "img_001")

        assert status == "SUCCESS"
        assert handler._client.send_image.call_count == 1

    def test_send_image_reconnects_on_broken_pipe(self, monkeypatch):
        from unittest.mock import MagicMock, patch
        handler = self._make_handler(monkeypatch)
        handler._sock = MagicMock()
        handler._client.send_image.side_effect = [
            BrokenPipeError("pipe"),
            ("SUCCESS", 1.0, 1.1),
        ]
        handler._client.connect.return_value = MagicMock()
        handler._client.authenticate.return_value = "SUCCESS"

        with patch("client.src.network.connection_handler.time.sleep"):
            status, _, _ = handler.send_image_with_retry(b"bytes", "img_001")

        assert status == "SUCCESS"
        assert handler._client.send_image.call_count == 2

    def test_send_image_raises_after_max_retries(self, monkeypatch):
        from unittest.mock import MagicMock, patch
        import pytest
        from client.src.network.connection_handler import MAX_SEND_RETRIES
        handler = self._make_handler(monkeypatch)
        handler._sock = MagicMock()
        handler._client.send_image.side_effect = OSError("down")
        handler._client.connect.return_value = MagicMock()
        handler._client.authenticate.return_value = "SUCCESS"

        with patch("client.src.network.connection_handler.time.sleep"):
            with pytest.raises(RuntimeError):
                handler.send_image_with_retry(b"bytes", "img_001")

        assert handler._client.send_image.call_count == MAX_SEND_RETRIES

    def test_send_image_same_image_id_on_retry(self, monkeypatch):
        """Retried calls must use the same image_id — no new IDs generated."""
        from unittest.mock import MagicMock, patch
        handler = self._make_handler(monkeypatch)
        handler._sock = MagicMock()
        handler._client.send_image.side_effect = [
            OSError("drop"),
            ("SUCCESS", 1.0, 1.1),
        ]
        handler._client.connect.return_value = MagicMock()
        handler._client.authenticate.return_value = "SUCCESS"

        with patch("client.src.network.connection_handler.time.sleep"):
            handler.send_image_with_retry(b"bytes", "img_042")

        for call in handler._client.send_image.call_args_list:
            assert call[0][2] == "img_042"

    # ── send_csv_with_retry ───────────────────────────────────────────

    def test_send_csv_succeeds(self, monkeypatch):
        from unittest.mock import MagicMock, patch
        handler = self._make_handler(monkeypatch)
        handler._sock = MagicMock()
        handler._client.send_csv.return_value = "SUCCESS"

        with patch("client.src.network.connection_handler.time.sleep"):
            status = handler.send_csv_with_retry("metrics.csv", "run_abc")

        assert status == "SUCCESS"

    def test_send_csv_reconnects_on_reset(self, monkeypatch):
        from unittest.mock import MagicMock, patch
        handler = self._make_handler(monkeypatch)
        handler._sock = MagicMock()
        handler._client.send_csv.side_effect = [
            ConnectionResetError("reset"),
            "SUCCESS",
        ]
        handler._client.connect.return_value = MagicMock()
        handler._client.authenticate.return_value = "SUCCESS"

        with patch("client.src.network.connection_handler.time.sleep"):
            status = handler.send_csv_with_retry("metrics.csv", "run_abc")

        assert status == "SUCCESS"
        assert handler._client.send_csv.call_count == 2

    def test_send_csv_raises_after_max_retries(self, monkeypatch):
        from unittest.mock import MagicMock, patch
        import pytest
        from client.src.network.connection_handler import MAX_SEND_RETRIES
        handler = self._make_handler(monkeypatch)
        handler._sock = MagicMock()
        handler._client.send_csv.side_effect = OSError("down")
        handler._client.connect.return_value = MagicMock()
        handler._client.authenticate.return_value = "SUCCESS"

        with patch("client.src.network.connection_handler.time.sleep"):
            with pytest.raises(RuntimeError):
                handler.send_csv_with_retry("metrics.csv", "run_abc")

        assert handler._client.send_csv.call_count == MAX_SEND_RETRIES

    # ── backoff ───────────────────────────────────────────────────────

    def test_backoff_increases_with_attempts(self, monkeypatch):
        self._make_handler(monkeypatch)
        from client.src.network.connection_handler import _backoff
        assert _backoff(1) > _backoff(0)
        assert _backoff(2) > _backoff(1)

    def test_backoff_capped_at_max(self, monkeypatch):
        self._make_handler(monkeypatch)
        from client.src.network.connection_handler import _backoff, MAX_BACKOFF_S
        assert _backoff(100) <= MAX_BACKOFF_S

    def test_backoff_attempt_0_equals_base(self, monkeypatch):
        self._make_handler(monkeypatch)
        from client.src.network.connection_handler import _backoff, BASE_BACKOFF_S
        assert abs(_backoff(0) - BASE_BACKOFF_S) < 1e-9
