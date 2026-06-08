import csv
import os

_LOG_DIR = os.path.join(
    os.path.dirname(__file__),
    "..", "..", "logs"
)


class ClientMetricsRecorder:
    def __init__(self, run_id, log_file=None):
        self.run_id = run_id
        self.sequence_number = 0

        if log_file:
            self.log_file = os.path.normpath(log_file)
        else:
            filename = f"{run_id}_network-delay.csv"
            self.log_file = os.path.normpath(
                os.path.join(_LOG_DIR, filename)
            )

        self.records = {}
        self._init_file()

    def _init_file(self):
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)

        if not os.path.exists(self.log_file):
            with open(self.log_file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "image_id",
                    "send_time",
                    "ack_receive_time",
                    "RTT_ack",
                    "network_delay",
                    "preprocess_ms",
                ])

    def generate_image_id(self):
        self.sequence_number += 1
        return f"{self.run_id}_img_{self.sequence_number:04d}"

    def record(self, image_id, send_time, ack_time, preprocess_ms=0.0):
        """
        Call once per image after send_image() returns.
        send_time / ack_time captured at socket boundary for accuracy.
        preprocess_ms measured before the send.
        """
        rtt           = ack_time - send_time
        network_delay = rtt / 2

        self._write(image_id, send_time, ack_time, rtt, network_delay, preprocess_ms)

    def _write(self, image_id, send, ack, rtt, delay, preprocess_ms):
        with open(self.log_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                image_id,
                send,
                ack,
                rtt,
                delay,
                round(preprocess_ms, 3),
            ])
