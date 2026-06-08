"""
system_logger.py
────────────────
Logs CPU temperature at a fixed interval to:
    data/metrics/<run_id>_system.csv

On non-Raspberry Pi hardware, temperature is logged as "NA".
"""

import csv
import os
import time
import subprocess
import platform
from datetime import datetime


class SystemLogger:
    def __init__(self, output_path: str, interval: float = 1.0):
        """
        Parameters
        ----------
        output_path : full file path, e.g. data/metrics/run_X_system.csv
        interval    : minimum seconds between log entries
        """
        self.interval      = interval
        self.last_log_time = time.time()
        self._is_pi        = platform.machine().startswith("arm")

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        self._file   = open(output_path, mode="a", newline="")
        self._writer = csv.writer(self._file)

        if os.path.getsize(output_path) == 0:
            self._writer.writerow(["timestamp", "cpu_temp_c"])

    def _get_cpu_temp(self):
        if self._is_pi:
            try:
                output = subprocess.check_output(
                    ["vcgencmd", "measure_temp"],
                    stderr=subprocess.DEVNULL,
                ).decode()
                return float(output.replace("temp=", "").replace("'C\n", ""))
            except Exception:
                return "NA"
        return "NA"

    def log_if_due(self):
        now = time.time()
        if now - self.last_log_time >= self.interval:
            self._writer.writerow([datetime.now().isoformat(), self._get_cpu_temp()])
            self._file.flush()
            self.last_log_time = now

    def close(self):
        self._file.close()
