import time


class LatencyTracker:
    def start(self):
        return time.perf_counter()

    def stop(self, start_time):
        return (time.perf_counter() - start_time) * 1000  # ms
