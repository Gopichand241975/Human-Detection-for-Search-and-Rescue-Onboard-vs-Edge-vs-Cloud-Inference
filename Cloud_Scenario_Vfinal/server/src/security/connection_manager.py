import time
import threading
from collections import defaultdict


class ConnectionManager:
    def __init__(self, max_connections=1, rate_limit=5):
        self._lock = threading.Lock()
        self.active_connections = 0
        self.max_connections = max_connections
        self.rate_limit = rate_limit
        self.client_requests = defaultdict(list)

    def can_accept(self, addr):
        with self._lock:
            if self.active_connections >= self.max_connections:
                return False

            now = time.time()
            self.client_requests[addr] = [
                t for t in self.client_requests[addr] if now - t < 10
            ]

            if len(self.client_requests[addr]) >= self.rate_limit:
                return False

            self.client_requests[addr].append(now)
            return True

    def add_connection(self):
        with self._lock:
            self.active_connections += 1

    def remove_connection(self):
        with self._lock:
            self.active_connections = max(0, self.active_connections - 1)