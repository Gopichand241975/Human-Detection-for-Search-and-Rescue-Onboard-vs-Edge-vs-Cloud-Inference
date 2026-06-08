import socket


class SafeSocket:
    def __init__(self, sock, timeout=10):
        self.sock = sock
        self.sock.settimeout(timeout)

    def recv(self, size):
        return self.sock.recv(size)

    def send(self, data):
        return self.sock.send(data)

    def recv_exact(self, size):
        data = b""

        while len(data) < size:
            chunk = self.sock.recv(size - len(data))

            if not chunk:
                raise ConnectionError("Connection closed")

            data += chunk

        return data

    def close(self):
        try:
            self.sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass

        try:
            self.sock.close()
        except OSError:
            pass