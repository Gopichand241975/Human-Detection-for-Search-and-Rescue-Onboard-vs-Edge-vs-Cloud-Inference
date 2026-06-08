import socket


class ServerSocket:
    def __init__(self, host="0.0.0.0", port=5000):
        self.host = host
        self.port = port

        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(10)  # increased from 1 to allow proper OS-level backlog

    def accept_client(self):
        conn, addr = self.server_socket.accept()
        print("[SERVER] Client connected:", addr)
        return conn

    def close(self):
        self.server_socket.close()
