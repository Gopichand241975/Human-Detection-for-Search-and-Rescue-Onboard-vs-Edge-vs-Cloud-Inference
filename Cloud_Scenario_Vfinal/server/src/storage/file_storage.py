import os

class FileStorage:

    def __init__(self, base_dir):
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)

    def save(self, filename, payload):
        path = os.path.join(self.base_dir, filename)

        with open(path, "wb") as f:
            f.write(payload)

        return path