def is_valid_image_filename(filename):
    allowed_extensions = [".jpg", ".jpeg", ".png"]
    filename = filename.lower()

    return any(filename.endswith(ext) for ext in allowed_extensions)