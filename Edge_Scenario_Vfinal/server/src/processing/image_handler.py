import cv2
import numpy as np

def bytes_to_image(payload):
    nparr = np.frombuffer(payload, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img


def show_image(img):
    cv2.imshow("Received Image", img)
    cv2.waitKey(1)