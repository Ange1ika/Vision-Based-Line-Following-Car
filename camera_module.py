import cv2
import numpy as np

try:
    from picamera2 import Picamera2
    PICAMERA2_AVAILABLE = True
except Exception:
    PICAMERA2_AVAILABLE = False


class MyPiCamera:
    """
    Обёртка над PiCamera2 с интерфейсом .read() -> np.ndarray (BGR).
    На ПК без камеры возвращает None (или можно заменить на VideoCapture).
    """
    def __init__(self, width=320, height=240, flip_mode=-1, fallback_webcam=False):
        self.width = width
        self.height = height
        self.flip_mode = flip_mode
        self.fallback_webcam = fallback_webcam

        self.cap = None
        self.picam2 = None

        if PICAMERA2_AVAILABLE:
            self.picam2 = Picamera2()
            config = self.picam2.create_preview_configuration(
                main={"size": (self.width, self.height), "format": "BGR888"}
            )
            self.picam2.configure(config)
            self.picam2.start()
        elif fallback_webcam:
            self.cap = cv2.VideoCapture(0)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

    def read(self):
        if self.picam2 is not None:
            frame = self.picam2.capture_array()
            if self.flip_mode in (0, 1, -1):
                frame = cv2.flip(frame, self.flip_mode)
            return frame

        if self.cap is not None:
            ok, frame = self.cap.read()
            if not ok:
                return None
            return frame

        return None  # ни камеры, ни вебкамеры

    def release(self):
        if self.picam2 is not None:
            self.picam2.stop()
        if self.cap is not None:
            self.cap.release()
