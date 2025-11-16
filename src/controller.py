import os
import csv
import cv2
import math
import time
import numpy as np

from angle_analyzer import AngleAnalyzer
from YoloLineDetector import YOLOLineDetector
from line_detector import LineDetector

class VisionController:
    """
    Новая версия:
    - чистый угол от fitLine
    - confidence = deviation/90
    - резкие углы: deviation ∈ [30, 70]
    - PID учитывает угол
    """

    def __init__(self, camera, motors,
                 base_speed=50, turn_speed=25,
                 slowdown_factor=0.5, maneuver_timeout=0.25,
                 min_line_pixels=700,
                 telemetry_path="./telemetry/telemetry_log.csv",
                 use_yolo=True,
                 yolo_model_path="./checkpoints/yolov8n_seg_last/tflite_export/best_float32.tflite",
                 yolo_img_size=320,
                 yolo_conf_thresh=0.7,
                 yolo_iou_thresh=0.5):

        self.camera = camera
        self.motors = motors

        # скорости
        self.base_speed = base_speed
        self.turn_speed = turn_speed
        self.min_line_pixels = min_line_pixels

        # PID
        self.Kp = 0.55
        self.Ki = 0.015
        self.Kd = 0.1
        self.K_angle = 0.35        # <<< важный новый параметр
        self.prev_error = 0
        self.integral = 0
        self.prev_time = time.time()

        # детектор
        if use_yolo:
            self.detector = YOLOLineDetector(
                tflite_path=yolo_model_path,
                img_size=yolo_img_size,
                conf_thresh=yolo_conf_thresh,
                iou_thresh=yolo_iou_thresh,
                min_contour_area=60
            )
        else:
            self.detector = LineDetector()

        # углы
        self.angle = AngleAnalyzer(min_points=30, cooldown=0.25)

        # телеметрия
        self.telemetry_path = telemetry_path
        os.makedirs(os.path.dirname(telemetry_path) or ".", exist_ok=True)
        with open(self.telemetry_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["time", "mode", "x_bottom", "left", "right"])

    # LOG -----------------------------------------------------
    def _log(self, mode, x_bottom):
        with open(self.telemetry_path, "a", newline="") as f:
            csv.writer(f).writerow([
                time.strftime("%H:%M:%S"),
                mode,
                "" if x_bottom is None else x_bottom,
                int(getattr(self, "left_speed", 0)),
                int(getattr(self, "right_speed", 0))
            ])

    # PID -----------------------------------------------------
    def _pid_follow(self, x_bottom, angle_deg, width):
        center = width / 2
        error_x = (x_bottom - center) / center

        # угол нормализуем в [-1..1]
        error_angle = angle_deg / 90.0

        # комбинированная ошибка
        error = error_x + self.K_angle * error_angle

        now = time.time()
        dt = now - self.prev_time
        self.prev_time = now

        self.integral += error * dt
        derivative = (error - self.prev_error) / dt if dt > 0 else 0
        self.prev_error = error

        u = self.Kp * error + self.Ki * self.integral + self.Kd * derivative

        L = self.base_speed + self.base_speed * u
        R = self.base_speed - self.base_speed * u

        max_speed = self.base_speed * 1.8
        L = max(min(L, max_speed), -self.base_speed * 0.3)
        R = max(min(R, max_speed), -self.base_speed * 0.3)

        self.left_speed, self.right_speed = L, R
        self.motors.set_speed(int(L), int(R))

    # TURN -----------------------------------------------------
    def _turn_left(self):
        self.left_speed = -self.turn_speed
        self.right_speed = self.turn_speed
        self.motors.set_speed(self.left_speed, self.right_speed)

    def _turn_right(self):
        self.left_speed = self.turn_speed
        self.right_speed = -self.turn_speed
        self.motors.set_speed(self.left_speed, self.right_speed)

    # STEP -----------------------------------------------------
    def step(self, debug=False):
        frame = self.camera.read()
        if frame is None:
            return None

        mask = self.detector.threshold(frame)

        if cv2.countNonZero(mask) < self.min_line_pixels:
            self.motors.stop()
            self._log("NO_LINE", None)
            return None

        skeleton = cv2.ximgproc.thinning(mask)
        h, w = skeleton.shape

        # анализ угла
        corner_type, direction, conf, angle_deg = self.angle.analyze(skeleton)

        # резкий поворот
        if corner_type == "right_turn":
            self._turn_right()
            self._log("ANGLE_RIGHT", None)
            return

        if corner_type == "left_turn":
            self._turn_left()
            self._log("ANGLE_LEFT", None)
            return

        # PID
        ys, xs = np.where(skeleton > 0)
        pts = np.column_stack((xs, ys))
        [vx, vy, x0, y0] = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01)

        x_bottom = x0 + vx * ((h - y0) / vy) if abs(vy) > 1e-5 else x0

        self._pid_follow(x_bottom, angle_deg, w)
        self._log("PID", x_bottom)

    def close(self):
        self.camera.release()
        self.motors.stop()
