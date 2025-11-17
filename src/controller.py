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
                 maneuver_timeout=0.2,
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
        self.Kp = 0.6
        self.Ki = 0.0
        self.Kd = 0.1
        self.K_angle = 0.0      
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

        L = self.base_speed - self.base_speed * u
        R = self.base_speed + self.base_speed * u

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

        # линия потеряна
        if cv2.countNonZero(mask) < self.min_line_pixels:
            self.motors.stop()
            mode = "NO_LINE"
            self._log(mode, None)
            if debug:
                return self._visualize(frame, mask, np.zeros_like(mask), None, mode, ("straight",0,0,0))
            return None

        skeleton = cv2.ximgproc.thinning(mask)
        h, w = skeleton.shape

        # === УГЛЫ ===
        corner_type, direction, conf, angle_deg = self.angle.analyze(skeleton)
        angle_info = (corner_type, direction, conf, angle_deg)

        # Резкие углы
        if corner_type == "right_turn":
            self._turn_right()
            mode = "TURN_RIGHT"
            self._log(mode, None)
            if debug:
                return self._visualize(frame, mask, skeleton, None, mode, angle_info)
            return None

        if corner_type == "left_turn":
            self._turn_left()
            mode = "TURN_LEFT"
            self._log(mode, None)
            if debug:
                return self._visualize(frame, mask, skeleton, None, mode, angle_info)
            return None

        # === PID ===
        ys, xs = np.where(skeleton > 0)
        pts = np.column_stack((xs, ys))
        [vx, vy, x0, y0] = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01)
        # --- защита от почти вертикальной линии ---
        if abs(vy) < 1e-6:
            x_bottom = float(x0)
        else:
            t = (h - y0) / vy
            # защита от выходов в ±inf
            t = max(min(t, 1e6), -1e6)
            x_bottom = float(x0 + vx * t)


        self._pid_follow(x_bottom, angle_deg, w)
        mode = "PID"

        self._log(mode, x_bottom)

        if debug:
            return self._visualize(frame, mask, skeleton, x_bottom, mode, angle_info)

        return None

    def close(self):
        self.camera.release()
        self.motors.stop()

    
    def _visualize(self, frame, mask, skeleton, x_bottom, mode, angle_info):
        """
        Красивый визуализатор:
        - цветная маска
        - красный скелет
        - жёлтая линия fitLine
        - точка PID
        - многострочная инфопанель
        """
        vis = frame.copy()
        h, w = frame.shape[:2]

        # === Маска поверх изображения ===
        color_mask = cv2.applyColorMap(mask, cv2.COLORMAP_TURBO)
        vis = cv2.addWeighted(vis, 0.65, color_mask, 0.35, 0)

        # === Скелет красным ===
        skel_show = cv2.cvtColor(skeleton, cv2.COLOR_GRAY2BGR)
        skel_show[skeleton > 0] = (0, 0, 255)
        vis = cv2.addWeighted(vis, 0.9, skel_show, 0.5, 0)

        # === Линия fitLine ===
        ys, xs = np.where(skeleton > 0)
        if len(xs) > 20:
            pts = np.column_stack((xs, ys))
            [vx, vy, x0, y0] = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01)

            # --- защита от вертикальной линии ---
            if abs(vy) < 1e-6:
                # линия почти горизонтальная → рисуем горизонтальную
                x_top = int(x0)
                x_bot = int(x0)
            else:
                # обычный случай
                t_top = -y0 / vy
                t_bot = (h - y0) / vy

                # защита от inf
                t_top = max(min(t_top, 1e6), -1e6)
                t_bot = max(min(t_bot, 1e6), -1e6)

                x_top = int(x0 + vx * t_top)
                x_bot = int(x0 + vx * t_bot)

            cv2.line(vis, (x_top, 0), (x_bot, h), (0, 255, 255), 2)


            # PID точка
            if x_bottom is not None:
                cv2.circle(vis, (int(x_bottom), h - 1), 6, (0, 255, 0), -1)

        # === Текстовая панель ===
        corner_type, direction, conf, angle_deg = angle_info

        lines = [
            f"MODE: {mode}",
            f"ANGLE: {angle_deg:.1f} deg",
            f"DIR: {'LEFT' if direction>0 else 'RIGHT' if direction<0 else 'STRAIGHT'}",
            f"CONF: {conf:.2f}",
        ]

        y0 = 20
        for i, txt in enumerate(lines):
            cv2.putText(vis, txt, (10, y0 + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        return vis
