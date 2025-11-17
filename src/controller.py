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
    Гибридная версия:
    - Новый PID с постоянным движением
    - Новый AngleAnalyzer (fitLine)
    - Старая логика маневров с проездом вперед
    """

    def __init__(self, camera, motors,
                 base_speed=50, turn_speed=25,
                 maneuver_timeout=1.5,
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

        # PID коэффициенты
        self.Kp = 1.2
        self.Ki = 0.0
        self.Kd = 0.05
        self.K_angle = 0.0
        
        self.prev_error = 0
        self.integral = 0
        self.integral_limit = 10.0
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

        # анализатор углов
        self.angle = AngleAnalyzer(min_points=30, cooldown=0.5)

        # === МАНЕВРЫ (из старого кода) ===
        self.maneuver_active = False
        self.maneuver_dir = 0
        self.maneuver_start = 0.0
        self.maneuver_timeout = maneuver_timeout
        self.last_known_x = None
        self.maneuver_cooldown = 0.1  # секунды после маневра
        self.last_maneuver_time = 0.0
        self.min_turn_confidence = 0.6  # минимальная уверенность для маневра

        # телеметрия
        self.telemetry_path = telemetry_path
        os.makedirs(os.path.dirname(telemetry_path) or ".", exist_ok=True)
        with open(self.telemetry_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["time", "mode", "x_bottom", "left", "right", "error", "angle"])

    # LOG -----------------------------------------------------
    def _log(self, mode, x_bottom, error=0, angle=0):
        with open(self.telemetry_path, "a", newline="") as f:
            csv.writer(f).writerow([
                time.strftime("%H:%M:%S"),
                mode,
                "" if x_bottom is None else f"{x_bottom:.1f}",
                int(getattr(self, "left_speed", 0)),
                int(getattr(self, "right_speed", 0)),
                f"{error:.3f}",
                f"{angle:.1f}"
            ])

    # PID -----------------------------------------------------
    def _pid_follow(self, x_bottom, angle_deg, width):
        center = width / 2
        error_x = (x_bottom - center) / center

        error_angle = angle_deg / 90.0
        error = error_x + self.K_angle * error_angle

        now = time.time()
        dt = now - self.prev_time
        if dt < 0.001:
            dt = 0.001
        self.prev_time = now

        self.integral += error * dt
        self.integral = max(min(self.integral, self.integral_limit), -self.integral_limit)
        
        derivative = (error - self.prev_error) / dt
        self.prev_error = error

        correction = self.Kp * error + self.Ki * self.integral + self.Kd * derivative

        # ПОСТОЯННОЕ ДВИЖЕНИЕ ВПЕРЕД + коррекция
        L = self.base_speed + correction * self.base_speed * 0.8
        R = self.base_speed - correction * self.base_speed * 0.8

        max_speed = self.base_speed * 2.0
        min_speed = -self.base_speed * 0.5
        
        L = max(min(L, max_speed), min_speed)
        R = max(min(R, max_speed), min_speed)

        self.left_speed, self.right_speed = L, R
        self.motors.set_speed(int(L), int(R))

    # МАНЕВРЫ (из старого кода) -----------------------------------------------------
    def _start_maneuver(self, direction):
        """Начало маневра: едем вперед перед поворотом"""
        self.maneuver_active = True
        self.maneuver_dir = direction
        self.maneuver_start = time.time()
        # Проезжаем вперед
        self.motors.move_forward(int(self.base_speed * 1.1), 0.15)

    def _perform_maneuver(self, mask, x_bottom):
        """
        Выполняем поворот НА МЕСТЕ, но продолжаем искать линию.
        Как только линия найдена — выходим из маневра.
        """
        # Поворот на месте
        self.left_speed = self.turn_speed * self.maneuver_dir
        self.right_speed = -self.turn_speed * self.maneuver_dir
        self.motors.set_speed(self.left_speed, self.right_speed)

        # Проверяем: линия вернулась?
        line_pixels = cv2.countNonZero(mask)
        if line_pixels > self.min_line_pixels and x_bottom is not None:
            print(f"[MANEUVER] Line found, exiting maneuver")
            self.maneuver_active = False
            self.last_maneuver_time = time.time()  # Запоминаем время выхода
            # СБРОС PID для чистого перехода
            self.integral = 0
            self.prev_error = 0
            self.prev_time = time.time()
            return

        # Таймаут
        if (time.time() - self.maneuver_start) > self.maneuver_timeout:
            print(f"[MANEUVER] Timeout, exiting maneuver")
            self.maneuver_active = False
            self.last_maneuver_time = time.time()  # Запоминаем время выхода
            # СБРОС PID
            self.integral = 0
            self.prev_error = 0
            self.prev_time = time.time()

    # STEP -----------------------------------------------------
    def step(self, debug=False):
        frame = self.camera.read()
        if frame is None:
            return None

        mask = self.detector.threshold(frame)
        h, w = mask.shape

        # === ЕСЛИ В МАНЕВРЕ ===
        if self.maneuver_active:
            # Продолжаем искать линию во время поворота
            skeleton = cv2.ximgproc.thinning(mask)
            ys, xs = np.where(skeleton > 0)
            
            x_bottom = None
            if len(xs) >= 10:
                pts = np.column_stack((xs, ys))
                [vx, vy, x0, y0] = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01)
                if abs(vy) > 1e-6:
                    t = (h - y0) / vy
                    t = max(min(t, 1e6), -1e6)
                    x_bottom = float(x0 + vx * t)
                else:
                    x_bottom = float(x0)

            self._perform_maneuver(mask, x_bottom)
            
            # ВАЖНО: если маневр завершился, НЕ возвращаем None!
            # Продолжаем выполнение и переходим к PID
            if self.maneuver_active:  # Маневр ещё активен
                mode = f"MANEUVER_{'LEFT' if self.maneuver_dir<0 else 'RIGHT'}"
                self._log(mode, x_bottom)
                if debug:
                    return self._visualize(frame, mask, skeleton if len(xs)>=10 else np.zeros_like(mask), 
                                          x_bottom, mode, ("maneuver",self.maneuver_dir,1.0,0))
                return None
            # Иначе маневр завершён, продолжаем к PID ниже

        # === ОБЫЧНЫЙ РЕЖИМ ===
        # линия потеряна
        if cv2.countNonZero(mask) < self.min_line_pixels:
            self.motors.stop()
            mode = "NO_LINE"
            self._log(mode, None)
            if debug:
                return self._visualize(frame, mask, np.zeros_like(mask), None, mode, ("straight",0,0,0))
            return None

        skeleton = cv2.ximgproc.thinning(mask)

        # === АНАЛИЗ УГЛОВ ===
        corner_type, direction, conf, angle_deg = self.angle.analyze(skeleton)
        angle_info = (corner_type, direction, conf, angle_deg)

        # Детекция резких углов с проверками:
        # 1. Достаточная уверенность (conf > threshold)
        # 2. Прошло достаточно времени после последнего маневра (cooldown)
        time_since_last_maneuver = time.time() - self.last_maneuver_time
        
        if corner_type in ["right_turn", "left_turn"]:
            if conf < self.min_turn_confidence:
                # Уверенность слишком низкая
                pass
            elif time_since_last_maneuver <= self.maneuver_cooldown:
                # Ещё в cooldown
                print(f"[ANGLE] {corner_type} detected but cooldown active ({time_since_last_maneuver:.2f}s)")
            else:
                # ВСЕ УСЛОВИЯ ВЫПОЛНЕНЫ - запускаем маневр!
                print(f"[ANGLE] Detected {corner_type}, conf={conf:.2f}, starting maneuver")
                self._start_maneuver(direction)
                mode = f"START_MANEUVER_{corner_type.upper()}"
                self._log(mode, None, 0, angle_deg)
                if debug:
                    return self._visualize(frame, mask, skeleton, None, mode, angle_info)
                return None


        # === PID СЛЕДОВАНИЕ ===
        ys, xs = np.where(skeleton > 0)
        if len(xs) < 10:
            # Едем вперед, используя последнюю известную позицию
            if self.last_known_x is not None:
                x_bottom = self.last_known_x
            else:
                self.motors.stop()
                mode = "NO_SKELETON"
                self._log(mode, None)
                if debug:
                    return self._visualize(frame, mask, skeleton, None, mode, angle_info)
                return None
        else:
            pts = np.column_stack((xs, ys))
            [vx, vy, x0, y0] = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01)
            
            if abs(vy) < 1e-6:
                x_bottom = float(x0)
            else:
                t = (h - y0) / vy
                t = max(min(t, 1e6), -1e6)
                x_bottom = float(x0 + vx * t)
            
            self.last_known_x = x_bottom

        # Вычисляем ошибку для логирования
        center = w / 2
        error = (x_bottom - center) / center + self.K_angle * (angle_deg / 90.0)

        self._pid_follow(x_bottom, angle_deg, w)
        mode = "PID"

        self._log(mode, x_bottom, error, angle_deg)

        if debug:
            return self._visualize(frame, mask, skeleton, x_bottom, mode, angle_info)

        return None

    def close(self):
        self.camera.release()
        self.motors.stop()

    def _visualize(self, frame, mask, skeleton, x_bottom, mode, angle_info):
        """Визуализатор"""
        vis = frame.copy()
        h, w = frame.shape[:2]

        # Маска
        color_mask = cv2.applyColorMap(mask, cv2.COLORMAP_TURBO)
        vis = cv2.addWeighted(vis, 0.65, color_mask, 0.35, 0)

        # Скелет
        skel_show = cv2.cvtColor(skeleton, cv2.COLOR_GRAY2BGR)
        skel_show[skeleton > 0] = (0, 0, 255)
        vis = cv2.addWeighted(vis, 0.9, skel_show, 0.5, 0)

        # Линия fitLine
        ys, xs = np.where(skeleton > 0)
        if len(xs) > 20:
            pts = np.column_stack((xs, ys))
            [vx, vy, x0, y0] = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01)

            if abs(vy) < 1e-6:
                x_top = int(x0)
                x_bot = int(x0)
            else:
                t_top = -y0 / vy
                t_bot = (h - y0) / vy
                t_top = max(min(t_top, 1e6), -1e6)
                t_bot = max(min(t_bot, 1e6), -1e6)
                x_top = int(x0 + vx * t_top)
                x_bot = int(x0 + vx * t_bot)

            cv2.line(vis, (x_top, 0), (x_bot, h), (0, 255, 255), 2)

            if x_bottom is not None:
                cv2.circle(vis, (int(x_bottom), h - 1), 6, (0, 255, 0), -1)

        # Текстовая панель
        corner_type, direction, conf, angle_deg = angle_info
        L = int(getattr(self, "left_speed", 0))
        R = int(getattr(self, "right_speed", 0))

        # Cooldown статус
        time_since_maneuver = time.time() - self.last_maneuver_time
        cooldown_active = time_since_maneuver < self.maneuver_cooldown
        cooldown_text = f"CD:{self.maneuver_cooldown - time_since_maneuver:.1f}s" if cooldown_active else "READY"

        lines = [
            f"MODE: {mode}",
            f"ANGLE: {angle_deg:.1f}deg",
            f"TYPE: {corner_type}",
            f"CONF: {conf:.2f}",
           # f"MOTORS: L={L} R={R}",
            f"TURN: {cooldown_text}",
        ]

        y0 = 22
        for i, txt in enumerate(lines):
            cv2.putText(vis, txt, (10, y0 + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        return vis