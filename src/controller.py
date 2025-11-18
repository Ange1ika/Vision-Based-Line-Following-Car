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
    Гибридный контроллер с использованием PID на основе fitLine,
    улучшенной логикой поворотов и обработки потери линии.
    """

    def __init__(self, camera, motors,
                 base_speed=50, turn_speed=25,
                 maneuver_timeout=2.5,
                 min_line_pixels=700,
                 telemetry_path="./telemetry/telemetry_log.csv",
                 use_yolo=True,
                 yolo_model_path="./checkpoints/yolov8n_seg_last/tflite_export/best_float32.tflite",
                 yolo_img_size=320,
                 yolo_conf_thresh=0.7,
                 yolo_iou_thresh=0.5,
                 Kp=0.4, Ki=0.005, Kd=0.15, K_angle=0.08):
        
        self.camera = camera
        self.motors = motors

        self.base_speed = base_speed
        self.turn_speed = turn_speed
        self.min_line_pixels = min_line_pixels

        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.K_angle = K_angle
        
        self.no_line_frames = 0
        self.no_line_max = 5
        self.no_skeleton_frames = 0
        self.no_skeleton_max = 3
        
        self.prev_error = 0
        self.integral = 0
        self.integral_limit = 10.0
        self.pid_prev_time = time.time() # Переименовал, чтобы не конфликтовало с FPS
        self.prev_frame_time = time.time()

        if use_yolo:
            self.detector = YOLOLineDetector(
                tflite_path=yolo_model_path,
                img_size=yolo_img_size,
                conf_thresh=yolo_conf_thresh,
                iou_thresh=yolo_iou_thresh,
                min_contour_area=60
            )
        else:
            raise NotImplementedError("OpenCV LineDetector не реализован в этом примере. Установите use_yolo=True.")

        self.angle_analyzer = AngleAnalyzer(
            min_points=30,
            cooldown=0.3,
            straight_threshold_deg=80,
            turn_trigger_deg=50
        )

        self.maneuver_active = False
        self.maneuver_dir = 0
        self.maneuver_start = 0.0
        self.maneuver_timeout = maneuver_timeout
        self.maneuver_cooldown = 0.5
        self.last_maneuver_time = 0.0
        self.min_turn_confidence = 0.5

        self.dir_history = []
        self.dir_hist_size = 15
        self.last_known_x_bottom = None
        self.last_known_angle = 0.0

        self.BOTTOM_ROW_OFFSET = 10

        self.telemetry_path = telemetry_path
        os.makedirs(os.path.dirname(telemetry_path) or ".", exist_ok=True)
        with open(self.telemetry_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["time", "mode", "x_bottom", "left", "right", "error", "angle_deg", "line_quality", "fps"]) # Добавили FPS

        # --- Для расчета FPS ---
        self.frame_count = 0
        self.start_time = time.time()
        self.fps = 0.0
        self.fps_avg_alpha = 0.9 # Коэффициент для скользящего среднего FPS

    def _update_dir_history(self, direction):
        self.dir_history.append(direction)
        if len(self.dir_history) > self.dir_hist_size:
            self.dir_history.pop(0)

    def _get_history_direction(self):
        if not self.dir_history:
            return 0
        return sum(self.dir_history) / len(self.dir_history)

    def _calculate_line_params(self, skeleton, h, w):
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(skeleton, 8, cv2.CV_32S)
        
        largest_skeleton_pixels = None
        max_area = 0
        
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            
            if area > self.angle_analyzer.min_points / 2:
                current_skeleton_mask = (labels == i).astype(np.uint8) * 255
                ys_current, xs_current = np.where(current_skeleton_mask > 0)
                
                if area > max_area:
                    max_area = area
                    largest_skeleton_pixels = (xs_current, ys_current)
        
        if largest_skeleton_pixels is None:
            return None, 0.0, 0.0
            
        xs, ys = largest_skeleton_pixels
        
        if len(xs) < self.angle_analyzer.min_points:
            return None, 0.0, 0.0
        
        pts = np.column_stack((xs, ys))
        
        try:
            [vx, vy, x0, y0] = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01)
        except cv2.error:
            return None, 0.0, 0.0
        
        angle_from_vertical_rad = math.acos(max(-1.0, min(1.0, -float(vy))))
        angle_deg = math.degrees(angle_from_vertical_rad)

        x_bottom = None
        if abs(vy) < 1e-6:
            x_bottom = float(x0)
        else:
            bottom_y = h - self.BOTTOM_ROW_OFFSET
            t = (bottom_y - y0) / vy
            t = max(min(t, 1e4), -1e4)
            x_bottom = float(x0 + vx * t)
            
            x_bottom = max(0.0, min(float(w - 1), x_bottom))

        line_quality = len(xs) / (h * w)

        return x_bottom, angle_deg, line_quality

    # LOG -----------------------------------------------------
    def _log(self, mode, x_bottom, error=0, angle_deg=0, line_quality=0):
        with open(self.telemetry_path, "a", newline="") as f:
            csv.writer(f).writerow([
                time.strftime("%H:%M:%S"),
                mode,
                "" if x_bottom is None else f"{x_bottom:.1f}",
                int(getattr(self, "left_speed", 0)),
                int(getattr(self, "right_speed", 0)),
                f"{error:.3f}",
                f"{angle_deg:.1f}",
                f"{line_quality:.3f}",
                f"{self.fps:.1f}" # Добавили FPS
            ])

    # PID -----------------------------------------------------
    def _pid_follow(self, x_bottom, angle_deg, width):
        center = width / 2
        
        error_x = (x_bottom - center) / center
        error_angle = (90 - angle_deg) / 90.0

        error = error_x + self.K_angle * error_angle

        now = time.time()
        dt = now - self.pid_prev_time # Используем pid_prev_time
        if dt < 0.001:
            dt = 0.001
        self.pid_prev_time = now

        self.integral += error * dt
        self.integral = max(min(self.integral, self.integral_limit), -self.integral_limit)
        
        derivative = (error - self.prev_error) / dt
        self.prev_error = error

        correction = self.Kp * error + self.Ki * self.integral + self.Kd * derivative

        L = self.base_speed + correction * self.base_speed
        R = self.base_speed - correction * self.base_speed

        max_speed = self.base_speed * 1.5
        min_speed = -self.base_speed * 0.2
        
        L = max(min(L, max_speed), min_speed)
        R = max(min(R, max_speed), min_speed)

        self.left_speed, self.right_speed = L, R
        self.motors.set_speed(int(L), int(R))

    def _reset_pid(self):
        self.integral = 0
        self.prev_error = 0
        self.pid_prev_time = time.time() # Обновить время PID

    # МАНЕВРЫ -----------------------------------------------------
    def _start_maneuver(self, direction):
        self.maneuver_active = True
        self.maneuver_dir = direction
        self.maneuver_start = time.time()
        
        self.left_speed = self.base_speed * 1.1
        self.right_speed = self.base_speed * 1.1
        self.motors.set_speed(int(self.left_speed), int(self.right_speed))
        print(f"[MANEUVER] Starting { 'LEFT' if direction > 0 else 'RIGHT' } maneuver: drive forward briefly")
    
    def _perform_maneuver(self, skeleton, h, w):
        turn_speed = self.turn_speed
        
        time_since_start = time.time() - self.maneuver_start
        forward_duration = 0.15
        
        if time_since_start < forward_duration:
            self.left_speed = self.base_speed * 1.1
            self.right_speed = self.base_speed * 1.1
            self.motors.set_speed(int(self.left_speed), int(self.right_speed))
            return False

        self.left_speed = -turn_speed * self.maneuver_dir
        self.right_speed = turn_speed * self.maneuver_dir
        self.motors.set_speed(int(self.left_speed), int(self.right_speed))

        x_bottom, angle_deg, line_quality = self._calculate_line_params(skeleton, h, w)
        
        if x_bottom is not None:
            deviation_from_straight = abs(angle_deg - 90)
            center_error = abs(x_bottom - w / 2) / (w / 2)
            
            if deviation_from_straight < (90 - self.angle_analyzer.straight_threshold_deg) and center_error < 0.2:
                print(f"[MANEUVER] Exit criteria met: angle={angle_deg:.1f}deg, center_error={center_error:.2f}")
                self.maneuver_active = False
                self.last_maneuver_time = time.time()
                self._reset_pid()
                return True
        
        if (time.time() - self.maneuver_start) > self.maneuver_timeout:
            print("[MANEUVER] Timeout, exiting maneuver (fallback)")
            self.maneuver_active = False
            self.last_maneuver_time = time.time()
            self._reset_pid()
            return True

        return False
    
    def _is_cooldown_active(self):
        return (time.time() - self.last_maneuver_time) < self.maneuver_cooldown

    # ОСНОВНОЙ ЦИКЛ (STEP) -----------------------------------------------------
    def step(self, debug=False):
        # --- Расчет FPS (скользящее среднее) ---
        current_time = time.time()
        dt = current_time - self.prev_frame_time
        if dt > 0:
            instant_fps = 1.0 / dt
            # Экспоненциальное скользящее среднее:
            self.fps = self.fps_avg_alpha * self.fps + (1.0 - self.fps_avg_alpha) * instant_fps
        self.prev_frame_time = current_time
        
        frame = self.camera.read()
        if frame is None:
            return None

        h, w = frame.shape[:2]
        
        mask = self.detector.threshold(frame)
        skeleton = cv2.ximgproc.thinning(mask)

        line_pixels = cv2.countNonZero(mask)

        if line_pixels < self.min_line_pixels:
            self.no_line_frames += 1
        else:
            self.no_line_frames = 0
            if self.no_line_frames == 0 and len(self.dir_history) > 0:
                self.dir_history.clear()

        if self.maneuver_active:
            self._print_action(f"MANEUVER_{'LEFT' if self.maneuver_dir > 0 else 'RIGHT'}")
            
            maneuver_finished = self._perform_maneuver(skeleton, h, w)
            
            if not maneuver_finished:
                mode = f"MANEUVER_{'LEFT' if self.maneuver_dir > 0 else 'RIGHT'}"
                self._log(mode, None, 0, 0, 0) 
                if debug:
                    angle_info = ("maneuver", self.maneuver_dir, 1.0, 0)
                    return self._visualize(frame, mask, skeleton, None, mode, angle_info)
                return None
            else:
                print("[MANEUVER] Finished, resuming normal operation.")
                self.last_known_x_bottom = None

        if self.no_line_frames >= self.no_line_max:
            if self._is_cooldown_active():
                self.motors.stop()
                mode = "NO_LINE_COOLDOWN"
                self._print_action(mode)
                self._log(mode, None, 0, 0, 0)
                if debug:
                    return self._visualize(frame, mask, skeleton, None, mode, ("straight", 0, 0, 0))
                return None

            avg_dir = self._get_history_direction()
            turn_magnitude = self.turn_speed * 0.7

            if self.last_known_x_bottom is not None:
                center_error = (self.last_known_x_bottom - w / 2) / (w / 2)
                if abs(center_error) > 0.1:
                    if center_error > 0:
                        self.left_speed = turn_magnitude
                        self.right_speed = -turn_magnitude
                        mode = "SEARCH_RIGHT"
                    else:
                        self.left_speed = -turn_magnitude
                        self.right_speed = turn_magnitude
                        mode = "SEARCH_LEFT"
                else:
                    if avg_dir > 0.1:
                        self.left_speed = -turn_magnitude
                        self.right_speed = turn_magnitude
                        mode = "SEARCH_LEFT"
                    elif avg_dir < -0.1:
                        self.left_speed = turn_magnitude
                        self.right_speed = -turn_magnitude
                        mode = "SEARCH_RIGHT"
                    else:
                        self.motors.stop()
                        mode = "NO_LINE_STOP"
            else:
                self.motors.stop()
                mode = "NO_LINE_STOP"

            self.motors.set_speed(int(self.left_speed), int(self.right_speed))
            self._print_action(mode)
            self._log(mode, None, 0, 0, 0)

            if debug:
                return self._visualize(frame, mask, skeleton, None, mode, ("straight", 0, 0, 0))
            return None
        
        x_bottom, line_angle_deg, line_quality = self._calculate_line_params(skeleton, h, w)
        
        if x_bottom is not None:
            self.last_known_x_bottom = x_bottom
            self.last_known_angle = line_angle_deg

        if x_bottom is None or line_quality < 0.005:
            self.no_skeleton_frames += 1
        else:
            self.no_skeleton_frames = 0

        if x_bottom is None and self.no_skeleton_frames >= self.no_skeleton_max:
            if self.last_known_x_bottom is not None:
                self.motors.set_speed(self.base_speed, self.base_speed)
                mode = "SKELETON_LOST_HOLD"
                self._print_action(mode)
                self._log(mode, self.last_known_x_bottom, 0, self.last_known_angle, line_quality)
                if debug:
                    return self._visualize(frame, mask, skeleton, self.last_known_x_bottom, mode, ("straight", 0, 0, self.last_known_angle))
                return None
            else:
                self.motors.stop()
                mode = "SKELETON_LOST_STOP"
                self._print_action(mode)
                self._log(mode, None, 0, 0, line_quality)
                if debug:
                    return self._visualize(frame, mask, skeleton, None, mode, ("straight", 0, 0, 0))
                return None
        elif x_bottom is None:
            mode = "SKELETON_TEMPORARY_LOSS"
            self._print_action(mode)
            self._log(mode, None, 0, 0, line_quality)
            if debug:
                return self._visualize(frame, mask, skeleton, None, mode, ("straight", 0, 0, 0))
            return None

        corner_type, direction_sign, conf, angle_deg_analyzer = self.angle_analyzer.analyze(skeleton)
        angle_info = (corner_type, direction_sign, conf, angle_deg_analyzer)

        if corner_type in ["right_turn", "left_turn"]:
            if conf < self.min_turn_confidence:
                pass
            elif self._is_cooldown_active():
                time_since = time.time() - self.last_maneuver_time
                print(f"[ANGLE] {corner_type} detected but cooldown active ({time_since:.2f}s left)")
            else:
                print(f"[ANGLE] Detected {corner_type}, conf={conf:.2f}, starting maneuver.")
                self._start_maneuver(direction_sign)
                mode = f"START_MANEUVER_{corner_type.upper()}"
                self._log(mode, x_bottom, 0, angle_deg_analyzer, line_quality)
                self._reset_pid()
                if debug:
                    return self._visualize(frame, mask, skeleton, x_bottom, mode, angle_info)
                return None

        center = w / 2
        direction_sign_for_history = np.sign(x_bottom - center)
        self._update_dir_history(direction_sign_for_history)

        error = (x_bottom - center) / center + self.K_angle * ((90 - line_angle_deg) / 90.0)
        
        self._print_action("PID_FOLLOW")
        self._pid_follow(x_bottom, line_angle_deg, w)
        mode = "PID"

        self._log(mode, x_bottom, error, line_angle_deg, line_quality)

        if debug:
            return self._visualize(frame, mask, skeleton, x_bottom, mode, angle_info)

        return None

    def close(self):
        self.camera.release()
        self.motors.stop()

    def _visualize(self, frame, mask, skeleton, x_bottom_vis, mode, angle_info):
        vis = frame.copy()
        h, w = frame.shape[:2]

        color_mask = cv2.applyColorMap(mask, cv2.COLORMAP_TURBO)
        vis = cv2.addWeighted(vis, 0.65, color_mask, 0.35, 0)

        skel_show = cv2.cvtColor(skeleton, cv2.COLOR_GRAY2BGR)
        skel_show[skeleton > 0] = (0, 0, 255)
        vis = cv2.addWeighted(vis, 0.9, skel_show, 0.5, 0)

        ys, xs = np.where(skeleton > 0)
        
        xs_vis = xs
        ys_vis = ys

        if len(xs_vis) >= self.angle_analyzer.min_points:
            pts_vis = np.column_stack((xs_vis, ys_vis))
            
            try:
                [vx, vy, x0, y0] = cv2.fitLine(pts_vis, cv2.DIST_L2, 0, 0.01, 0.01)

                if abs(vy) < 1e-6:
                    x_top = int(x0)
                    x_bot = int(x0)
                else:
                    t_top = (-y0) / vy
                    x_top = int(x0 + vx * t_top)
                    
                    bottom_y = h - self.BOTTOM_ROW_OFFSET
                    t_bot = (bottom_y - y0) / vy
                    x_bot = int(x0 + vx * t_bot)

                x_top = max(-1000, min(w + 1000, x_top))
                x_bot = max(-1000, min(w + 1000, x_bot))
                
                cv2.line(vis, (x_top, 0), (x_bot, h - 1), (0, 255, 255), 2)

                if x_bottom_vis is not None:
                    cv2.circle(vis, (int(x_bottom_vis), h - 1), 6, (0, 255, 0), -1)
            except cv2.error:
                pass

        corner_type, direction, conf, angle_deg_analyzer = angle_info
        L = int(getattr(self, "left_speed", 0))
        R = int(getattr(self, "right_speed", 0))

        cooldown_active = self._is_cooldown_active()
        if cooldown_active:
            time_remaining = self.maneuver_cooldown - (time.time() - self.last_maneuver_time)
            cooldown_text = f"CD:{time_remaining:.1f}s"
        else:
            cooldown_text = "READY"

        avg_dir = self._get_history_direction()
        hist_text = "LEFT" if avg_dir > 0.2 else ("RIGHT" if avg_dir < -0.2 else "CENTER")

        lines = [
            f"MODE: {mode}",
            f"L_SPD: {L}",
            f"R_SPD: {R}",
            f"ANGLE: {angle_deg_analyzer:.1f}deg (Analyzer)",
            f"TYPE: {corner_type}",
            f"CONF: {conf:.2f}",
            f"TURN CD: {cooldown_text}",
            f"HIST: {hist_text} ({avg_dir:.2f})",
            f"LAST X: {self.last_known_x_bottom:.1f}" if self.last_known_x_bottom is not None else "LAST X: N/A",
            f"LINE Pixels: {cv2.countNonZero(mask)}",
            f"FPS: {self.fps:.1f}" # Добавили FPS в визуализацию
        ]

        y0 = 22
        for i, txt in enumerate(lines):
            cv2.putText(vis, txt, (10, y0 + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        return vis

    def _print_action(self, mode):
        if not hasattr(self, '_last_printed_mode') or self._last_printed_mode != mode:
            print(f"[ACTION] {mode}")
            self._last_printed_mode = mode
