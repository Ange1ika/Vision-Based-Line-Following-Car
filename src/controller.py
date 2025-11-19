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
    Clean version:
    - No skeleton at all (fitLine on mask only)
    - PID continues working even when mask is missing (uses last valid x_bottom)
    - Correct turn directions (aligned with MotorController)
    - Search mode not blocked by cooldown
    - Minimal cooldown after maneuvers
    - Full telemetry for transient analysis
    """

    def __init__(self, camera, motors,
                 base_speed=50, turn_speed=25,
                 maneuver_timeout=1.3,
                 min_line_pixels=700,
                 telemetry_path="./telemetry/telemetry_log.csv",
                 use_yolo=True,
                 yolo_model_path="./checkpoints/yolov8n_seg_last/tflite_export/best_float32.tflite",
                 yolo_img_size=320,
                 yolo_conf_thresh=0.7,
                 yolo_iou_thresh=0.5):

        self.camera = camera
        self.motors = motors

        # speeds
        self.base_speed = base_speed
        self.turn_speed = turn_speed
        self.min_line_pixels = min_line_pixels

        # PID coefficients
        self.Kp = 0.35
        self.Ki = 0.002
        self.Kd = 0.12
        self.K_angle = 0.2

        # Line-loss counters
        self.no_line_frames = 0
        self.no_line_max = 4

        # PID internal state
        self.prev_error = 0.0
        self.integral = 0.0
        self.integral_limit = 10.0
        self.prev_time = time.time()      # для PID
        self.prev_frame_time = time.time()  # для FPS
        self.fps = 0.0
        self.last_u = 0.0                 # последний управляющий сигнал (correction)

        # last valid values for fallback PID
        self.last_valid_x_bottom = None
        self.last_valid_angle = 0.0

        # detector
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

        # angle analyzer works on mask now
        self.angle = AngleAnalyzer(min_points=30, cooldown=0.15)

        # === MANEUVERS ===
        self.maneuver_active = False
        self.maneuver_dir = 0          # +1 left, -1 right
        self.maneuver_start = 0.0
        self.maneuver_timeout = maneuver_timeout
        self.maneuver_cooldown = 0.05
        self.last_maneuver_time = 0.0
        self.min_turn_confidence = 0.6

        # direction history (for search direction)
        self.dir_history = []
        self.dir_hist_size = 10

        # constants
        self.BOTTOM_ROW_OFFSET = 10

        # telemetry
        self.telemetry_path = telemetry_path
        os.makedirs(os.path.dirname(telemetry_path) or ".", exist_ok=True)
        with open(self.telemetry_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "time",
                "mode",
                "x_bottom",
                "x_valid",
                "error",
                "angle",
                "left_speed",
                "right_speed",
                "u_control",
                "fps",
                "line_pixels"
            ])

        self.left_speed = 0.0
        self.right_speed = 0.0

    # ---------------------------------------------------------
    def _update_dir_history(self, direction):
        self.dir_history.append(direction)
        if len(self.dir_history) > self.dir_hist_size:
            self.dir_history.pop(0)

    def _get_history_direction(self):
        if not self.dir_history:
            return 0.0
        return sum(self.dir_history) / len(self.dir_history)

    # ---------------------------------------------------------
    def _quick_detect_x(self, mask, h, w):
        """
        Fast x detection in lower ROI (no fitLine).
        Used in search mode to quickly re-acquire line.
        """
        roi = mask[int(h * 0.60):h, :]
        ys, xs = np.where(roi > 0)
        if len(xs) < 40:
            return None
        return float(np.mean(xs))

    # ---------------------------------------------------------
    def _calculate_x_bottom(self, mask, h, w):
        ys, xs = np.where(mask > 0)
        idx = ys > h * 0.2
        xs = xs[idx]
        ys = ys[idx]

        if len(xs) < 10:
            return None, 0.0

        pts = np.column_stack((xs, ys))
        vx, vy, x0, y0 = cv2.fitLine(
            pts, cv2.DIST_L2, 0, 0.01, 0.01
        )

        angle_deg = math.degrees(math.atan2(float(vy), float(vx)))

        if abs(vy) < 1e-6:
            x_bottom = float(x0)
        else:
            bottom_y = h - self.BOTTOM_ROW_OFFSET
            t = (bottom_y - y0) / vy
            t = np.clip(t, -1e6, 1e6)
            x_bottom = float(x0 + vx * t)

        return x_bottom, angle_deg

    # ---------------------------------------------------------
    def _log(self, mode, x_bottom, error=0.0, angle=0.0, line_pixels=0):
        """
        Full telemetry row:
        time, mode, x_bottom, x_valid, error, angle,
        left_speed, right_speed, u_control, fps, line_pixels
        """
        x_valid = self.last_valid_x_bottom
        with open(self.telemetry_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                time.strftime("%H:%M:%S"),
                mode,
                "" if x_bottom is None else f"{x_bottom:.1f}",
                "" if x_valid is None else f"{x_valid:.1f}",
                f"{error:.3f}",
                f"{angle:.1f}",
                int(self.left_speed),
                int(self.right_speed),
                f"{self.last_u:.3f}",
                f"{self.fps:.2f}",
                int(line_pixels)
            ])

    # ---------------------------------------------------------
    def _pid_follow(self, x_bottom, angle_deg, width):
        center = width / 2
        error_x = (x_bottom - center) / center
        error_angle = angle_deg / 90.0
        error = error_x + self.K_angle * error_angle

        now = time.time()
        dt = max(now - self.prev_time, 0.001)
        self.prev_time = now

        self.integral += error * dt
        self.integral = np.clip(self.integral, -self.integral_limit, self.integral_limit)

        derivative = (error - self.prev_error) / dt
        self.prev_error = error

        correction = (
            self.Kp * error +
            self.Ki * self.integral +
            self.Kd * derivative
        )
        self.last_u = float(correction)

        L = self.base_speed + correction * self.base_speed
        R = self.base_speed - correction * self.base_speed

        max_speed = self.base_speed * 1.8
        min_speed = -self.base_speed * 0.3

        L = float(np.clip(L, min_speed, max_speed))
        R = float(np.clip(R, min_speed, max_speed))

        self.left_speed, self.right_speed = L, R
        self.motors.set_speed(int(L), int(R))

    def _reset_pid(self):
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_time = time.time()
        self.last_u = 0.0

    # ================= MANEUVERS =============================
    def _start_maneuver(self, direction):
        # direction: +1 (left), -1 (right)
        self.maneuver_active = True
        self.maneuver_dir = direction
        self.maneuver_start = time.time()
        self.motors.move_forward(int(self.base_speed * 1.2), 0.10)

    def _perform_maneuver(self, mask, h, w):
        turn_speed = self.turn_speed

        self.left_speed = -turn_speed * self.maneuver_dir
        self.right_speed = turn_speed * self.maneuver_dir
        self.motors.set_speed(int(self.left_speed), int(self.right_speed))

        x2, _ = self._calculate_x_bottom(mask, h, w)
        if x2 is not None:
            print("[MANEUVER] normalized, exit")
            self.maneuver_active = False
            self.last_maneuver_time = time.time()
            self._reset_pid()
            return True

        if time.time() - self.maneuver_start > self.maneuver_timeout:
            print("[MANEUVER] timeout, exit")
            self.maneuver_active = False
            self.last_maneuver_time = time.time()
            self._reset_pid()
            return True

        return False

    def _is_cooldown_active(self):
        return (time.time() - self.last_maneuver_time) < self.maneuver_cooldown

    # ================= MAIN STEP =============================
    def step(self, debug=False):
        frame = self.camera.read()
        if frame is None:
            return None

        # ===== FPS update =====
        now_f = time.time()
        dt_f = now_f - self.prev_frame_time
        if dt_f > 0:
            instant_fps = 1.0 / dt_f
            # скользящее среднее FPS
            self.fps = 0.9 * self.fps + 0.1 * instant_fps
        self.prev_frame_time = now_f

        mask = self.detector.threshold(frame)
        h, w = mask.shape

        # line pixel count for telemetry
        line_pixels = cv2.countNonZero(mask)

        # ===== MANEUVER MODE =====
        if self.maneuver_active:
            mode = f"MANEUVER_{'LEFT' if self.maneuver_dir > 0 else 'RIGHT'}"
            self._print_action(mode)

            x_bottom, _ = self._calculate_x_bottom(mask, h, w)
            finished = self._perform_maneuver(mask, h, w)

            if not finished:
                self._log(mode, x_bottom, error=0.0, angle=0.0, line_pixels=line_pixels)
                return None if not debug else \
                    self._visualize(frame, mask, x_bottom, mode, ("maneuver", self.maneuver_dir, 1.0, 0.0))

        # ===== LINE LOST =====
        if line_pixels < self.min_line_pixels:
            self.no_line_frames += 1
        else:
            self.no_line_frames = 0

        if self.no_line_frames >= self.no_line_max:
            avg_dir = self._get_history_direction()

            # --- fast re-acquire in bottom ROI ---
            x_quick = self._quick_detect_x(mask, h, w)

            if x_quick is not None:
                # Found line during search → immediately go to PID
                self.last_valid_x_bottom = x_quick
                self.last_valid_angle = 0.0
                mode = "SEARCH_FOUND"
                self._pid_follow(x_quick, 0.0, w)
                self._log(mode, x_quick, error=0.0, angle=0.0, line_pixels=line_pixels)
                self._print_action(mode)
                return None

            # otherwise — spin to search
            if avg_dir > 0.1:
                mode = "SEARCH_LEFT"
                self.left_speed = -self.turn_speed
                self.right_speed = self.turn_speed
                self.motors.set_speed(int(self.left_speed), int(self.right_speed))
                self._update_dir_history(-1)

            elif avg_dir < -0.1:
                mode = "SEARCH_RIGHT"
                self.left_speed = self.turn_speed
                self.right_speed = -self.turn_speed
                self.motors.set_speed(int(self.left_speed), int(self.right_speed))
                self._update_dir_history(1)

            else:
                # unknown previous side
                mode = "NO_LINE"
                self.motors.stop()
                self.left_speed = 0.0
                self.right_speed = 0.0
                self._update_dir_history(0)

            self._log(mode, None, error=0.0, angle=0.0, line_pixels=line_pixels)
            self._print_action(mode)
            return None if not debug else \
                self._visualize(frame, mask, None, mode, ("straight", 0, 0.0, 0.0))

        # ===== ANGLE ANALYSIS =====
        corner_type, direction, conf, angle_deg = self.angle.analyze(mask)
        angle_info = (corner_type, direction, conf, angle_deg)

        if corner_type in ["left_turn", "right_turn"]:
            if conf >= self.min_turn_confidence and not self._is_cooldown_active():
                print(f"[ANGLE] {corner_type} conf={conf:.2f} → maneuver")
                self._start_maneuver(direction)
                mode = f"START_{corner_type.upper()}"
                self._log(mode, None, error=0.0, angle=angle_deg, line_pixels=line_pixels)
                return None if not debug else \
                    self._visualize(frame, mask, None, mode, angle_info)

        # ===== PID FOLLOWING =====
        x_bottom, line_angle = self._calculate_x_bottom(mask, h, w)

        if x_bottom is None:
            # fallback: use last known good measurement
            if self.last_valid_x_bottom is not None:
                mode = "PID_FALLBACK"
                self._pid_follow(self.last_valid_x_bottom, self.last_valid_angle, w)
                self._log(mode, None, error=0.0, angle=self.last_valid_angle, line_pixels=line_pixels)
                self._print_action(mode)
                return None
            else:
                mode = "NO_MASK_FORWARD"
                self.left_speed = self.base_speed
                self.right_speed = self.base_speed
                self.motors.set_speed(self.base_speed, self.base_speed)
                self._print_action(mode)
                self._log(mode, None, error=0.0, angle=0.0, line_pixels=line_pixels)
                return None

        # valid measurement: save as last valid
        self.last_valid_x_bottom = x_bottom
        self.last_valid_angle = line_angle

        center = w / 2
        direction_sign = np.sign(x_bottom - center)
        self._update_dir_history(direction_sign)

        error = (x_bottom - center) / center

        mode = "PID"
        self._pid_follow(x_bottom, angle_deg, w)
        self._log(mode, x_bottom, error=error, angle=angle_deg, line_pixels=line_pixels)

        return None if not debug else \
            self._visualize(frame, mask, x_bottom, mode, angle_info)

    # ---------------------------------------------------------
    def close(self):
        self.camera.release()
        self.motors.stop()

    # ---------------------------------------------------------
    def _visualize(self, frame, mask, x_bottom, mode, angle_info):
        vis = frame.copy()
        h, w = frame.shape[:2]

        color_mask = cv2.applyColorMap(mask, cv2.COLORMAP_TURBO)
        vis = cv2.addWeighted(vis, 0.65, color_mask, 0.35, 0)

        ys, xs = np.where(mask > 0)
        idx = ys > h * 0.4
        xs = xs[idx]
        ys = ys[idx]

        if len(xs) >= 10:
            pts = np.column_stack((xs, ys))
            vx, vy, x0, y0 = cv2.fitLine(
                pts, cv2.DIST_L2, 0, 0.01, 0.01
            )

            if abs(vy) < 1e-6:
                x_top = int(x0)
                x_bot = int(x0)
            else:
                t_top = -y0 / vy
                bottom_y = h - self.BOTTOM_ROW_OFFSET
                t_bot = (bottom_y - y0) / vy
                t_top = np.clip(t_top, -1e6, 1e6)
                t_bot = np.clip(t_bot, -1e6, 1e6)
                x_top = int(x0 + vx * t_top)
                x_bot = int(x0 + vx * t_bot)

            cv2.line(vis, (x_top, 0), (x_bot, h - 1), (0, 255, 255), 2)

        if x_bottom is not None:
            cv2.circle(vis, (int(x_bottom), h - 1), 6, (0, 255, 0), -1)

        corner_type, direction, conf, angle_deg = angle_info
        L = int(self.left_speed)
        R = int(self.right_speed)

        cooldown = "READY" if not self._is_cooldown_active() else "CD"

        avg_dir = self._get_history_direction()
        hist_text = "L" if avg_dir > 0.2 else ("R" if avg_dir < -0.2 else "C")

        x_valid = self.last_valid_x_bottom

        lines = [
            f"MODE: {mode}",
            f"ANGLE: {angle_deg:.1f}",
            f"TYPE: {corner_type}",
            f"CONF: {conf:.2f}",
            f"HIST: {hist_text} ({avg_dir:.2f})",
            f"L/R: {L} / {R}",
            f"x: {'' if x_bottom is None else f'{x_bottom:.1f}'}  "
            f"x_valid: {'' if x_valid is None else f'{x_valid:.1f}'}",
            f"u: {self.last_u:.3f}",
            f"FPS: {self.fps:.1f}"
        ]

        y0 = 22
        for i, txt in enumerate(lines):
            cv2.putText(
                vis, txt, (10, y0 + i * 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 2
            )

        return vis

    def _print_action(self, mode):
        print(mode)
