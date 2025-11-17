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
    Hybrid version with corrected logic:
    - Fixed race condition in maneuvers
    - No frame loss after maneuver exit
    - Unified coordinate calculations
    - Direction history always updates
    - Cooldown applied everywhere
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
        self.K_angle = 0.0
        
        # === Line / skeleton loss (anti-shaking) ===
        self.no_line_frames = 0
        self.no_line_max = 5        # treat as lost only after 3 consecutive frames

        self.no_skeleton_frames = 0
        self.no_skeleton_max = 5     # same for skeleton
        
        self.prev_error = 0
        self.integral = 0
        self.integral_limit = 10.0
        self.prev_time = time.time()

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

        # angle analyzer
        self.angle = AngleAnalyzer(min_points=30, cooldown=0.2)

        # === MANEUVERS ===
        self.maneuver_active = False
        self.maneuver_dir = 0
        self.maneuver_start = 0.0
        self.maneuver_timeout = maneuver_timeout
        self.maneuver_cooldown = 0.1  # seconds after maneuver
        self.last_maneuver_time = 0.0
        self.min_turn_confidence = 0.6

        # === Direction history ===
        self.dir_history = []
        self.dir_hist_size = 10

        # === Coordinate constants ===
        self.BOTTOM_ROW_OFFSET = 10 

        # telemetry
        self.telemetry_path = telemetry_path
        os.makedirs(os.path.dirname(telemetry_path) or ".", exist_ok=True)
        with open(self.telemetry_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["time", "mode", "x_bottom", "left", "right", "error", "angle"])

    def _update_dir_history(self, direction):
        """direction: -1 / 0 / +1 — line trend"""
        self.dir_history.append(direction)
        if len(self.dir_history) > self.dir_hist_size:
            self.dir_history.pop(0)

    def _get_history_direction(self):
        """Returns average direction: -1..1"""
        if not self.dir_history:
            return 0
        return sum(self.dir_history) / len(self.dir_history)

    def _calculate_x_bottom(self, skeleton, h, w):
        """
        Unified method for calculating bottom line point using fitLine.
        Returns (x_bottom, angle_deg) or (None, 0)
        """
        ys, xs = np.where(skeleton > 0)
        
        # take only the lower part of the skeleton
        idx = ys > h * 0.2
        xs = xs[idx]
        ys = ys[idx]
        
        if len(xs) < 10:
            return None, 0
        
        pts = np.column_stack((xs, ys))
        vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01)
        
        # slope angle
        angle_deg = math.degrees(math.atan2(float(vy), float(vx)))
        
        if abs(vy) < 1e-6:
            x_bottom = float(x0)
        else:
            # single value for the bottom row
            bottom_y = h - self.BOTTOM_ROW_OFFSET
            t = (bottom_y - y0) / vy
            t = max(min(t, 1e6), -1e6)
            x_bottom = float(x0 + vx * t)
        
        return x_bottom, angle_deg

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

        L = self.base_speed + correction * self.base_speed
        R = self.base_speed - correction * self.base_speed

        max_speed = self.base_speed * 1.8
        min_speed = -self.base_speed * 0.3
        
        L = max(min(L, max_speed), min_speed)
        R = max(min(R, max_speed), min_speed)

        self.left_speed, self.right_speed = L, R
        self.motors.set_speed(int(L), int(R))

    def _reset_pid(self):
        """Reset PID controller before transition"""
        self.integral = 0
        self.prev_error = 0
        self.prev_time = time.time()

    # MANEUVERS -----------------------------------------------------
    def _start_maneuver(self, direction):
        """Start maneuver: drive forward before turning"""
        self.maneuver_active = True
        self.maneuver_dir = direction
        self.maneuver_start = time.time()
        self.motors.move_forward(int(self.base_speed * 1.2), 0.24)
    
    def _perform_maneuver(self, mask, skeleton, h, w):
        turn_speed = self.turn_speed
        dir_fixed = -self.maneuver_dir

        self.left_speed  = turn_speed * dir_fixed
        self.right_speed = -turn_speed * dir_fixed
        self.motors.set_speed(self.left_speed, self.right_speed)

        x2, ang2 = self._calculate_x_bottom(skeleton, h, w)
        if x2 is not None:
            print("[MANEUVER] Angle normalized, exiting")
            self.maneuver_active = False
            self.last_maneuver_time = time.time()
            self._reset_pid()
            return True
        
        # === Timeout — fallback ===
        if (time.time() - self.maneuver_start) > self.maneuver_timeout:
            print("[MANEUVER] Timeout, exiting maneuver")
            self.maneuver_active = False
            self.last_maneuver_time = time.time()
            self._reset_pid()
            return True

        return False
    
    def _is_cooldown_active(self):
        """Check if cooldown is active"""
        return (time.time() - self.last_maneuver_time) < self.maneuver_cooldown

    # STEP -----------------------------------------------------
    def step(self, debug=False):
        frame = self.camera.read()
        if frame is None:
            return None

        mask = self.detector.threshold(frame)
        h, w = mask.shape
        skeleton = cv2.ximgproc.thinning(mask)

        # === IF IN MANEUVER ===
        if self.maneuver_active:
            self._print_action(f"MANEUVER_{'LEFT' if self.maneuver_dir > 0 else 'RIGHT'}")
            
            x_bottom, _ = self._calculate_x_bottom(skeleton, h, w)
            maneuver_finished = self._perform_maneuver(mask, skeleton, h, w)
            
            # If still in maneuver — return visualization
            if not maneuver_finished:
                mode = f"MANEUVER_{'LEFT' if self.maneuver_dir > 0 else 'RIGHT'}"
                self._log(mode, x_bottom)
                if debug:
                    angle_info = ("maneuver", self.maneuver_dir, 1.0, 0)
                    return self._visualize(frame, mask, skeleton, x_bottom, mode, angle_info)
                return None
            # Maneuver finished — continue processing
            print("[MANEUVER] Continuing to PID with current frame")

        # === LINE LOST ===
        line_pixels = cv2.countNonZero(mask)

        if line_pixels < self.min_line_pixels:
            self.no_line_frames += 1
        else:
            self.no_line_frames = 0

        # react only if line is gone for several frames in a row
        if self.no_line_frames >= self.no_line_max:
            # get averaged direction
            avg_dir = self._get_history_direction()

            # check cooldown
            if self._is_cooldown_active():
                self.motors.stop()
                mode = "NO_LINE_COOLDOWN"
                self._print_action(mode)
                self._log(mode, None)
                if debug:
                    return self._visualize(frame, mask, skeleton, None, mode, ("straight", 0, 0, 0))
                return None

            if avg_dir > 0.1:
                # line was on the left → search left
                self.left_speed = -self.turn_speed
                self.right_speed = self.turn_speed
                self.motors.set_speed(self.left_speed, self.right_speed)
                mode = "SEARCH_LEFT"
                self._update_dir_history(-1)

            elif avg_dir < -0.1:
                # line was on the right → search right
                self.left_speed = self.turn_speed
                self.right_speed = -self.turn_speed
                self.motors.set_speed(self.left_speed, self.right_speed)
                mode = "SEARCH_RIGHT"
                self._update_dir_history(1)

            else:
                # neutral history → stop
                self.motors.stop()
                mode = "NO_LINE"
                self._update_dir_history(0)

            self._print_action(mode)
            self._log(mode, None)

            if debug:
                return self._visualize(frame, mask, skeleton, None, mode, ("straight", 0, 0, 0))
            return None

        # if no_line_frames < threshold → continue normal processing

        # === ANGLE ANALYSIS ===
        corner_type, direction, conf, angle_deg = self.angle.analyze(skeleton)
        angle_info = (corner_type, direction, conf, angle_deg)

        # turn detection with all checks
        if corner_type in ["right_turn", "left_turn"]:
            if conf < self.min_turn_confidence:
                pass
            elif self._is_cooldown_active():
                time_since = time.time() - self.last_maneuver_time
                print(f"[ANGLE] {corner_type} detected but cooldown active ({time_since:.2f}s)")
            else:
                print(f"[ANGLE] Detected {corner_type}, conf={conf:.2f}, starting maneuver")
                self._start_maneuver(direction)
                mode = f"START_MANEUVER_{corner_type.upper()}"
                self._log(mode, None, 0, angle_deg)
                if debug:
                    return self._visualize(frame, mask, skeleton, None, mode, angle_info)
                return None

        # === PID FOLLOWING ===
        x_bottom, line_angle = self._calculate_x_bottom(skeleton, h, w)

        if x_bottom is None:
            self.no_skeleton_frames += 1
        else:
            self.no_skeleton_frames = 0

        if x_bottom is None and self.no_skeleton_frames >= self.no_skeleton_max:
            # only if skeleton gone for several frames — stop and search
            self.motors.stop()
            mode = "NO_SKELETON"
            self._log(mode, None)
            self._print_action(mode)
            if debug:
                return self._visualize(frame, mask, skeleton, None, mode, angle_info)
            return None
        elif x_bottom is None:
            # short skeleton drop — skip frame without braking
            mode = "NO_SKELETON_WAIT"
            self._print_action(mode)
            self._log(mode, None)
            if debug:
                return self._visualize(frame, mask, skeleton, None, mode, angle_info)
            return None

        # update history
        center = w / 2
        direction_sign = np.sign(x_bottom - center)
        self._update_dir_history(direction_sign)

        # compute error for logs
        error = (x_bottom - center) / center + self.K_angle * (angle_deg / 90.0)
        
        self._print_action("PID")
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
        """Visualizer"""
        vis = frame.copy()
        h, w = frame.shape[:2]

        # Mask
        color_mask = cv2.applyColorMap(mask, cv2.COLORMAP_TURBO)
        vis = cv2.addWeighted(vis, 0.65, color_mask, 0.35, 0)

        # Skeleton
        skel_show = cv2.cvtColor(skeleton, cv2.COLOR_GRAY2BGR)
        skel_show[skeleton > 0] = (0, 0, 255)
        vis = cv2.addWeighted(vis, 0.9, skel_show, 0.5, 0)

        # fitLine
        ys, xs = np.where(skeleton > 0)
        idx = ys > h * 0.4
        xs = xs[idx]
        ys = ys[idx]
        
        if len(xs) >= 10:
            pts = np.column_stack((xs, ys))
            [vx, vy, x0, y0] = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01)

            if abs(vy) < 1e-6:
                x_top = int(x0)
                x_bot = int(x0)
            else:
                t_top = -y0 / vy
                bottom_y = h - self.BOTTOM_ROW_OFFSET
                t_bot = (bottom_y - y0) / vy
                t_top = max(min(t_top, 1e6), -1e6)
                t_bot = max(min(t_bot, 1e6), -1e6)
                x_top = int(x0 + vx * t_top)
                x_bot = int(x0 + vx * t_bot)

            cv2.line(vis, (x_top, 0), (x_bot, h - 1), (0, 255, 255), 2)

            if x_bottom is not None:
                cv2.circle(vis, (int(x_bottom), h - 1), 6, (0, 255, 0), -1)

        # Text panel
        corner_type, direction, conf, angle_deg = angle_info
        L = int(getattr(self, "left_speed", 0))
        R = int(getattr(self, "right_speed", 0))

        # Cooldown status
        cooldown_active = self._is_cooldown_active()
        if cooldown_active:
            time_remaining = self.maneuver_cooldown - (time.time() - self.last_maneuver_time)
            cooldown_text = f"CD:{time_remaining:.1f}s"
        else:
            cooldown_text = "READY"

        # Direction history
        avg_dir = self._get_history_direction()
        hist_text = "L" if avg_dir > 0.2 else ("R" if avg_dir < -0.2 else "C")

        lines = [
            f"MODE: {mode}",
            f"ANGLE: {angle_deg:.1f}deg",
            f"TYPE: {corner_type}",
            f"CONF: {conf:.2f}",
            f"TURN: {cooldown_text}",
            f"HIST: {hist_text} ({avg_dir:.2f})",
        ]

        y0 = 22
        for i, txt in enumerate(lines):
            cv2.putText(vis, txt, (10, y0 + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        return vis

    def _print_action(self, mode):
        print(mode)
