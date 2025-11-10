import time
import csv
import cv2
import os
import numpy as np
from line_detector import LineDetector
from angle_analyzer import AngleAnalyzer
from YoloLineDetector import YOLOLineDetector

class VisionController:
    """
    Объединяет: детекцию, анализ углов, манёвры, адаптивное замедление,
    запись телеметрии.
    """
    def __init__(self, camera, motors,
                 base_speed=50, turn_speed=25,
                 slowdown_factor=0.8,
                 maneuver_timeout=0.2,
                 min_line_pixels=700,
                 telemetry_path="./telemetry/telemetry_log.csv",
                # === Новые параметры для выбора детектора ===
                 use_yolo=False,
                 yolo_model_path="./checkpoints/yolov8n_seg_last/tflite_export/best_float32.tflite",
                 yolo_img_size=320,
                 yolo_conf_thresh=0.7,
                 yolo_iou_thresh=0.5):
        self.camera = camera
        self.motors = motors

        # === Выбор детектора ===
        self.use_yolo = use_yolo
        if use_yolo:
            print("Using YOLOv8-seg")
            self.detector = YOLOLineDetector(
                tflite_path=yolo_model_path,
                img_size=yolo_img_size,
                conf_thresh=yolo_conf_thresh,
                iou_thresh=yolo_iou_thresh
            )
        else:
            print(" Using OpenCV HSV")
            self.detector = LineDetector()
            
        self.angles = AngleAnalyzer(turn_threshold_ratio=0.35,  # ИЗМЕНЕНО: было 0.40
                                    confirm_frames=2,            # ИЗМЕНЕНО: было 3
                                    region_confirm_frames=2)

        self.base_speed = base_speed
        self.turn_speed = turn_speed
        self.slowdown_factor = slowdown_factor
        self.maneuver_timeout = maneuver_timeout
        self.min_line_pixels = min_line_pixels

        self.last_known_x = None
        self.maneuver_active = False
        self.maneuver_dir = 0
        self.maneuver_start = 0.0
        self.current_speed = base_speed
        self.current_turn_factor = 0.9
        self.debug_last_state = None
        
        self.prev_time = time.time()
        self.fps = 0.0
                
        self.raw_video_path = "./raw_videos/raw.avi"

        # --- Телеметрия ---
        self.telemetry_path = telemetry_path
        self.left_speed = 0
        self.right_speed = 0
        self.current_state = "idle"

        # создаём CSV с заголовком
        with open(self.telemetry_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp", "upper_x", "lower_x",
                "left_speed", "right_speed",
                "state", "maneuver_dir"
            ])
        
        
        ## два видео одновременно не пишем, тогда два видеокодека с разными разрешениями нужны 
        self.save_raw_video = True
        self.raw_video_writer = None

        if self.save_raw_video:
            os.makedirs(os.path.dirname(self.raw_video_path) or ".", exist_ok=True)

            # Попробуем несколько кодеков, чтобы не было ошибок FFmpeg
            possible_codecs = [
                ('MJPG', 'avi'),
                ('mp4v', 'mp4'),
                ('XVID', 'avi')
            ]

            frame_size = (int(camera.width), int(camera.height))
            fps = 20.0
            for fourcc_name, ext in possible_codecs:
                path = os.path.splitext(self.raw_video_path)[0] + f".{ext}"
                fourcc = cv2.VideoWriter_fourcc(*fourcc_name)
                writer = cv2.VideoWriter(path, fourcc, fps, frame_size)

                if writer.isOpened():
                    print(f"[INFO] 🎥 Raw video recording enabled → {path} ({fourcc_name})")
                    self.raw_video_writer = writer
                    break
                else:
                    writer.release()

            if self.raw_video_writer is None:
                print("⚠️ Failed to initialize any codec! Raw video will not be saved.")


    # ------------------------------------------------------------
    def _log_telemetry(self, upper_x, lower_x):
        """Записывает текущие данные в CSV"""
        with open(self.telemetry_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                time.strftime("%H:%M:%S"),
                upper_x if upper_x is not None else "",
                lower_x if lower_x is not None else "",
                int(self.left_speed),
                int(self.right_speed),
                self.current_state,
                self.maneuver_dir
            ])

    # ------------------------------------------------------------
    def _debug_state(self, msg):
        """Выводит состояние, если оно изменилось"""
        if msg != self.debug_last_state:
            print(f"[DEBUG] {time.strftime('%H:%M:%S')} → {msg}")
            self.debug_last_state = msg
        self.current_state = msg  # сохраняем состояние в лог

    # ------------------------------------------------------------
    def _start_maneuver(self, direction):
        self._debug_state(f"MANEUVER_START_{'LEFT' if direction < 0 else 'RIGHT'}")
        self.maneuver_active = True
        self.maneuver_dir = direction
        self.maneuver_start = time.time()
        self.motors.stop()
        time.sleep(0.001)
        self.motors.move_forward(int(self.base_speed * 1.1), 0.15)

    def _perform_maneuver(self, frame, mask, lower_x):  # ИЗМЕНЕНО: добавлен lower_x
        self._debug_state(f"MANEUVER_TURN_{'LEFT' if self.maneuver_dir<0 else 'RIGHT'}")
        self.left_speed = self.turn_speed * self.maneuver_dir
        self.right_speed = -self.turn_speed * self.maneuver_dir
        self.motors.set_speed(self.left_speed, self.right_speed)
        self.current_speed = self.base_speed * self.slowdown_factor

        line_pixels = cv2.countNonZero(mask)
        if line_pixels > self.min_line_pixels and lower_x is not None:
            self._debug_state("LINE_FOUND_EXIT_MANEUVER")
            self.maneuver_active = False
            self.motors.stop()
            time.sleep(0.001)
            return

        if (time.time() - self.maneuver_start) > self.maneuver_timeout:
            self._debug_state("TIMEOUT_EXIT_MANEUVER")
            self.maneuver_active = False
            self.motors.stop()
            time.sleep(0.001)

    # ------------------------------------------------------------
    def _move_towards(self, point_x, img_width):
        if point_x is None:
            if self.last_known_x is None:
                self._debug_state("LINE_LOST_STOP")
                self.left_speed = self.right_speed = 0
                #self.motors.stop()
                return
            point_x = self.last_known_x
        else:
            self._debug_state("FOLLOW_LINE")
            self.last_known_x = point_x

        center_x = img_width / 2
        error = point_x - center_x
        error_normalized = error / center_x
        speed = self.current_speed
        turn_factor = self.current_turn_factor
        turn_adjustment = speed * turn_factor * error_normalized

        self.left_speed = speed + turn_adjustment
        self.right_speed = speed - turn_adjustment

        max_speed = speed * 1.8
        self.left_speed = max(min(self.left_speed, max_speed), -speed * 0.3)
        self.right_speed = max(min(self.right_speed, max_speed), -speed * 0.3)
        self.motors.set_speed(int(self.left_speed), int(self.right_speed))

    # ------------------------------------------------------------
    def step(self, debug=False):
        frame = self.camera.read()
        if frame is None:
            return None
        
        if self.save_raw_video and self.raw_video_writer is not None:
            self.raw_video_writer.write(frame)
            
        mask = self.detector.threshold(frame)
        upper_mask, lower_mask = self.detector.split_upper_lower(mask)

        upper_x, upper_disp = self.detector.largest_contour_center_x(upper_mask)
        lower_x, lower_disp = self.detector.largest_contour_center_x(lower_mask)
        
        # ИЗМЕНЕНО: сделал порог мягче и добавил логирование
        if upper_disp > 60:  # было 40
            if upper_x is not None and debug:
                print(f"[WARN] Upper line dispersion too high: {upper_disp:.1f}")
            upper_x = None

        h, w = frame.shape[:2]
        upper_counts = self.detector.roi_bins(upper_mask, bins=3)
        lower_counts = self.detector.roi_bins(lower_mask, bins=3)

        if self.maneuver_active:
            self._perform_maneuver(frame, mask, lower_x)  # ИЗМЕНЕНО: передаём lower_x
        else:
            corner_type, direction, conf = self.angles.decide(
                upper_x=upper_x, width=w,
                upper_counts=upper_counts, lower_counts=lower_counts
            )

            if corner_type == 'right_angle' and conf >= 0.5:  # было 0.7
                if debug:
                    print(f"[TURN] Detected {direction} turn, conf={conf:.2f}")
                self._start_maneuver(direction)
            else:
                if lower_x is not None or self.last_known_x is not None:
                    self._move_towards(lower_x, w)
                else:
                    self._debug_state("NO_LINE_STOP")
                    self.motors.stop()

        # записываем телеметрию каждый кадр
        self._log_telemetry(upper_x, lower_x)
        
        # === ⏱ FPS-счётчик ===
        current_time = time.time()
        dt = current_time - self.prev_time
        if dt > 0:
            instant_fps = 1.0 / dt
            self.fps = 0.9 * self.fps + 0.1 * instant_fps
        self.prev_time = current_time

        if not debug:
            return None

        vis = self.detector.visualize(frame, mask, upper_x, lower_x)

        # === Панель детекции линии ===
        h, w = vis.shape[:2]
        panel_width = 120
        line_pixels = cv2.countNonZero(mask)
        min_required = 700

        panel = np.zeros((h, panel_width, 3), dtype=np.uint8)

        ratio = min(line_pixels / min_required, 1.0)
        bar_height = int(h * ratio)

        cv2.rectangle(panel,
                      (20, h - bar_height),
                      (80, h),
                      (0, 255, 0) if line_pixels >= min_required else (0, 0, 255),
                      -1)

        cv2.putText(panel, f"{line_pixels}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(panel, f"/{min_required}", (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(panel, "PIXELS", (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                # Индикатор детектора
        detector_text = "YOLO" if self.use_yolo else "OpenCV"
        detector_color = (255, 150, 0) if self.use_yolo else (0, 200, 255)
        cv2.putText(panel, detector_text, (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, detector_color, 2)
             
        cv2.putText(panel, f"FPS:{self.fps:.1f}", (10, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        vis = np.hstack([vis, panel])

        if self.maneuver_active:
            cv2.putText(vis, f"MANEUVER: {'LEFT' if self.maneuver_dir<0 else 'RIGHT'}",
                        (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        else:
            cv2.putText(vis, f"U:{upper_x}  L:{lower_x}", (6, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)

        return vis

    def close(self):
        """Останавливает запись видео и освобождает ресурсы"""
        if self.raw_video_writer is not None:
            self.raw_video_writer.release()
        self.camera.release()
        self.motors.stop()