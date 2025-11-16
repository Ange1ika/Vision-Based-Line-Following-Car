import cv2
import numpy as np
import time
import math

from angle_analyzer import AngleAnalyzer
from YoloLineDetector import YOLOLineDetector   # или LineDetector

# ==== Заглушка моторов ====
class DummyMotors:
    def set_speed(self, L, R):
        pass
    def stop(self):
        pass

# ==== Заглушка камеры ====
class VideoCamera:
    def __init__(self, path):
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise RuntimeError("Не удалось открыть видео")

    def read(self):
        ret, frame = self.cap.read()
        if not ret:
            return None
        return frame

    def release(self):
        self.cap.release()


# ==== Класс OFFLINE контроллера ====
class OfflineAnalyzer:
    def __init__(self, video_path, model_path):
        self.cam = VideoCamera(video_path)
        self.motors = DummyMotors()

        self.detector = YOLOLineDetector(
            tflite_path=model_path,
            img_size=320,
            conf_thresh=0.7,
            iou_thresh=0.45,
            min_contour_area=60
        )

        self.angle = AngleAnalyzer(min_points=30, cooldown=0.0)
    
    def _visualize(self, frame, mask, skeleton, x_bottom, mode, angle_info):
        """
        Возвращает визуализированный кадр:
        - маска поверх RGB
        - скелет
        - линия fitLine
        - точка пересечения
        - текстовое состояние
        """
        vis = frame.copy()
        h, w = frame.shape[:2]

        # ===== Маска (полупрозрачная) =====
        color_mask = cv2.applyColorMap(mask, cv2.COLORMAP_TURBO)
        vis = cv2.addWeighted(vis, 0.65, color_mask, 0.35, 0)

        # ===== Скелет (красным) =====
        skel_show = cv2.cvtColor(skeleton, cv2.COLOR_GRAY2BGR)
        skel_show[skeleton > 0] = (0, 0, 255)
        vis = cv2.addWeighted(vis, 0.9, skel_show, 0.6, 0)

        # ===== FitLine линия =====
        ys, xs = np.where(skeleton > 0)
        if len(xs) > 20:
            pts = np.column_stack((xs, ys))
            [vx, vy, x0, y0] = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01)

            # точка на верхней границе
            t_top = -y0 / vy
            x_top = int(x0 + vx * t_top)

            # точка на нижней границе
            t_bot = (h - y0) / vy
            x_bot = int(x0 + vx * t_bot)

            cv2.line(vis, (x_top, 0), (x_bot, h), (0, 255, 255), 2)

            # точка пересечения PID
            if x_bottom is not None:
                cv2.circle(vis, (int(x_bottom), h - 1), 6, (0, 255, 0), -1)

            corner_type, direction, conf, angle_deg = angle_info

            cv2.putText(vis, f"MODE: {corner_type}", (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            cv2.putText(vis, f"deg={angle_deg:.1f}  dir={direction}", (10, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 200, 0), 2)

            cv2.putText(vis, f"conf={conf:.2f}", (10, 67),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 200, 0), 2)

        return vis

    def process_frame(self, frame):
        mask = self.detector.threshold(frame)
        skeleton = cv2.ximgproc.thinning(mask)
        h, w = skeleton.shape

        # === Анализ угла ===
        corner_type, direction, conf, angle_deg = self.angle.analyze(skeleton)
        #angle_info = f"{corner_type} | deg={angle_deg:.1f} | dir={direction} | conf={conf:.2f}"
        angle_info = (corner_type, direction, conf, angle_deg)

        if corner_type in ("left_turn", "right_turn"):
            return self._visualize(frame, mask, skeleton, None, corner_type, angle_info)

        # === PID точка ===
        ys, xs = np.where(skeleton > 0)
        if len(xs) < 20:
            return self._visualize(frame, mask, skeleton, None, "NO_LINE", angle_info)

        pts = np.column_stack((xs, ys))
        [vx, vy, x0, y0] = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01)

        x_bottom = None
        if abs(vy) > 1e-5:
            t = (h - y0) / vy
            x_bottom = x0 + vx * t
        else:
            x_bottom = x0

        return self._visualize(frame, mask, skeleton, x_bottom, "PID", angle_info)

    def run(self, save_path="analysis.avi", fps=5):
        # Открываем writer
        w, h = 440, 240
        out = cv2.VideoWriter(
            save_path,
            cv2.VideoWriter_fourcc(*"MJPG"),
            fps,
            (w, h)
        )
        print("Сохраняю визуализацию в:", save_path)

        while True:
            frame = self.cam.read()
            if frame is None:
                break

            vis = self.process_frame(frame)
            vis = cv2.resize(vis, (w, h))
            out.write(vis)

        out.release()
        self.cam.release()
        print("Готово!")


if __name__ == "__main__":
    analyzer = OfflineAnalyzer(
        video_path="./raw_videos/raw.avi",
        model_path="./checkpoints/yolov8n_seg_last/tflite_export/best_float32.tflite"
    )
    analyzer.run("./videos/analysis_result.avi")
