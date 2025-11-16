import time
import math
import numpy as np
import cv2

class AngleAnalyzer:
    """
    - angle_deg: +90..0..−90
    - deviation = | angle |  
          0° → максимальное отклонение (угол)
         90° → строго вертикально (straight)
    - confidence = deviation / 90
          conf=0  → строго прямо
          conf=1  → сильный угол
    - right_turn  → angle < 0
    - left_turn   → angle > 0
    """

    def __init__(self, min_points=30, cooldown=0.25):
        self.min_points = min_points
        self.cooldown = cooldown
        self.last_time = 0.0

    def can_trigger(self):
        return (time.time() - self.last_time) > self.cooldown

    def analyze(self, skeleton):
        ys, xs = np.where(skeleton > 0)
        if len(xs) < self.min_points:
            return ('straight', 0, 0.0, 0.0)

        pts = np.column_stack((xs, ys))
        [vx, vy, x0, y0] = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01)

        # ---- Угол ----
        theta = math.degrees(math.atan2(vy, vx))   # -180..180

        # вертикаль = ±90°
        angle_deg = theta  # мы уже видели, что fitLine даёт такие значения

        # направление
        direction = 1 if angle_deg > 0 else -1 if angle_deg < 0 else 0

        # отклонение от вертикали
        deviation = abs(abs(angle_deg) - 90)     # 0..90

        # conf=1 → сильный угол (deviation=90)
        confidence = deviation / 90.0

        # ---- детекция угла ----
        if 30 <= deviation <= 70 and self.can_trigger():
            self.last_time = time.time()
            if direction < 0:
                return ('right_turn', -1, confidence, angle_deg)
            else:
                return ('left_turn', +1, confidence, angle_deg)

        return ('straight', direction, confidence, angle_deg)
