import time
import math
import numpy as np
import cv2


class AngleAnalyzer:
    def __init__(self, min_points=30, cooldown=0.2):
        self.min_points = min_points
        self.cooldown = cooldown
        self.last_time = 0.0

    def can_trigger(self):
        return (time.time() - self.last_time) > self.cooldown

    def analyze(self, mask):
        """
        mask: бинарная маска (0 / 255), где линия > 0.
        Возвращает:
            corner_type: 'left_turn' / 'right_turn' / 'straight' / 'no_line'
            direction:   +1 (лево), -1 (право), 0 (нет линии)
            confidence:  [0..1]
            angle_abs:   |угол| в градусах
        """
        ys, xs = np.where(mask > 0)
        if len(xs) < self.min_points:
            return ('no_line', 0, 0.0, 0.0)

        pts = np.column_stack((xs, ys))
        vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01)

        # === 1. Угол линии ===
        angle = math.degrees(math.atan2(float(vy), float(vx)))
        angle_abs = abs(angle)

        # 90° — идеально вертикально (прямо), 0° — горизонтально
        deviation = abs(angle_abs - 90.0)

        # Насколько угол "резкий" [0..1]
        confidence = deviation / 90.0

        # Направление:
        # angle > 0 → левый поворот
        # angle < 0 → правый поворот
        if angle > 0:
            direction = 1     # left
        else:
            direction = -1    # right

        # === Триггер резкого поворота ===
        if confidence > 0.25 and self.can_trigger():
            self.last_time = time.time()
            if direction > 0:
                return ("left_turn", 1, confidence, angle_abs)
            else:
                return ("right_turn", -1, confidence, angle_abs)

        return ("straight", direction, confidence, angle_abs)