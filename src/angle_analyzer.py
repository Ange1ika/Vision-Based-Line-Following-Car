import time
import math
import numpy as np
import cv2

class AngleAnalyzer:
    """
    - angle_deg: +90..0..−90
    - deviation = |angle|
          0°  → maximal deviation from vertical (horizontal line)
         90° → strictly vertical (straight line)
    - confidence = deviation / 90
          conf=0  → perfectly straight
          conf=1  → strong angle
    - right_turn  → angle < 0
    - left_turn   → angle > 0
    """
    
    def __init__(self, min_points=30, cooldown=0.2):
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

        # ---- Angle from fitLine ----
        theta = math.degrees(math.atan2(vy, vx)) 
        # vertical = ±90°
        angle_deg = theta  # fitLine already provides this form

        # direction sign
        direction = 1 if angle_deg > 0 else -1 if angle_deg < 0 else 0

        # deviation from vertical (0..90)
        deviation = abs(angle_deg)

        # conf=1 → strong angle (deviation=90)
        confidence = abs(deviation - 90) / 90.0

        # ---- turn detection ----
        if 4 <= deviation <= 60 and self.can_trigger():
            self.last_time = time.time()
            if direction < 0:
                return ('right_turn', -1, confidence, angle_deg)
            else:
                return ('left_turn', +1, confidence, angle_deg)

        return ('straight', direction, confidence, angle_deg)
