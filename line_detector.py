import cv2
import numpy as np


class LineDetector:
    """
    Отвечает за:
      - бинаризацию по HSV (черная линия),
      - разделение на верх/низ,
      - центроид линии,
      - «региональную» оценку (как в Adilnasceng: лево/центр/право).
    """
    def __init__(self,
                 hsv_black_lower=(0, 0, 0),
                 hsv_black_upper=(179, 240, 50),
                 min_contour_area=60):
        self.hsv_black_lower = np.array(hsv_black_lower, dtype=np.uint8)
        self.hsv_black_upper = np.array(hsv_black_upper, dtype=np.uint8)
        self.min_contour_area = min_contour_area

    def threshold(self, frame_bgr):
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.hsv_black_lower, self.hsv_black_upper)
        # морфология для устойчивости
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  np.ones((3, 3), np.uint8), iterations=1)
        return mask

    def split_upper_lower(self, mask):
        h = mask.shape[0]
        return mask[:h // 2, :], mask[h // 2:, :]
    
    def largest_contour_center_x(self, mask, min_area=None):
        """
        Вычисляет центр линии по всем контурам (взвешенно по площади).
        Возвращает:
            cx (float or None): взвешенный центр
            dispersion (float): разброс (сигма_x)
        """
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return None, 0.0

        min_area = min_area or self.min_contour_area
        total_area = 0
        weighted_sum = 0
        all_centers = []

        for c in cnts:
            area = cv2.contourArea(c)
            if area < min_area:
                continue
            M = cv2.moments(c)
            if M["m00"] == 0:
                continue
            cx = M["m10"] / M["m00"]
            weighted_sum += cx * area
            total_area += area
            all_centers.append(cx)

        if total_area == 0:
            return None, 0.0

        weighted_center = weighted_sum / total_area
        dispersion = float(np.std(all_centers)) if len(all_centers) > 1 else 0.0

        return int(weighted_center), dispersion


    def roi_bins(self, mask, bins=3):
        """Подсчет активных пикселей по колонкам ROI: [left, center, right]"""
        h, w = mask.shape
        step = w // bins
        counts = []
        for i in range(bins):
            x0 = i * step
            x1 = (i + 1) * step if i < bins - 1 else w
            sub = mask[:, x0:x1]
            counts.append(int(cv2.countNonZero(sub)))
        return counts  # [left, center, right]

    def visualize(self, frame, mask, upper_x, lower_x):
        out = frame.copy()
        h, w = out.shape[:2]
        # полупрозрачная маска
        color_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        out = cv2.addWeighted(out, 0.75, color_mask, 0.25, 0)

        # линия раздела кадра
        cv2.line(out, (0, h // 2), (w, h // 2), (0, 255, 255), 1)

        if upper_x is not None:
            cv2.circle(out, (upper_x, h // 4), 4, (0, 255, 255), -1)
            cv2.putText(out, f"U:{upper_x}", (upper_x + 5, h // 4 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        if lower_x is not None:
            cv2.circle(out, (lower_x, (3 * h) // 4), 4, (0, 255, 0), -1)
            cv2.putText(out, f"L:{lower_x}", (lower_x + 5, (3 * h) // 4 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
        return out
