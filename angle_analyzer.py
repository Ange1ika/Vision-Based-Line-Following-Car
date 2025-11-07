import time
class AngleAnalyzer:
    """
    Выявляет острые/90° повороты:
      - по смещению линии в верхней части кадра,
      - по «региональным» шаблонам : лево/центр/право.
    Возвращает: (corner_type, direction, confidence)
      corner_type: 'right_angle' | 'sharp_turn' | 'straight'
      direction: -1 (лево) | +1 (право) | 0
      confidence: 0..1
    """
    def __init__(self,
                 turn_threshold_ratio=0.65,
                 confirm_frames=3,
                 region_confirm_frames=4):
        self.turn_threshold_ratio = turn_threshold_ratio
        self.confirm_frames = confirm_frames
        self.region_confirm_frames = region_confirm_frames

        self.left_cnt = 0
        self.right_cnt = 0
        self.last_decision_time = 0.0
        self.cooldown = 0.2

        # для регионов
        self.region_left_cnt = 0
        self.region_right_cnt = 0

    def can_trigger(self):
        return (time.time() - self.last_decision_time) > self.cooldown

    def update_by_upper(self, upper_x, width):
        if upper_x is None:
            self.left_cnt = max(0, self.left_cnt - 1)
            self.right_cnt = max(0, self.right_cnt - 1)
            return None

        center = width / 2
        left_thresh = center * (1 - self.turn_threshold_ratio)
        right_thresh = center * (1 + self.turn_threshold_ratio)

        if upper_x < left_thresh:
            self.left_cnt += 1
            self.right_cnt = 0
        elif upper_x > right_thresh:
            self.right_cnt += 1
            self.left_cnt = 0
        else:
            self.left_cnt = 0
            self.right_cnt = 0

        if self.left_cnt >= self.confirm_frames:
            return ('right_angle', -1, min(1.0, self.left_cnt / (self.confirm_frames + 1)))
        if self.right_cnt >= self.confirm_frames:
            return ('right_angle', +1, min(1.0, self.right_cnt / (self.confirm_frames + 1)))

        return None

    def update_by_regions(self, upper_counts, lower_counts):
        """
        Простая региональная эвристика 90°:
          - если сверху сильный левый и слабый центр/право -> левый угол,
          - если сверху сильный правый и слабый центр/лево -> правый угол.
        """
        if not upper_counts or len(upper_counts) < 3:
            self.region_left_cnt = max(0, self.region_left_cnt - 1)
            self.region_right_cnt = max(0, self.region_right_cnt - 1)
            return None

        L, C, R = upper_counts
        # нормализация для устойчивости
        s = max(1, L + C + R)
        l_ratio, c_ratio, r_ratio = L / s, C / s, R / s

        # пороги можно калибровать
        if l_ratio > 0.75 and r_ratio < 0.25:
            self.region_left_cnt += 1
            self.region_right_cnt = 0
        elif r_ratio > 0.55 and l_ratio < 0.25:
            self.region_right_cnt += 1
            self.region_left_cnt = 0
        else:
            self.region_left_cnt = max(0, self.region_left_cnt - 1)
            self.region_right_cnt = max(0, self.region_right_cnt - 1)

        if self.region_left_cnt >= self.region_confirm_frames:
            return ('right_angle', -1, min(1.0, self.region_left_cnt / (self.region_confirm_frames + 1)))
        if self.region_right_cnt >= self.region_confirm_frames:
            return ('right_angle', +1, min(1.0, self.region_right_cnt / (self.region_confirm_frames + 1)))

        return None

    def decide(self, upper_x, width, upper_counts, lower_counts):
        """
        Комбинированное решение:
          1) быстрый канал по верхней половине,
          2) подтверждение регионом,
          3) иначе – straight.
        """
        if not self.can_trigger():
            return ('straight', 0, 0.0)

        # 1) по координате верхнего центроида
        by_upper = self.update_by_upper(upper_x, width)
        if by_upper:
            self.last_decision_time = time.time()
            return by_upper

        # 2) по региональным шаблонам
        by_regions = self.update_by_regions(upper_counts, lower_counts)
        if by_regions:
            self.last_decision_time = time.time()
            return by_regions

        return ('straight', 0, 0.0)
