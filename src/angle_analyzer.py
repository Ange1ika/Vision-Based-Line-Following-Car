import time
import csv
import cv2
import os
import numpy as np
import math

# Ваши классы
# from angle_analyzer import AngleAnalyzer # Будем использовать AngleAnalyzer из переработанного класса ниже
# from YoloLineDetector import YOLOLineDetector # Используем ваш YoloLineDetector
# from line_detector import LineDetector # Используем ваш LineDetector (если будете его использовать)

# Переопределим AngleAnalyzer для большей гибкости, если потребуется
class AngleAnalyzer:
    """
    Анализатор угла линии, полученной из fitLine.
    - angle_deg: Угол линии относительно вертикали.
                 +90 (вертикально вверх) .. 0 (горизонтально) .. -90 (вертикально вниз).
                 Для линии на дороге, которая идет вперед:
                 + ~90° для прямой линии.
                 Меньше 90 (ближе к 0) для поворота.
    - deviation_from_straight: Отклонение от идеальной прямой (90°). 0 - прямая, 90 - горизонтальная.
    - confidence: Как сильно линия отклоняется от прямой. 0 - прямая, 1 - сильный поворот.
    """

    def __init__(self, min_points=30, cooldown=0.2, straight_threshold_deg=85, turn_trigger_deg=65):
        self.min_points = min_points
        self.cooldown = cooldown
        self.last_trigger_time = 0.0
        self.straight_threshold_deg = straight_threshold_deg # Угол, близкий к 90, считается прямой
        self.turn_trigger_deg = turn_trigger_deg # Угол, при котором начинается маневр поворота

    def can_trigger(self):
        return (time.time() - self.last_trigger_time) > self.cooldown

    def analyze(self, skeleton):
        ys, xs = np.where(skeleton > 0)
        if len(xs) < self.min_points:
            # Недостаточно точек для анализа
            return ('no_line', 0, 0.0, 0.0)

        pts = np.column_stack((xs, ys))
        [vx, vy, x0, y0] = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01)

        # Угол от fitLine
        # atan2(vy, vx) дает угол в радианах от -pi до pi.
        # Горизонтальная линия: 0 или pi. Вертикальная: pi/2 или -pi/2.
        # Нам нужен угол относительно вертикали.
        
        # Если линия почти вертикальна (vx очень мало), vy/vx стремится к бесконечности.
        # atan2(vy, vx) дает угол относительно оси X.
        # Если линия направлена ВВЕРХ (vy отрицательно в координатах изображения), угол будет между -pi/2 и pi/2
        # Нам нужно перевести в формат, где 90° - это прямо вперед.

        # Получаем угол относительно горизонтали (0 - горизонталь, 90 - вертикаль)
        angle_rad = math.atan2(vy, vx) # [-pi, pi]
        angle_deg_raw = math.degrees(angle_rad) # [-180, 180]

        # Преобразуем так, чтобы 90 было строго вертикально, 0 - горизонтально
        # Если линия идет "вверх" по изображению (vy < 0), то atan2(vy, vx) будет в (-pi, 0)
        # Если линия идет "вниз" (vy > 0), то atan2(vy, vx) будет в (0, pi)
        # Для линии, идущей вперед по трассе, vy обычно отрицательно.
        
        # Нормализуем угол к диапазону 0-180 (если смотрим на абсолютный наклон)
        # Или к +90..0..-90 для удобства, где 90 - вертикаль.
        # Для fitLine, если линия идет вверх, vy < 0. Тогда atan2(vy, vx) будет в (-pi/2, 0) или (-pi, -pi/2)
        # Мы хотим, чтобы 90° было прямо вперед (вертикально).

        # Пересчитаем угол относительно вертикали, где 90° - прямая линия.
        # vx, vy - это вектор направления.
        # Угол между вектором (vx, vy) и вектором (0, -1) (направление "вперед" в OpenCV-координатах)
        # dot_product = vx*0 + vy*(-1) = -vy
        # magnitude_v = sqrt(vx^2 + vy^2) = 1 (т.к. fitLine нормализует)
        # magnitude_up = 1
        # cos_theta = -vy / (1*1) = -vy
        
        angle_from_vertical_rad = math.acos(max(-1.0, min(1.0, -float(vy)))) # [0, pi]
        angle_deg = math.degrees(angle_from_vertical_rad) # [0, 180]

        # Теперь angle_deg: 0 - строго горизонтально, 90 - строго вертикально, 180 - строго горизонтально в другом направлении
        # Нас интересуют углы близкие к 90.
        
        # Определим направление поворота
        # Если vx > 0, линия наклонена вправо (для камеры, смотрящей вперед)
        # Если vx < 0, линия наклонена влево
        direction_sign = 0
        if vx > 0.05: # Небольшой порог для шума
            direction_sign = -1 # Уходит вправо
        elif vx < -0.05:
            direction_sign = 1 # Уходит влево

        # Отклонение от прямой линии (90 градусов)
        deviation_from_straight = abs(angle_deg - 90) # 0 для прямой, 90 для горизонтальной

        # Уверенность в повороте (чем больше отклонение от 90, тем выше уверенность)
        confidence = deviation_from_straight / 90.0 # 0 - прямая, 1 - горизонтальная

        if confidence > 0.1 and self.can_trigger(): # Малый порог для шума
            # Проверяем на сильный поворот
            if deviation_from_straight > (90 - self.turn_trigger_deg): # Если угол значительно отличается от 90
                self.last_trigger_time = time.time()
                if direction_sign < 0:
                    return ('right_turn', -1, confidence, angle_deg)
                elif direction_sign > 0:
                    return ('left_turn', 1, confidence, angle_deg)
            
        return ('straight', direction_sign, confidence, angle_deg)