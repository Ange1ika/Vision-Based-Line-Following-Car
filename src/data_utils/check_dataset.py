import os
import random
import cv2
import numpy as np
from glob import glob
import matplotlib.pyplot as plt

# -------------------------
# НАСТРОЙКИ
# -------------------------
DATA_ROOT = "/home/angelika/Desktop/Seoul/Intelligent Control/DATA_annotation/datasets/dataset3"  # путь к merged_dataset

import os
from glob import glob

MIN_POINTS = 6     # если меньше -> исправлять
TARGET_POINTS = 20 # сколько точек хотим получить после интерполяции


def interpolate_points(polygon, target_n):
    """Интерполирует точки между соседними вершинами полигона."""
    new_poly = []

    for i in range(len(polygon)):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % len(polygon)]  # замкнутый контур

        # кладём первую точку
        new_poly.append((x1, y1))

        # добавляем промежуточные точки
        for t in range(1, target_n):
            alpha = t / target_n
            nx = x1 * (1 - alpha) + x2 * alpha
            ny = y1 * (1 - alpha) + y2 * alpha
            new_poly.append((nx, ny))

    return new_poly


def fix_label_file(path):
    with open(path, "r") as f:
        lines = f.readlines()

    new_lines = []

    for line in lines:
        parts = line.strip().split()

        if len(parts) < 6:
            continue

        cls_id = parts[0]
        cx, cy, w, h = parts[1:5]
        coords = parts[5:]

        # Парсим точки
        poly = []
        for i in range(0, len(coords), 2):
            x = float(coords[i])
            y = float(coords[i + 1])
            poly.append((x, y))

        # Если точек достаточно — оставляем
        if len(poly) >= MIN_POINTS:
            new_lines.append(line)
            continue

        # иначе — восстанавливаем
        fixed_poly = interpolate_points(poly, target_n=TARGET_POINTS)

        # Собираем новую строку
        new_coords = " ".join([f"{x:.6f} {y:.6f}" for x, y in fixed_poly])
        new_line = f"{cls_id} {cx} {cy} {w} {h} {new_coords}\n"

        new_lines.append(new_line)

    # Перезаписываем файл
    with open(path, "w") as f:
        f.writelines(new_lines)


def process_all_labels():
    label_files = glob(os.path.join(DATA_ROOT, "labels", "**", "*.txt"), recursive=True)

    print(f"Найдено файлов: {len(label_files)}\n")

    for lb in label_files:
        fix_label_file(lb)
        print(f"✔ Исправлено: {lb}")

    print("\nГотово! Маски автоматически сглажены.")


if __name__ == "__main__":
    process_all_labels()

