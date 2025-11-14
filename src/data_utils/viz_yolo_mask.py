import os
import random
import cv2
import numpy as np
from glob import glob
import matplotlib.pyplot as plt

# -------------------------
# НАСТРОЙКИ
# -------------------------
DATA_ROOT = "/home/angelika/Desktop/Seoul/Intelligent Control/DATA_annotation/datasets/dataset7"  # путь к merged_dataset
SPLIT = "train"   # или "val"
NUM_IMAGES = 30
IMG_DIR = os.path.join(DATA_ROOT, "images", SPLIT)
LBL_DIR = os.path.join(DATA_ROOT, "labels", SPLIT)
# -------------------------

def load_yolo_segmentation_mask(label_path, img_w, img_h):
    """
    Восстанавливает маску по YOLO segmentation labels:
    class cx cy w h x1 y1 x2 y2 ...
    """
    mask = np.zeros((img_h, img_w), dtype=np.uint8)

    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()

            if len(parts) < 6:
                continue  # мало точек

            # первые 5 элементов: cls, cx, cy, w, h
            coords = parts[5:]

            # переводим нормированные координаты полигона
            poly = []
            for i in range(0, len(coords), 2):
                x = float(coords[i]) * img_w
                y = float(coords[i+1]) * img_h
                poly.append([int(x), int(y)])
            
            poly = np.array([poly], dtype=np.int32)

            # рисуем filled polygon
            cv2.fillPoly(mask, poly, 255)

    return mask


def visualize_random_masks():
    # список всех изображений
    images = sorted(glob(os.path.join(IMG_DIR, "*.jpg")) + 
                    glob(os.path.join(IMG_DIR, "*.png")))

    if len(images) == 0:
        print("❌ Нет изображений в", IMG_DIR)
        return

    chosen = random.sample(images, min(NUM_IMAGES, len(images)))

    for img_path in chosen:

        # соответствующий .txt
        name = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(LBL_DIR, name + ".txt")

        img = cv2.imread(img_path)
        img_h, img_w = img.shape[:2]

        if os.path.exists(label_path):
            mask = load_yolo_segmentation_mask(label_path, img_w, img_h)
        else:
            mask = np.zeros((img_h, img_w), dtype=np.uint8)

        # создаём цветную маску
        mask_color = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
        mask_vis = cv2.addWeighted(img, 0.7, mask_color, 0.3, 0)

        # matplotlib для удобства
        plt.figure(figsize=(6, 4))
        plt.title(name)
        plt.imshow(cv2.cvtColor(mask_vis, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.show()


if __name__ == "__main__":
    visualize_random_masks()
