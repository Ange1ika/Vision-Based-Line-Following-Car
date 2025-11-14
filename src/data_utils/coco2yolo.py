
import os
import json
import cv2
import shutil
import numpy as np
from glob import glob
from pycocotools import mask as mask_utils
from tqdm import tqdm



def clip(v, lo, hi):
    return max(lo, min(v, hi))


def convert_coco_to_yolo_fixed(coco_json, out_dir):
    with open(coco_json, "r") as f:
        coco = json.load(f)

    os.makedirs(out_dir, exist_ok=True)

    images = {img["id"]: img for img in coco["images"]}

    for ann in coco["annotations"]:

        img = images[ann["image_id"]]
        w, h = img["width"], img["height"]

        file = os.path.splitext(img["file_name"])[0]
        out_path = os.path.join(out_dir, f"{file}.txt")

        cls = ann["category_id"]
        seg = ann["segmentation"]

        # пропускаем RLE
        if isinstance(seg, dict):
            continue

        seg = seg[0]  # берем первый полигон
        xs = seg[::2]
        ys = seg[1::2]

        # Клипаем все координаты (исправляет твои ошибки!)
        fixed = []
        for x, y in zip(xs, ys):
            x = clip(x, 0, w - 1)
            y = clip(y, 0, h - 1)
            fixed.append(x / w)
            fixed.append(y / h)

        # Проверка на валидность
        if len(fixed) < 6:
            continue

        # YOLO bbox
        x_min = min(xs)
        x_max = max(xs)
        y_min = min(ys)
        y_max = max(ys)

        cx = ((x_min + x_max) / 2) / w
        cy = ((y_min + y_max) / 2) / h
        bw = (x_max - x_min) / w
        bh = (y_max - y_min) / h

        # Формируем строку
        line = (
            f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f} " +
            " ".join(f"{p:.6f}" for p in fixed)
        )

        with open(out_path, "a") as f:
            f.write(line + "\n")

    print("✓ Готово! Все маски исправлены, обрезаны и конвертированы.")

if __name__ == "__main__":
    # ============ НАСТРОЙКИ ============
    COCO_JSON = "/home/angelika/Desktop/Seoul/Intelligent Control/DATA_annotation/datasets/dataset5/annotations.json"

    OUT = "/home/angelika/Desktop/Seoul/Intelligent Control/DATA_annotation/datasets/dataset6"      

    convert_coco_to_yolo_fixed(COCO_JSON, OUT)