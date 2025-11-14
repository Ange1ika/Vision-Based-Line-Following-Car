import os
import json
import random
import cv2
import numpy as np
from pycocotools import mask as mask_utils

# -----------------------------
# НАСТРОЙКИ
# -----------------------------
IMAGES_DIR = "/home/angelika/Desktop/Seoul/Intelligent Control/DATA_annotation/datasets/dataset3/images"
ANNOT_PATH = "/home/angelika/Desktop/Seoul/Intelligent Control/DATA_annotation/datasets/dataset3/coco/annotations.json"
SHOW = False
SAVE = True
SAVE_DIR = "./vis_coco_masks"
# -----------------------------

os.makedirs(SAVE_DIR, exist_ok=True)

# Загружаем COCO аннотации
with open(ANNOT_PATH, "r") as f:
    coco = json.load(f)

images = {img["id"]: img for img in coco["images"]}
annotations = coco["annotations"]
categories = {cat["id"]: cat["name"] for cat in coco["categories"]}


def decode_mask(ann, image_h, image_w):
    """
    Возвращает бинарную маску 0/1.
    Поддержка polygon + RLE.
    """
    seg = ann["segmentation"]

    # Если segmentation == RLE
    if isinstance(seg, dict):
        rle = seg
        mask = mask_utils.decode(rle)

    # Если segmentation == polygon (список списков)
    elif isinstance(seg, list):
        mask = np.zeros((image_h, image_w), dtype=np.uint8)

        for poly in seg:
            pts = np.array(poly, dtype=np.float32).reshape(-1, 2)
            cv2.fillPoly(mask, [pts.astype(np.int32)], 1)

    else:
        raise ValueError("Unknown segmentation format")

    return mask


def visualize_coco_masks():
    for img_id, img_info in images.items():
        img_name = img_info["file_name"]
        img_path = os.path.join(IMAGES_DIR, img_name)

        img = cv2.imread(img_path)
        if img is None:
            print(f"❌ Не найдено изображение: {img_path}")
            continue

        h, w = img.shape[:2]

        # ищем все аннотации для этого изображения
        anns = [a for a in annotations if a["image_id"] == img_id]

        # создаём копию для отображения
        vis = img.copy()

        for ann in anns:
            mask = decode_mask(ann, h, w).astype(np.uint8)

            # случайный цвет
            color = [random.randint(0, 255) for _ in range(3)]

            # создаём цветную маску
            colored = np.zeros_like(vis)
            for c in range(3):
                colored[:, :, c] = mask * color[c]

            # смешиваем маску и изображение
            vis = cv2.addWeighted(vis, 1.0, colored, 0.5, 0)

            # рисуем bbox
            x, y, bw, bh = ann["bbox"]
            cv2.rectangle(vis, (int(x), int(y)), (int(x+bw), int(y+bh)), color, 2)

            # подпись класса
            class_name = categories[ann["category_id"]]
            cv2.putText(vis, class_name, (int(x), int(y)-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # показываем
        if SHOW:
            cv2.imshow("COCO Masks", vis)
            cv2.waitKey(0)

        # сохраняем
        if SAVE:
            save_path = os.path.join(SAVE_DIR, img_name)
            cv2.imwrite(save_path, vis)
            print(f"💾 Сохранено: {save_path}")


if __name__ == "__main__":
    visualize_coco_masks()
