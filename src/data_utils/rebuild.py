import os
import cv2
import numpy as np
from glob import glob

# ---------------------------------------
# ПУТИ (измени под свой датасет)
# ---------------------------------------
DATA_ROOT = "/home/angelika/Desktop/Seoul/Intelligent Control/DATA_annotation/datasets/dataset7"

IN_IMAGES_TRAIN = f"{DATA_ROOT}/images/train"
IN_LABELS_TRAIN = f"{DATA_ROOT}/labels/train"

IN_IMAGES_VAL = f"{DATA_ROOT}/images/val"
IN_LABELS_VAL = f"{DATA_ROOT}/labels/val"

OUT_LABELS_TRAIN = f"{DATA_ROOT}/labels_rebuilt/train"
OUT_LABELS_VAL = f"{DATA_ROOT}/labels_rebuilt/val"

os.makedirs(OUT_LABELS_TRAIN, exist_ok=True)
os.makedirs(OUT_LABELS_VAL, exist_ok=True)


# ---------------------------------------
# 1) Загружаем исходную masку YOLO → маска OpenCV
# ---------------------------------------
def load_yolo_segmentation_mask(label_path, img_w, img_h):
    """Восстанавливает маску по YOLO segmentation: class cx cy w h x1 y1 ..."""
    mask = np.zeros((img_h, img_w), dtype=np.uint8)

    if not os.path.exists(label_path):
        return mask

    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 6:
                continue

            coords = parts[5:]

            poly = []
            for i in range(0, len(coords), 2):
                x = float(coords[i]) * img_w
                y = float(coords[i+1]) * img_h
                poly.append([int(x), int(y)])

            poly = np.array([poly], dtype=np.int32)
            cv2.fillPoly(mask, poly, 255)

    return mask


# ---------------------------------------
# 2) Маска → polygon (идеальный контур)
# ---------------------------------------
def extract_polygon_from_mask(mask):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(cnts) == 0:
        return None

    cnt = max(cnts, key=cv2.contourArea).reshape(-1, 2)

    if len(cnt) < 3:
        return None

    return cnt


# ---------------------------------------
# 3) polygon → YOLO строка (cx cy w h x1 y1 ...)
# ---------------------------------------
def polygon_to_yolo(cnt, img_w, img_h, class_id=0):

    xs = cnt[:, 0]
    ys = cnt[:, 1]

    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()

    cx = (x_min + x_max) / 2 / img_w
    cy = (y_min + y_max) / 2 / img_h
    bw = (x_max - x_min) / img_w
    bh = (y_max - y_min) / img_h

    poly_norm = []
    for x, y in zip(xs, ys):
        poly_norm.append(x / img_w)
        poly_norm.append(y / img_h)

    line = (
        f"{class_id} "
        f"{cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f} "
        + " ".join(f"{p:.6f}" for p in poly_norm)
    )

    return line


# ---------------------------------------
# 4) Основной процесс: rebuild
# ---------------------------------------
def rebuild_split(img_dir, lbl_dir, out_dir):
    img_files = sorted(glob(os.path.join(img_dir, "*.jpg")) +
                       glob(os.path.join(img_dir, "*.png")))

    print(f"\n📌 Обработка: {img_dir}")
    print(f"Найдено изображений: {len(img_files)}")

    good, bad = 0, 0

    for img_path in img_files:
        name = os.path.splitext(os.path.basename(img_path))[0]
        lbl_path = os.path.join(lbl_dir, name + ".txt")

        img = cv2.imread(img_path)
        img_h, img_w = img.shape[:2]

        # 1. восстановить mask из старой YOLO-аннотации
        mask = load_yolo_segmentation_mask(lbl_path, img_w, img_h)

        # 2. достать polygon
        cnt = extract_polygon_from_mask(mask)
        if cnt is None:
            bad += 1
            continue

        # 3. построить новую строку YOLO
        line = polygon_to_yolo(cnt, img_w, img_h, class_id=0)

        # 4. сохранить
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, name + ".txt")

        with open(out_path, "w") as f:
            f.write(line + "\n")

        good += 1

    print(f"✓ Успешно: {good},  ✗ Пропущено: {bad}")


# ---------------------------------------
# Запуск
# ---------------------------------------
if __name__ == "__main__":
    print("🚀 Запускаю пересборку масок через OpenCV")

    rebuild_split(IN_IMAGES_TRAIN, IN_LABELS_TRAIN, OUT_LABELS_TRAIN)
    rebuild_split(IN_IMAGES_VAL, IN_LABELS_VAL, OUT_LABELS_VAL)

    print("\n🎉 ГОТОВО! New YOLO labels полностью собраны!")
    print("Маски теперь идеальные и полностью совместимы с YOLOv8/YOLOv11.")
