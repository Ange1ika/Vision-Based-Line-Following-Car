import os
import random
import shutil
from glob import glob

# ==============================
# НАСТРОЙКИ
# ==============================
DATASET_DIR = "/home/angelika/Desktop/Seoul/Intelligent Control/DATA_annotation/datasets/main"      # путь к dataset4
TRAIN_RATIO = 0.8      # 80% train, 20% val
# ==============================

IMG_DIR = os.path.join(DATASET_DIR, "images")
LBL_DIR = os.path.join(DATASET_DIR, "labels")

# создаём выходные папки
for sub in ["images/train", "images/val", "labels/train", "labels/val"]:
    os.makedirs(os.path.join(DATASET_DIR, sub), exist_ok=True)

# список изображений
images = sorted(
    glob(os.path.join(IMG_DIR, "*.jpg")) +
    glob(os.path.join(IMG_DIR, "*.png")) +
    glob(os.path.join(IMG_DIR, "*.jpeg"))
)

print(f"Найдено изображений: {len(images)}")

random.shuffle(images)
split_idx = int(len(images) * TRAIN_RATIO)

train_imgs = images[:split_idx]
val_imgs = images[split_idx:]

print(f"TRAIN: {len(train_imgs)}")
print(f"VAL:   {len(val_imgs)}")

def move_pair(img_list, split):
    """Перенос изображения и соответствующего txt."""
    out_img_dir = os.path.join(DATASET_DIR, "images", split)
    out_lbl_dir = os.path.join(DATASET_DIR, "labels", split)

    for img_path in img_list:
        fname = os.path.basename(img_path)
        name = os.path.splitext(fname)[0]

        txt_src = os.path.join(LBL_DIR, name + ".txt")
        img_dst = os.path.join(out_img_dir, fname)
        txt_dst = os.path.join(out_lbl_dir, name + ".txt")

        shutil.move(img_path, img_dst)

        if os.path.exists(txt_src):
            shutil.move(txt_src, txt_dst)
        else:
            # создаём пустую аннотацию (YOLO требует txt)
            with open(txt_dst, "w") as f:
                pass

        print(f"✔ {fname} → {split}")

# переносим train
move_pair(train_imgs, "train")
# переносим val
move_pair(val_imgs, "val")

print("\n🎉 Датасет успешно разделён!")
