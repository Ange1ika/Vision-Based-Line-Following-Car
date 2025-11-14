import os
from glob import glob

# >>> ПУТИ (укажи train или val)
IMAGES_DIR = "/home/angelika/Desktop/Seoul/Intelligent Control/DATA_annotation/datasets/main/images"
LABELS_DIR = "/home/angelika/Desktop/Seoul/Intelligent Control/DATA_annotation/datasets/main/labels"

os.makedirs(LABELS_DIR, exist_ok=True)

# Все изображения
img_files = sorted(
    glob(os.path.join(IMAGES_DIR, "*.jpg")) +
    glob(os.path.join(IMAGES_DIR, "*.png")) +
    glob(os.path.join(IMAGES_DIR, "*.jpeg"))
)

print(f"Найдено изображений: {len(img_files)}")

added = 0

for img_path in img_files:
    name = os.path.splitext(os.path.basename(img_path))[0]
    txt_path = os.path.join(LABELS_DIR, name + ".txt")

    # если файла нет – создаём пустой
    if not os.path.exists(txt_path):
        with open(txt_path, "w") as f:
            pass  # создаёт пустой файл
        added += 1
        print(f"➕ Создан пустой label: {txt_path}")

print(f"\nГотово! Добавлено пустых файлов: {added}")
