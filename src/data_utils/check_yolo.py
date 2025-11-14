import os
import cv2
from ultralytics import YOLO

# ========= НАСТРОЙКИ =========
MODEL_PATH = "/home/angelika/Desktop/Seoul/Vision-Based-Line-Following-Car/checkpoints/yolov8n_seg_last/best.pt"                    # путь к модели YOLOv8-seg
IMAGES_DIR = "data/test_images"                     # папка с изображениями
OUTPUT_DIR = "data/test_images/output"                     # папка для результатов
CONF = 0.7                                # порог confidence
SHOW = False                              # показывать окна с результатами
# =====================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Загружаем модель
model = YOLO(MODEL_PATH)

# Получаем список файлов
images = [f for f in os.listdir(IMAGES_DIR)
          if f.lower().endswith((".jpg", ".jpeg", ".png"))]

print(f"Найдено изображений: {len(images)}\n")

# Обработка папки
for img_name in images:
    img_path = os.path.join(IMAGES_DIR, img_name)
    print(f"Обрабатываю: {img_name}")

    # Прогон через YOLO
    results = model(img_path, conf=CONF, verbose=False)[0]

    # Визуализация (модель сама рисует маски и боксы)
    plotted = results.plot()

    # Сохраняем
    save_path = os.path.join(OUTPUT_DIR, img_name)
    cv2.imwrite(save_path, plotted)

    if SHOW:
        cv2.imshow("YOLOv8 Seg", plotted)
        cv2.waitKey(0)

print("\nГотово! Все результаты лежат в:", OUTPUT_DIR)
