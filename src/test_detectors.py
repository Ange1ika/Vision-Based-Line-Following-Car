#!/usr/bin/env python3
import cv2
import time
import argparse
from pathlib import Path
from YoloLineDetector import YOLOLineDetector
from line_detector import LineDetector

OUTPUT_DIR = Path("output_vis")
OUTPUT_DIR.mkdir(exist_ok=True)


def save_result(img_name, vis, name="yolo"):
    """Сохраняет визуализацию в output_vis/"""
    out_path = OUTPUT_DIR / f"{img_name}_{name}.jpg"
    cv2.imwrite(str(out_path), vis)
    print(f"💾 Сохранено: {out_path}")


def test_single_image(img_path, detector, name="detector", show_steps=False):
    """Тестирование на одном изображении. Возвращает (визуализация, fps, ms)"""
    print(f"\n{'='*60}")
    print(f" Тестирование: {img_path} [{name}]")
    print(f"{'='*60}")
    
    frame = cv2.imread(img_path)
    if frame is None:
        print(f" Не удалось загрузить изображение: {img_path}")
        return None, 0, 0
    
    h, w = frame.shape[:2]
    print(f"Размер изображения: {w}x{h}")
    
    # Засекаем время
    start = time.time()
    mask = detector.threshold(frame)

    # Разделение на регионы
    upper_mask, lower_mask = detector.split_upper_lower(mask)
    upper_x, _ = detector.largest_contour_center_x(upper_mask)
    lower_x, _ = detector.largest_contour_center_x(lower_mask)
    print(f"📍 Upper_x={upper_x}, Lower_x={lower_x}")

    # ROI bins
    upper_counts = detector.roi_bins(upper_mask, bins=3)
    lower_counts = detector.roi_bins(lower_mask, bins=3)
    print(f"Upper ROI: L={upper_counts[0]} C={upper_counts[1]} R={upper_counts[2]}")
    print(f"Lower ROI: L={lower_counts[0]} C={lower_counts[1]} R={lower_counts[2]}")

    inference_time = (time.time() - start) * 1000
    fps = 1000.0 / inference_time if inference_time > 0 else 0.0
    
    print(f" Время инференса: {inference_time:.2f} ms ({fps:.1f} FPS)")
    print(f" Пикселей линии: {cv2.countNonZero(mask)}")

    vis = detector.visualize(frame, mask, upper_x, lower_x)

    # Добавляем текст
    label_color = (0, 255, 255)
    cv2.putText(vis, f"{name}: {inference_time:.1f} ms", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, label_color, 2)
    cv2.putText(vis, f"{fps:.1f} FPS", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, label_color, 2)

    return vis, fps, inference_time


def main():
    BASE_DIR = Path(__file__).resolve().parents[1]
    
    parser = argparse.ArgumentParser(description="Тестирование YOLO детектора")
    parser.add_argument('--image', type=str, default=BASE_DIR / "data/test_images",
                        help='Путь к изображению или папке')
    parser.add_argument('--model', type=str, default="/home/angelika/Desktop/Seoul/Vision-Based-Line-Following-Car/checkpoints/yolov8n_seg_last/tflite_export/best_float32.tflite",
                        help='Путь к TFLite модели YOLO')
    parser.add_argument('--size', type=int, default=320)
    parser.add_argument('--conf', type=float, default=0.7)
    
    args = parser.parse_args()

    # Загрузка детектора
    detector = YOLOLineDetector(
        tflite_path=args.model,
        img_size=args.size,
        conf_thresh=args.conf
    )

    img_path = Path(args.image)
    if img_path.is_dir():
        images = sorted(list(img_path.glob("*.jpg")) + list(img_path.glob("*.png")))
        print(f" Найдено изображений: {len(images)}")
    else:
        images = [img_path]

    for img in images:
        print(f"\n Обработка {img.name} ...")
        vis, fps, t = test_single_image(str(img), detector, "YOLOv8-TFLite")
        if vis is not None:
            save_result(img.stem, vis, "yolo")


if __name__ == "__main__":
    main()
