#!/usr/bin/env python3
import cv2
import time
import argparse
from pathlib import Path
from YoloLinedetector import YOLOLineDetector
from line_detector import LineDetector

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
    upper_x, upper_disp = detector.largest_contour_center_x(upper_mask)
    lower_x, lower_disp = detector.largest_contour_center_x(lower_mask)
    print(f"📍 Upper_x={upper_x}, Lower_x={lower_x}")

    # ROI bins
    upper_counts = detector.roi_bins(upper_mask, bins=3)
    lower_counts = detector.roi_bins(lower_mask, bins=3)
    print(f"Upper ROI: L={upper_counts[0]} C={upper_counts[1]} R={upper_counts[2]}")
    print(f"Lower ROI: L={lower_counts[0]} C={lower_counts[1]} R={lower_counts[2]}")
    inference_time = (time.time() - start)*1000
    fps = 1000.0 / inference_time if inference_time > 0 else 0.0
    
    print(f" Время инференса: {inference_time:.2f} ms ({fps:.1f} FPS)")
    print(f" Пикселей линии: {cv2.countNonZero(mask)}")

    vis = detector.visualize(frame, mask, upper_x, lower_x)
    label_color = (0, 255, 255) if "OpenCV" in name else (255, 150, 0)
    # Корректное добавление нескольких строк текста
    text1 = f"{name}: {inference_time:.1f} ms"
    text2 = f"{fps:.1f} FPS"
    cv2.putText(vis, text1, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, label_color, 2)
    cv2.putText(vis, text2, (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, label_color, 2)

    if show_steps:
        cv2.imshow(f"{name} - Mask", mask)
        cv2.imshow(f"{name} - Result", vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return vis, fps, inference_time


def benchmark_detectors(img_path, opencv_detector, yolo_detector, show_steps=False):
    """Бенчмарк: тест каждого детектора отдельно и склейка визуализаций"""
    print(f"\n{'='*60}")
    print(f"⚡ БЕНЧМАРК OpenCV vs YOLO")
    print(f"{'='*60}")

    vis_cv, fps_cv, t_cv = test_single_image(img_path, opencv_detector, "OpenCV HSV", show_steps)
    vis_yolo, fps_yolo, t_yolo = test_single_image(img_path, yolo_detector, "YOLOv8-seg", show_steps)
    
    if vis_cv is None or vis_yolo is None:
        print(" Ошибка при визуализации. Пропуск.")
        return

    comparison = cv2.hconcat([vis_cv, vis_yolo])

    print(f"\n⚖️ Сравнение:")
    print(f"   OpenCV: {t_cv:.1f} ms ({fps_cv:.1f} FPS)")
    print(f"   YOLOv8: {t_yolo:.1f} ms ({fps_yolo:.1f} FPS)")
    print(f"   Относительная скорость: OpenCV быстрее в {t_yolo / t_cv:.2f}x\n")

    cv2.imshow("Comparison", comparison)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Тестирование YOLO и OpenCV детекторов")
    parser.add_argument('--image', type=str, default="./test_img",
                        help='Путь к изображению или папке')
    parser.add_argument('--model', type=str, default='./checkpoints/best_fixed_float16.tflite',
                        help='Путь к TFLite модели YOLO')
    parser.add_argument('--size', type=int, default=320, help='Размер входа YOLO')
    parser.add_argument('--conf', type=float, default=0.7, help='Порог уверенности YOLO')
    parser.add_argument('--benchmark', action='store_true',
                        help='Сравнить производительность OpenCV vs YOLO')
    parser.add_argument('--steps', action='store_true',
                        help='Показать промежуточные шаги')
    args = parser.parse_args()

    detectors = {}
    detectors['yolo'] = YOLOLineDetector(tflite_path=args.model, img_size=args.size, conf_thresh=args.conf)
    detectors['opencv'] = LineDetector()

    img_path = Path(args.image)
    if img_path.is_dir():
        images = sorted(list(img_path.glob("*.jpg")) + list(img_path.glob("*.png")))
        print(f" Найдено изображений: {len(images)}")
    else:
        images = [img_path]

    for img in images:
        print(f"\n Обработка {img.name} ...")
        if args.benchmark:
            benchmark_detectors(str(img), detectors['opencv'], detectors['yolo'], show_steps=args.steps)
        else:
            vis, fps, t = test_single_image(str(img), detectors['yolo'], "YOLOv8-seg", show_steps=args.steps)
            if vis is not None:
                cv2.imshow(f"{img.name} - YOLOv8", vis)
                cv2.waitKey(0)
                cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
