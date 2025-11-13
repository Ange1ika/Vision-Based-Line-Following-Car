"""
Тестирование YOLOv8 Segmentation модели (PyTorch и ONNX)
"""
import numpy as np
import cv2
from pathlib import Path
import matplotlib.pyplot as plt

def test_with_ultralytics(model_path, image_path, output_dir):
    """
    Тестирование с использованием Ultralytics библиотеки
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        print("Ultralytics не установлен. Установите: pip install ultralytics")
        return None
    
    print("=" * 60)
    print("ТЕСТИРОВАНИЕ С ULTRALYTICS (PyTorch)")
    print("=" * 60)
    
    # Загрузка модели
    print(f"\nЗагрузка модели: {model_path}")
    model = YOLO(model_path)
    
    print(f"Тип задачи: {model.task}")
    print(f"Имена классов: {model.names}")
    
    # Инференс
    print(f"\nОбработка изображения: {image_path}")
    results = model(image_path, save=False, verbose=False)
    
    result = results[0]
    
    print(f"\nРезультаты:")
    print(f"  Форма изображения: {result.orig_shape}")
    
    # Детекции
    if result.boxes is not None and len(result.boxes) > 0:
        print(f"  Найдено объектов: {len(result.boxes)}")
        for i, box in enumerate(result.boxes):
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            print(f"    [{i}] Класс: {model.names[cls]}, Уверенность: {conf:.3f}")
    else:
        print(f"  Найдено объектов: 0")
    
    # Маски сегментации
    if result.masks is not None:
        print(f"  Маски: {result.masks.data.shape}")
        masks = result.masks.data.cpu().numpy()
    else:
        print(f"  Маски: отсутствуют")
        masks = None
    
    # Визуализация
    output_path = output_dir / "ultralytics_pytorch_result.jpg"
    annotated = result.plot()
    cv2.imwrite(str(output_path), annotated)
    print(f"\nРезультат сохранён: {output_path}")
    
    return result, masks

def test_onnx_with_opencv(onnx_path, image_path, output_dir, conf_threshold=0.25):
    """
    Тестирование ONNX модели с OpenCV DNN
    """
    print("\n" + "=" * 60)
    print("ТЕСТИРОВАНИЕ ONNX С OPENCV DNN")
    print("=" * 60)
    
    # Загрузка модели
    print(f"\nЗагрузка ONNX: {onnx_path}")
    net = cv2.dnn.readNetFromONNX(str(onnx_path))
    
    # Загрузка изображения
    img = cv2.imread(str(image_path))
    original_shape = img.shape[:2]
    print(f"Размер изображения: {original_shape}")
    
    # Предобработка
    input_size = (320, 320)
    blob = cv2.dnn.blobFromImage(
        img,
        scalefactor=1/255.0,
        size=input_size,
        mean=(0, 0, 0),
        swapRB=True,
        crop=False
    )
    
    print(f"Форма blob: {blob.shape}")
    
    # Инференс
    net.setInput(blob)
    
    # Получаем имена выходных слоёв
    output_layers = net.getUnconnectedOutLayersNames()
    print(f"Выходные слои: {output_layers}")
    
    outputs = net.forward(output_layers)
    
    print(f"\nКоличество выходов: {len(outputs)}")
    for i, output in enumerate(outputs):
        print(f"  Выход [{i}]: shape={output.shape}, dtype={output.dtype}")
    
    return outputs, img

def test_onnx_with_onnxruntime(onnx_path, image_path, output_dir):
    """
    Тестирование ONNX модели с ONNX Runtime
    """
    import onnxruntime as ort
    
    print("\n" + "=" * 60)
    print("ТЕСТИРОВАНИЕ ONNX С ONNX RUNTIME")
    print("=" * 60)
    
    # Загрузка модели
    print(f"\nЗагрузка ONNX: {onnx_path}")
    session = ort.InferenceSession(str(onnx_path), providers=['CPUExecutionProvider'])
    
    # Информация о модели
    print("\nИнформация о модели:")
    for inp in session.get_inputs():
        print(f"  Вход: {inp.name}, shape: {inp.shape}, type: {inp.type}")
    
    for out in session.get_outputs():
        print(f"  Выход: {out.name}, shape: {out.shape}, type: {out.type}")
    
    # Загрузка и предобработка изображения
    img = cv2.imread(str(image_path))
    original_shape = img.shape[:2]
    
    input_size = (320, 320)
    img_resized = cv2.resize(img, input_size)
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_normalized = img_rgb.astype(np.float32) / 255.0
    img_input = np.transpose(img_normalized, (2, 0, 1))
    img_input = np.expand_dims(img_input, axis=0)
    
    print(f"\nФорма входа: {img_input.shape}")
    
    # Инференс
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: img_input})
    
    print(f"\nВыходы:")
    for i, output in enumerate(outputs):
        print(f"  [{i}] shape: {output.shape}, dtype: {output.dtype}")
        print(f"       range: [{output.min():.3f}, {output.max():.3f}]")
    
    # Визуализация
    visualize_yolo_segmentation_output(
        outputs, img, img_resized, output_dir / "onnxruntime_result.jpg"
    )
    
    return outputs, img

def visualize_yolo_segmentation_output(outputs, original_img, resized_img, output_path):
    """
    Визуализация выхода YOLO сегментации
    """
    print(f"\nВизуализация результатов...")
    
    # YOLOv8-seg обычно возвращает 2 выхода:
    # output0: [1, 116, 8400] - детекции (x, y, w, h, conf, classes..., mask_coeffs)
    # output1: [1, 32, 160, 160] - прототипы масок
    
    if len(outputs) >= 2:
        detections = outputs[0][0]  # [116, 8400]
        protos = outputs[1][0]      # [32, 160, 160]
        
        print(f"Детекции: {detections.shape}")
        print(f"Прототипы: {protos.shape}")
        
        # Транспонируем для удобства
        detections = detections.T  # [8400, 116]
        
        # Первые 4 - bbox, потом objectness/class scores, последние 32 - mask coefficients
        num_classes = detections.shape[1] - 4 - 32  # 116 - 4 - 32 = 80 классов
        
        print(f"Количество классов: {num_classes}")
        
        # Простая визуализация - покажем, что в выходе есть данные
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Исходное изображение
        axes[0, 0].imshow(cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title('Исходное изображение')
        axes[0, 0].axis('off')
        
        # Тепловая карта confidence
        conf_scores = detections[:, 4:4+num_classes].max(axis=1)
        conf_map = conf_scores.reshape(80, 105)  # Примерная форма сетки
        axes[0, 1].imshow(conf_map, cmap='hot')
        axes[0, 1].set_title('Confidence Map')
        axes[0, 1].axis('off')
        
        # Первый прототип маски
        axes[1, 0].imshow(protos[0], cmap='viridis')
        axes[1, 0].set_title('Mask Prototype 0')
        axes[1, 0].axis('off')
        
        # Статистика
        stats_text = f"Детекций: {detections.shape[0]}\n"
        stats_text += f"Классов: {num_classes}\n"
        stats_text += f"Прототипов масок: {protos.shape[0]}\n"
        stats_text += f"Max confidence: {conf_scores.max():.3f}\n"
        stats_text += f"Детекций > 0.25: {(conf_scores > 0.25).sum()}"
        
        axes[1, 1].text(0.1, 0.5, stats_text, fontsize=12, verticalalignment='center')
        axes[1, 1].set_title('Статистика')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Визуализация сохранена: {output_path}")
        plt.close()
    else:
        print(f"⚠️  Неожиданное количество выходов: {len(outputs)}")

def main():
    # Пути
    BASE_DIR = Path(__file__).resolve().parents[2]
    
    pt_model_path = BASE_DIR / "checkpoints" / "last_model" / "best.pt"
    onnx_model_path = BASE_DIR / "checkpoints" / "last_model" / "best.onnx"
    image_path = BASE_DIR / "data" / "test_images" / "45.jpg"
    
    output_dir = BASE_DIR / "data" / "comparison_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Тест с Ultralytics (PyTorch)
    try:
        pt_result, pt_masks = test_with_ultralytics(pt_model_path, image_path, output_dir)
    except Exception as e:
        print(f"\n✗ Ошибка PyTorch тестирования: {e}")
        import traceback
        traceback.print_exc()
    
    # 2. Тест ONNX с ONNX Runtime
    try:
        onnx_outputs, img = test_onnx_with_onnxruntime(onnx_model_path, image_path, output_dir)
    except Exception as e:
        print(f"\n✗ Ошибка ONNX Runtime: {e}")
        import traceback
        traceback.print_exc()
    
    # 3. Опционально: тест с OpenCV DNN
    try:
        cv_outputs, img = test_onnx_with_opencv(onnx_model_path, image_path, output_dir)
    except Exception as e:
        print(f"\n⚠️  OpenCV DNN недоступен или не поддерживает эту модель: {e}")
    
    print("\n" + "=" * 60)
    print("ТЕСТИРОВАНИЕ ЗАВЕРШЕНО")
    print(f"Результаты в: {output_dir}")
    print("=" * 60)

if __name__ == "__main__":
    main()
