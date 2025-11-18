import cv2
import numpy as np
import tensorflow as tf
import time # Добавим импорт time для совместимости


class YOLOLineDetector:
    """
    YOLOv8-seg детектор линии для замены OpenCV threshold.
    Совместим с интерфейсом LineDetector.
    """
    def __init__(self,
                 tflite_path="./checkpoints/last_model/tflite_export/best500ep_float16.tflite",
                 img_size=320,
                 conf_thresh=0.7,
                 iou_thresh=0.45,
                 min_contour_area=60):
        """
        Args:
            tflite_path: путь к TFLite модели YOLOv8-seg
            img_size: размер входа модели (обычно 320 или 640)
            conf_thresh: порог уверенности детекции
            iou_thresh: порог IoU для NMS
            min_contour_area: минимальная площадь контура (для совместимости)
        """
        self.img_size = img_size
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.min_contour_area = min_contour_area

        # Загрузка TFLite модели
        self.interpreter = tf.lite.Interpreter(model_path=tflite_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        print(f"✓ YOLOv8-seg загружена: {tflite_path}")
        print(f"  Input: {self.input_details[0]['shape']}")
        print(f"  Outputs: {len(self.output_details)}")

    def threshold(self, frame_bgr):
        """
        Основной метод - заменяет HSV threshold на YOLO инференс.
        Возвращает бинарную маску (как у LineDetector), представляющую
        НАИБОЛЕЕ ВЕРОЯТНУЮ ОДНУ ЛИНИЮ.

        Args:
            frame_bgr: входной кадр BGR

        Returns:
            mask: бинарная маска uint8 (255 - линия, 0 - фон)
        """
        h0, w0 = frame_bgr.shape[:2]

        # 1. Предобработка
        img_resized = cv2.resize(frame_bgr, (self.img_size, self.img_size))
        inp = np.expand_dims(img_resized / 255.0, 0).astype(np.float32)

        # 2. Инференс
        self.interpreter.set_tensor(self.input_details[0]['index'], inp)
        self.interpreter.invoke()

        # 3. Получение выходов
        det_out = self.interpreter.get_tensor(self.output_details[0]['index'])[0]  # (37, 2100)
        proto = self.interpreter.get_tensor(self.output_details[1]['index'])[0]    # (80, 80, 32)

        boxes = det_out[:4].T           # [2100, 4] (cx, cy, w, h)
        scores = det_out[4]             # [2100]
        mask_coef = det_out[5:].T       # [2100, 32]

        # 4. Фильтрация по confidence
        keep = scores > self.conf_thresh
        boxes = boxes[keep]
        scores = scores[keep]
        mask_coef = mask_coef[keep]

        if len(scores) == 0:
            # Линия не найдена
            return np.zeros((h0, w0), dtype=np.uint8)

        # 5. NMS (Non-Maximum Suppression)
        # NMSBoxes ожидает [x, y, w, h]
        boxes_xywh = []
        for cx, cy, w, h in boxes:
            x = int((cx - w / 2) * w0 / self.img_size)
            y = int((cy - h / 2) * h0 / self.img_size)
            width = int(w * w0 / self.img_size)
            height = int(h * h0 / self.img_size)
            boxes_xywh.append([x, y, width, height])

        indices = cv2.dnn.NMSBoxes(boxes_xywh, scores.tolist(),
                                    self.conf_thresh, self.iou_thresh)

        if len(indices) == 0:
            return np.zeros((h0, w0), dtype=np.uint8)

        indices = np.array(indices).flatten()
        boxes = boxes[indices]
        scores = scores[indices] # Не забудьте фильтровать scores тоже
        mask_coef = mask_coef[indices]

        # --- НОВОЕ: Выбор лучшей детекции ---
        # Вместо объединения всех масок, выбираем ту, у которой самый высокий score
        # Или можно выбрать самую большую по площади (после реконструкции масок)
        if len(scores) > 0:
            best_idx = np.argmax(scores)
            selected_box = boxes[best_idx]
            selected_mask_coef = mask_coef[best_idx]
        else:
            return np.zeros((h0, w0), dtype=np.uint8)


        # 6. Реконструкция маски для ОДНОЙ выбранной детекции
        proto_flat = proto.reshape(-1, proto.shape[-1])  # [6400, 32]
        # Используем только один выбранный набор коэффициентов
        masks_single = 1 / (1 + np.exp(-np.dot(proto_flat, selected_mask_coef.T)))  # sigmoid
        masks_single = masks_single.reshape(80, 80) # Теперь это одна маска 80x80

        # 7. Финальная маска
        final_mask = np.zeros((h0, w0), dtype=np.uint8)

        cx, cy, w_box, h_box = selected_box # Размеры из YOLO-пространства
        
        # Координаты бокса в исходном разрешении
        x1 = max(0, min(w0 - 1, int((cx - w_box / 2) * w0 / self.img_size)))
        y1 = max(0, min(h0 - 1, int((cy - h_box / 2) * h0 / self.img_size)))
        x2 = max(0, min(w0, int((cx + w_box / 2) * w0 / self.img_size)))
        y2 = max(0, min(h0, int((cy + h_box / 2) * h0 / self.img_size)))
        
        # Ресайз выбранной маски к исходному размеру
        mask_resized = cv2.resize(masks_single, (w0, h0))
        mask_bin = (mask_resized > 0.5).astype(np.uint8) * 255
        
        # Обрезка маски по боксу (чтобы избежать артефактов за пределами детекции)
        cropped_mask = np.zeros_like(mask_bin)
        cropped_mask[y1:y2, x1:x2] = mask_bin[y1:y2, x1:x2]
        
        final_mask = cropped_mask # Теперь это только одна маска

        # 8. Морфологическая обработка (как в оригинале)
        kernel = np.ones((3, 3), np.uint8)
        final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, kernel, iterations=1)

        return final_mask

    def split_upper_lower(self, mask):
        """Совместимость с LineDetector API"""
        h = mask.shape[0]
        return mask[:h // 2, :], mask[h // 2:, :]

    def largest_contour_center_x(self, mask, min_area=None):
        """
        Вычисляет центр линии (взвешенный по площади контуров).
        Совместимость с LineDetector API.
        Теперь, поскольку `threshold` должен возвращать ОДНУ линию,
        эта функция, вероятно, будет работать с одним большим контуром.
        """
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return None, 0.0

        min_area = min_area or self.min_contour_area
        
        # --- НОВОЕ: Выбор самого большого контура ---
        largest_c = None
        max_area = 0
        for c in cnts:
            area = cv2.contourArea(c)
            if area > max_area and area >= min_area:
                max_area = area
                largest_c = c
        
        if largest_c is None:
            return None, 0.0

        M = cv2.moments(largest_c)
        if M["m00"] == 0:
            return None, 0.0
        cx = M["m10"] / M["m00"]
        
        # Дисперсию для одного контура считать бессмысленно, но сохраним API
        # Можно считать дисперсию X-координат пикселей внутри контура
        # Но для этого API она не очень полезна, если у нас один контур.
        # Просто возвращаем 0.0
        return int(cx), 0.0

    def roi_bins(self, mask, bins=3):
        """Подсчет активных пикселей по регионам"""
        h, w = mask.shape
        step = w // bins
        counts = []
        for i in range(bins):
            x0 = i * step
            x1 = (i + 1) * step if i < bins - 1 else w
            sub = mask[:, x0:x1]
            counts.append(int(cv2.countNonZero(sub)))
        return counts

    def visualize(self, frame, mask, upper_x, lower_x):
        """Визуализация (копия из LineDetector)"""
        out = frame.copy()
        h, w = out.shape[:2]

        # Полупрозрачная маска
        color_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        out = cv2.addWeighted(out, 0.75, color_mask, 0.25, 0)

        # Линия раздела кадра
        cv2.line(out, (0, h // 2), (w, h // 2), (0, 255, 255), 1)

        if upper_x is not None:
            cv2.circle(out, (upper_x, h // 4), 4, (0, 255, 255), -1)
            cv2.putText(out, f"U:{upper_x}", (upper_x + 5, h // 4 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        if lower_x is not None:
            cv2.circle(out, (lower_x, (3 * h) // 4), 4, (0, 255, 0), -1)
            cv2.putText(out, f"L:{lower_x}", (lower_x + 5, (3 * h) // 4 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
        return out