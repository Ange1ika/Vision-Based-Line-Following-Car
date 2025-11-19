import cv2
import numpy as np
from picamera2 import Picamera2
import RPi.GPIO as GPIO
import time
import threading
import os

# Попытка импорта YOLO (опционально)
try:
    from YoloLineDetector import YOLOLineDetector
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("[WARN] YOLOLineDetector not found, using threshold")

# ------------------- ПИНЫ МОТОРА ------------------- #
PWMA = 18
AIN1 = 22
AIN2 = 27
PWMB = 23
BIN1 = 25
BIN2 = 24

GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)
GPIO.setup(PWMA, GPIO.OUT)
GPIO.setup(AIN1, GPIO.OUT)
GPIO.setup(AIN2, GPIO.OUT)
GPIO.setup(PWMB, GPIO.OUT)
GPIO.setup(BIN1, GPIO.OUT)
GPIO.setup(BIN2, GPIO.OUT)

L_Motor = GPIO.PWM(PWMA, 500)
L_Motor.start(0)
R_Motor = GPIO.PWM(PWMB, 500)
R_Motor.start(0)


def motor_stop():
    GPIO.output(AIN1, 0)
    GPIO.output(AIN2, 1)
    L_Motor.ChangeDutyCycle(0)

    GPIO.output(BIN1, 0)
    GPIO.output(BIN2, 1)
    R_Motor.ChangeDutyCycle(0)


def set_wheel_speeds(left_speed, right_speed):
    """
    Универсальная функция:
    - left_speed / right_speed в диапазоне [-100, 100]
    - знак задаёт направление
    """
    left_speed = max(-100, min(100, float(left_speed)))
    right_speed = max(-100, min(100, float(right_speed)))

    # ЛЕВЫЙ МОТОР
    if left_speed >= 0:
        GPIO.output(AIN1, 0)
        GPIO.output(AIN2, 1)
        L_Motor.ChangeDutyCycle(left_speed)
    else:
        GPIO.output(AIN1, 1)
        GPIO.output(AIN2, 0)
        L_Motor.ChangeDutyCycle(-left_speed)

    # ПРАВЫЙ МОТОР
    if right_speed >= 0:
        GPIO.output(BIN1, 0)
        GPIO.output(BIN2, 1)
        R_Motor.ChangeDutyCycle(right_speed)
    else:
        GPIO.output(BIN1, 1)
        GPIO.output(BIN2, 0)
        R_Motor.ChangeDutyCycle(-right_speed)


def rotate_in_place(direction, speed):
    """
    direction: -1 (влево), +1 (вправо)
    speed: 0..100
    """
    speed = max(0, min(100, speed))
    if direction < 0:
        set_wheel_speeds(-speed, speed)
    else:
        set_wheel_speeds(speed, -speed)


class CameraThread(threading.Thread):
    def __init__(self, cam, flip_image=False):
        super().__init__(daemon=True)
        self.cam = cam
        self.flip_image = flip_image
        self.frame = None
        self.lock = threading.Lock()
        self.running = True

    def run(self):
        while self.running:
            frame = self.cam.capture_array()
            if self.flip_image:
                frame = cv2.flip(frame, -1)
            with self.lock:
                self.frame = frame

    def get_frame(self):
        with self.lock:
            if self.frame is None:
                return None
            return self.frame.copy()

    def stop(self):
        self.running = False


def threshold_detect(roi):
    """Простой threshold для сравнения"""
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(
        blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    bin_mask = cv2.erode(thresh, None, iterations=1)
    bin_mask = cv2.dilate(bin_mask, None, iterations=1)
    return bin_mask


def main():
    # --------- ПАРАМЕТРЫ --------- #
    width, height = 320, 240

    base_speed = 55
    flip_image = True  # Камера перевёрнута
    invert_x_coords = True  # Инвертируем X-координаты из-за flip
    deadband_error = 0.08

    min_points_fitline = 80
    no_line_max = 3
    search_turn_speed = 40

    # PID коэффициенты (чуть мягче)
    Kp = 0.75
    Ki = 0.0
    Kd = 0.10

    integral = 0.0
    integral_limit = 5.0
    prev_error = 0.0
    prev_time = time.time()

    last_error_sign = 0.0
    no_line_frames = 0

    prev_frame_time = time.time()
    fps = 0.0

    error_filtered = 0.0
    error_smooth_alpha = 0.4

    straight_boost = 0.0
    straight_boost_step = 2.0
    straight_boost_max = 35.0

    # ========== ДЕТЕКТОР: YOLO или Threshold ========== #
    use_yolo = YOLO_AVAILABLE  # автоматически, если доступна
    yolo_detector = None
    
    if use_yolo:
        try:
            yolo_detector = YOLOLineDetector(
                tflite_path="./checkpoints/last_model/tflite_export/best500ep_float16.tflite",
                img_size=320,
                conf_thresh=0.65,
                iou_thresh=0.45,
                min_contour_area=50
            )
            print("[INFO] Using YOLO segmentation")
        except Exception as e:
            print(f"[WARN] Failed to load YOLO: {e}, using threshold")
            use_yolo = False

    # --------- ВИДЕО + ЛОГИ --------- #
    save_dir = "/home/raspberry/Desktop/data_mining/Vision-Based-Line-Following-Car/src/video"
    os.makedirs(save_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    detector_name = "yolo" if use_yolo else "thresh"
    video_path = os.path.join(save_dir, f"linetracer_{detector_name}_{timestamp}.avi")
    label_path = os.path.join(save_dir, f"linetracer_{detector_name}_{timestamp}_labels.txt")
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    video_writer = cv2.VideoWriter(video_path, fourcc, 30.0, (width, height))

    if not video_writer.isOpened():
        print("[ERROR] Failed to open VideoWriter")
        return

    label_file = open(label_path, "w")
    label_file.write("frame,detector,direction,error_raw,error_filtered,angle_deg,v,straight_boost,left_cmd,right_cmd,fps,x_bottom,center,line_pixels,error_direction\n")

    # --------- КАМЕРА --------- #
    cam = Picamera2()
    cam.configure(
        cam.create_video_configuration(
            main={"format": "BGR888", "size": (width, height)}
        )
    )
    cam.start()
    time.sleep(0.2)

    cam_thread = CameraThread(cam, flip_image=flip_image)
    cam_thread.start()

    frame_count = 0

    try:
        while True:
            frame = cam_thread.get_frame()
            if frame is None:
                time.sleep(0.001)
                continue

            now_f = time.time()
            dt_f = now_f - prev_frame_time
            if dt_f > 0:
                instant_fps = 1.0 / dt_f
                fps = 0.9 * fps + 0.1 * instant_fps
            prev_frame_time = now_f

            h, w = frame.shape[:2]

            # ========== ROI: нижняя треть ========== #
            roi_y_start = h * 2 // 3
            roi = frame[roi_y_start:h, :]
            roi_h, roi_w = roi.shape[:2]

            # ========== ДЕТЕКЦИЯ ЛИНИИ ========== #
            if use_yolo and yolo_detector is not None:
                # YOLO работает на всём кадре, но мы используем только ROI
                bin_mask = yolo_detector.threshold(roi)
            else:
                bin_mask = threshold_detect(roi)

            line_pixels = cv2.countNonZero(bin_mask)

            direction_label = "stop"
            angle_deg = 0.0
            has_line = False
            error_raw = 0.0
            left_cmd = 0.0
            right_cmd = 0.0
            v = 0.0
            x_bottom = 0.0
            center = roi_w / 2.0

            # ========== ВИЗУАЛИЗАЦИЯ ========== #
            vis_frame = frame.copy()
            
            # ROI граница
            cv2.rectangle(vis_frame, (0, roi_y_start), (w-1, h-1), (255, 255, 0), 2)
            
            # Центральная линия кадра
            frame_center_x = w // 2
            cv2.line(vis_frame, (frame_center_x, 0), (frame_center_x, h-1), (0, 255, 255), 2)
            
            # Центральная линия ROI
            roi_center_x = roi_w // 2
            cv2.line(vis_frame, (roi_center_x, roi_y_start), (roi_center_x, h-1), (0, 255, 0), 1)

            # Накладываем маску (зелёным цветом)
            mask_colored = np.zeros((roi_h, roi_w, 3), dtype=np.uint8)
            mask_colored[:, :, 1] = bin_mask  # зелёный канал
            vis_frame[roi_y_start:h, :] = cv2.addWeighted(
                vis_frame[roi_y_start:h, :], 0.6, mask_colored, 0.4, 0
            )

            # ========== ОБРАБОТКА ЛИНИИ ========== #
            if line_pixels >= 50:  # минимальный порог пикселей
                ys, xs = np.where(bin_mask > 0)
                
                # === ФИЛЬТРАЦИЯ: берём только нижние 60% маски для fitLine === #
                filter_y = int(roi_h * 0.4)
                idx = ys >= filter_y
                xs_filtered = xs[idx]
                ys_filtered = ys[idx]
                
                # Рисуем ВСЕ точки маски жёлтым (каждую 5-ю)
                for i in range(0, len(xs), 5):
                    pt_x = int(xs[i])
                    pt_y = int(ys[i]) + roi_y_start
                    cv2.circle(vis_frame, (pt_x, pt_y), 1, (0, 255, 255), -1)
                
                # Рисуем ОТФИЛЬТРОВАННЫЕ точки для fitLine зелёным
                for i in range(0, len(xs_filtered), 5):
                    pt_x = int(xs_filtered[i])
                    pt_y = int(ys_filtered[i]) + roi_y_start
                    cv2.circle(vis_frame, (pt_x, pt_y), 2, (0, 255, 0), -1)

                if len(xs_filtered) >= min_points_fitline:
                    has_line = True
                    no_line_frames = 0

                    pts = np.column_stack((xs_filtered, ys_filtered))
                    vx, vy, x0, y0 = cv2.fitLine(
                        pts, cv2.DIST_L2, 0, 0.01, 0.01
                    )

                    angle_deg = np.degrees(np.arctan2(float(vy), float(vx)))

                    # Пересечение с нижней линией ROI
                    if abs(vy) < 1e-6:
                        x_bottom = float(x0)
                    else:
                        bottom_y = roi_h - 1
                        t = (bottom_y - y0) / vy
                        t = np.clip(t, -1e6, 1e6)
                        x_bottom = float(x0 + vx * t)

                    # === ИНВЕРСИЯ X из-за flip_image === #
                    # При flip изображение отражается: правое становится левым
                    if invert_x_coords:
                        x_bottom = roi_w - x_bottom

                    # Визуализация fitLine
                    line_y1 = 0
                    line_y2 = roi_h - 1
                    if abs(vy) > 1e-6:
                        t1 = (line_y1 - y0) / vy
                        t2 = (line_y2 - y0) / vy
                        t_top = np.clip(t1, -1e6, 1e6)
                        t_bot = np.clip(t2, -1e6, 1e6)
                        line_x1 = int(x0 + vx * t_top)
                        line_x2 = int(x0 + vx * t_bot)
                    else:
                        line_x1 = line_x2 = int(x0)
                    
                    cv2.line(vis_frame, 
                            (line_x1, line_y1 + roi_y_start), 
                            (line_x2, line_y2 + roi_y_start), 
                            (255, 0, 0), 2)
                    
                    # Точка пересечения (БЕЗ повторной инверсии для визуализации!)
                    # x_bottom уже инвертирован выше, рисуем как есть
                    bottom_point_x = int(x_bottom)
                    bottom_point_y = roi_y_start + roi_h - 1
                    cv2.circle(vis_frame, (bottom_point_x, bottom_point_y), 8, (0, 0, 255), -1)
                    
                    # Линия ошибки
                    cv2.line(vis_frame, 
                            (roi_center_x, bottom_point_y), 
                            (bottom_point_x, bottom_point_y), 
                            (255, 255, 255), 2)

                    # ========== РАСЧЁТ ОШИБКИ ========== #
                    center = roi_w / 2.0
                    error_raw = (x_bottom - center) / center
                    
                    # DEBUG: показываем направление ошибки
                    error_direction = "LEFT" if error_raw < 0 else "RIGHT"
                    
                    last_error_sign = np.sign(error_raw) if abs(error_raw) > 1e-3 else last_error_sign

                    # Сглаживание
                    error_filtered = (
                        error_smooth_alpha * error_raw
                        + (1.0 - error_smooth_alpha) * error_filtered
                    )

                    # ========== PID ========== #
                    now = time.time()
                    dt = now - prev_time
                    if dt <= 0:
                        dt = 1e-3
                    prev_time = now

                    integral += error_filtered * dt
                    integral = max(-integral_limit, min(integral, integral_limit))

                    derivative = (error_filtered - prev_error) / dt
                    prev_error = error_filtered

                    u = Kp * error_filtered + Ki * integral + Kd * derivative
                    u = max(-1.0, min(1.0, u))

                    # Адаптация скорости по углу
                    angle_abs = min(abs(angle_deg), 90.0)
                    straightness = angle_abs / 90.0
                    speed_factor = 0.6 + 0.4 * straightness
                    v = base_speed * speed_factor

                    # Буст на прямых
                    if abs(error_filtered) < deadband_error and straightness > 0.85:
                        straight_boost = min(
                            straight_boost + straight_boost_step, straight_boost_max
                        )
                    else:
                        straight_boost *= 0.88

                    v = v + straight_boost
                    v = max(25.0, min(100.0, v))

                    omega = u * v
                    left_cmd = v - omega
                    right_cmd = v + omega

                    left_cmd = max(-100, min(100, left_cmd))
                    right_cmd = max(-100, min(100, right_cmd))

                    # === SWAP: моторы подключены наоборот === #
                    set_wheel_speeds(right_cmd, left_cmd)  # поменял местами!

                    if abs(error_filtered) < deadband_error:
                        direction_label = "go"
                    elif error_filtered < 0:
                        direction_label = "left"
                    else:
                        direction_label = "right"

                else:
                    has_line = False

            if not has_line:
                no_line_frames += 1
                if no_line_frames < no_line_max:
                    v = base_speed * 0.6
                    left_cmd = v
                    right_cmd = v
                    set_wheel_speeds(left_cmd, right_cmd)
                    direction_label = "fallback_forward"
                else:
                    search_dir = -1 if last_error_sign < 0 else 1
                    if search_dir < 0:
                        left_cmd = -search_turn_speed
                        right_cmd = search_turn_speed
                        direction_label = "search_left"
                    else:
                        left_cmd = search_turn_speed
                        right_cmd = -search_turn_speed
                        direction_label = "search_right"
                    rotate_in_place(search_dir, search_turn_speed)

            # ========== ТЕКСТ НА ВИДЕО ========== #
            detector_text = "YOLO" if use_yolo else "THRESH"
            cv2.putText(vis_frame, f"[{detector_text}] {direction_label}", (10, 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
            
            # Показываем направление ошибки для отладки
            if has_line:
                err_dir_text = f"ERR_DIR: {error_direction}"
                cv2.putText(vis_frame, err_dir_text, (10, 140),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
            
            cv2.putText(vis_frame, f"Err: {error_filtered:.3f} (raw: {error_raw:.3f})", (10, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.putText(vis_frame, f"Ang: {angle_deg:.1f} V: {v:.1f} Boost: {straight_boost:.1f}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.putText(vis_frame, f"L: {left_cmd:.1f} R: {right_cmd:.1f}", (10, 80),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.putText(vis_frame, f"FPS: {fps:.1f}  Pixels: {line_pixels}", (10, 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.putText(vis_frame, f"X: {x_bottom:.1f}  Ctr: {center:.1f}", (10, 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)

            video_writer.write(vis_frame)
            label_file.write(
                f"{frame_count},{detector_text},{direction_label},"
                f"{error_raw:.4f},{error_filtered:.4f},{angle_deg:.2f},"
                f"{v:.1f},{straight_boost:.1f},{left_cmd:.1f},{right_cmd:.1f},{fps:.1f},"
                f"{x_bottom:.2f},{center:.2f},{line_pixels},"
                f"{error_direction if has_line else 'N/A'}\n"
            )

            frame_count += 1
            if frame_count % 30 == 0:
                print(
                    f"[{detector_text}] frames: {frame_count}, dir: {direction_label}, "
                    f"err={error_filtered:.3f}, x={x_bottom:.1f}/{center:.1f}, "
                    f"pixels={line_pixels}, fps={fps:.1f}"
                )

            time.sleep(0.001)

    except KeyboardInterrupt:
        pass
    finally:
        print("Stopping...")
        motor_stop()
        cam_thread.stop()
        cam.stop()
        video_writer.release()
        label_file.close()
        cv2.destroyAllWindows()
        GPIO.cleanup()


if __name__ == "__main__":
    main()