import os
import cv2
import time
from datetime import datetime

from camera_module import MyPiCamera
from motor_controller import MotorController
from controller import VisionController


def main():
    telemetry_path = "./telemetry"
    os.makedirs(telemetry_path, exist_ok=True)
    
    print("Запуск: Визуальная детекция + углы 90° + доворот")
    camera = MyPiCamera(320, 240) 
    motors = MotorController()
    ctrl = VisionController(camera, motors,
                            base_speed=0,
                            turn_speed=68,
                            slowdown_factor=0.8,
                            maneuver_timeout=0.2,
                            min_line_pixels=700,
                            use_yolo=True)
    save_dir = os.path.expanduser("./videos")
    os.makedirs(save_dir, exist_ok=True)
    
    path = os.path.join(save_dir, f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.avi")
    
    frame = camera.read()
    if frame is None:
        print("Не удалось получить кадр с камеры. Пропускаем VideoWriter.")
        writer = None
    else:
        vis = ctrl.step(debug=True)
        if vis is not None:
            h, w = vis.shape[:2]
            frame_size = (w, h)
            print(f"[INFO] Видеоразмер по debug: {frame_size}")

            path = os.path.join(save_dir, f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.avi")
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            writer = cv2.VideoWriter(path, fourcc, 10.0, frame_size, True)
            print(f"[INFO] 🎥 Запись видео → {path}")
        else:
            writer = None

    if not writer.isOpened():
        print("❌ Ошибка: VideoWriter не открылся. Проверь кодек или путь.")
    else:
        print(f"[INFO] 🎥 Запись видео → {path}")

    ctrl.base_speed = 50
    try:
        print("✅ Готово. Нажми q для выхода.")
        while True:
            vis = ctrl.step(debug=True)
            if vis is not None:
                writer.write(vis)
                cv2.imshow("Line Follower", vis)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\n🛑 Остановка пользователем")
    finally:
        motors.cleanup()
        camera.release()
        writer.release()
        ctrl.close()
        cv2.destroyAllWindows()
        print("✅ Завершено")

if __name__ == "__main__":
    main()