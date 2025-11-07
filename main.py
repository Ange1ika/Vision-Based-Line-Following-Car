import os
import cv2
import time
from datetime import datetime

from camera_module import MyPiCamera
from motor_controller import MotorController
from controller import VisionController


def main():
    print("🚗 Запуск: Визуальная детекция + углы 90° + доворот")
    camera = MyPiCamera(320, 240)  # для RPi. На ПК можно MyPiCamera(..., fallback_webcam=True)
    motors = MotorController()
    ctrl = VisionController(camera, motors,
                            base_speed=50,
                            turn_speed=65,
                            maneuver_timeout=1.5,
                            min_line_pixels=700)

    # видео запись (опционально)
    save_dir = os.path.expanduser("/home/raspberry/Desktop/data_mining/line_follower/videos")
    telemetry_path = os.path.expanduser("/home/raspberry/Desktop/data_mining/line_follower/telemetry")
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(telemetry_path, exist_ok=True)
    path = os.path.join(save_dir, f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.avi")
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'XVID'), 25.0, (320, 240))

    cv2.namedWindow("Line Follower", cv2.WINDOW_NORMAL)

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
