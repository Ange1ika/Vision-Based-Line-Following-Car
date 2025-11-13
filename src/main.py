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
    
    print("🚗 Запуск: Визуальная детекция + углы 90° + доворот")
    camera = MyPiCamera(320, 240)
    motors = MotorController()
    ctrl = VisionController(
        camera, motors,
        base_speed=45,
        turn_speed=68,
        slowdown_factor=0.8,
        maneuver_timeout=0.2,
        min_line_pixels=700,
        use_yolo=True
    )

    # === Настройки записи видео ===
    save_dir = os.path.expanduser("./videos")
    os.makedirs(save_dir, exist_ok=True)
    video_path = os.path.join(save_dir, f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.avi")

    # Размер видео (можно подстроить под твою камеру)
    frame_w, frame_h = 440, 240
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    writer = cv2.VideoWriter(video_path, fourcc, 3.0, (frame_w, frame_h))

    if not writer.isOpened():
        print("❌ Ошибка: не удалось открыть VideoWriter. Проверь кодек или путь.")
        writer = None
    else:
        print(f"[INFO] 🎥 Видео будет сохранено в: {video_path}")

    # === Основной цикл ===
    try:
        print("✅ Готово. Для остановки нажми Ctrl+C.")
        while True:
            vis = ctrl.step(debug=True)
            if vis is not None and writer is not None:
                # Убедимся, что размер совпадает
                vis_resized = cv2.resize(vis, (frame_w, frame_h))
                writer.write(vis_resized)

            # без показа окна:
            # cv2.imshow("Line Follower", vis)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break

            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\n🛑 Остановка пользователем")

    finally:
        print("💾 Сохраняем и закрываем...")
        if writer is not None:
            writer.release()
        motors.cleanup()
        camera.release()
        ctrl.close()
        cv2.destroyAllWindows()
        print("✅ Завершено")


if __name__ == "__main__":
    main()
