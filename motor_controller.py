import time
import csv

try:
    import RPi.GPIO as GPIO
    GPIO_AVAILABLE = True
except Exception:
    GPIO_AVAILABLE = False


class MotorController:
    """
    Драйвер L298N (или аналог) под BCM-пины.
    Поддерживает управление скоростью в диапазоне [-100..100].
    Ведёт телеметрию всех команд.
    """
    def __init__(self, telemetry_path="telemetry/motor_telemetry.csv"):
        self.MOTOR_PINS = {
            'L_PWM': 18,
            'L_IN1': 22,
            'L_IN2': 27,
            'R_PWM': 23,
            'R_IN1': 25,
            'R_IN2': 24
        }
        self._mock_left = 0
        self._mock_right = 0
        self.telemetry_path = telemetry_path

        # Инициализация GPIO
        if GPIO_AVAILABLE:
            GPIO.setmode(GPIO.BCM)
            GPIO.setwarnings(False)
            for p in self.MOTOR_PINS.values():
                GPIO.setup(p, GPIO.OUT)
            self.L_Motor = GPIO.PWM(self.MOTOR_PINS['L_PWM'], 500)
            self.R_Motor = GPIO.PWM(self.MOTOR_PINS['R_PWM'], 500)
            self.L_Motor.start(0)
            self.R_Motor.start(0)
        else:
            self.L_Motor = None
            self.R_Motor = None

        # создаём файл телеметрии
        with open(self.telemetry_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "action", "left_speed", "right_speed"])

    # ------------------------------------------------------------
    def _log(self, action, left, right):
        """Записывает текущие данные в CSV"""
        with open(self.telemetry_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                time.strftime("%H:%M:%S"),
                action,
                int(left),
                int(right)
            ])

    # ------------------------------------------------------------
    def _apply_channel(self, side, speed):
        speed = max(min(speed, 100), -100)
        if not GPIO_AVAILABLE:
            if side == 'L':
                self._mock_left = speed
            else:
                self._mock_right = speed
            return

        if side == 'L':
            in1, in2, pwm = self.MOTOR_PINS['L_IN1'], self.MOTOR_PINS['L_IN2'], self.L_Motor
        else:
            in1, in2, pwm = self.MOTOR_PINS['R_IN1'], self.MOTOR_PINS['R_IN2'], self.R_Motor

        if speed >= 0:
            GPIO.output(in1, GPIO.LOW)
            GPIO.output(in2, GPIO.HIGH)
            pwm.ChangeDutyCycle(speed)
        else:
            GPIO.output(in1, GPIO.HIGH)
            GPIO.output(in2, GPIO.LOW)
            pwm.ChangeDutyCycle(-speed)

    # ------------------------------------------------------------
    def set_speed(self, left_speed, right_speed):
        """Основная функция для установки скоростей"""
        self._apply_channel('L', left_speed)
        self._apply_channel('R', right_speed)
        self._log("set_speed", left_speed, right_speed)

    def stop(self):
        self.set_speed(0, 0)
        self._log("stop", 0, 0)

    def turn_in_place(self, direction, speed, duration):
        """direction: -1 (влево), +1 (вправо)"""
        if direction < 0:
            self.set_speed(-speed, speed)
        else:
            self.set_speed(speed, -speed)
        self._log(f"turn_{'left' if direction<0 else 'right'}", 
                  -speed if direction<0 else speed,
                  speed if direction<0 else -speed)
        time.sleep(duration)
        self.stop()

    def move_forward(self, speed, duration):
        self.set_speed(speed, speed)
        self._log("forward", speed, speed)
        time.sleep(duration)
        self.stop()

    def cleanup(self):
        self.stop()
        if GPIO_AVAILABLE:
            self.L_Motor.stop()
            self.R_Motor.stop()
            import RPi.GPIO as GPIO
            GPIO.cleanup()
        self._log("cleanup", 0, 0)
