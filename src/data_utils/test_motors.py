import time
from motor_controller import MotorController

motors = MotorController()

print("\n=== ТЕСТ 1: Левый мотор вперёд (40) ===")
motors.set_speed(40, 0)
time.sleep(2)
motors.stop()

print("\n=== ТЕСТ 2: Левый мотор назад (-40) ===")
motors.set_speed(-40, 0)
time.sleep(5)
motors.stop()

print("\n=== ТЕСТ 3: Правый мотор вперёд (40) ===")
motors.set_speed(0, 40)
time.sleep(5)
motors.stop()

print("\n=== ТЕСТ 4: Правый мотор назад (-40) ===")
motors.set_speed(0, -40)
time.sleep(2)
motors.stop()

print("\n=== ТЕСТ 5: Вперёд оба ===")
motors.set_speed(40, 40)
time.sleep(2)
motors.stop()

print("\n=== ТЕСТ 6: Назад оба ===")
motors.set_speed(-40, -40)
time.sleep(2)
motors.stop()

print("\n=== ТЕСТ 7: Вращение на месте (влево) ===")
motors.set_speed(-40, 40)
time.sleep(2)
motors.stop()

print("\n=== ТЕСТ 8: Вращение на месте (вправо) ===")
motors.set_speed(40, -40)
time.sleep(2)
motors.stop()

motors.cleanup()
