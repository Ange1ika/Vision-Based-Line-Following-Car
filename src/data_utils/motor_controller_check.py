from motor_controller import MotorController
import time

motors = MotorController()

print("\n=== ТЕСТ 1: Левый мотор ВПЕРЁД (40) ===")
motors.set_speed(40, 0)
time.sleep(2)
motors.stop()

print("\n=== ТЕСТ 2: Левый мотор НАЗАД (-40) ===")
motors.set_speed(-40, 0)
time.sleep(2)
motors.stop()

print("\n=== ТЕСТ 3: Правый мотор ВПЕРЁД (40) ===")
motors.set_speed(0, 40)
time.sleep(2)
motors.stop()

print("\n=== ТЕСТ 4: Правый мотор НАЗАД (-40) ===")
motors.set_speed(0, -40)
time.sleep(2)
motors.stop()

print("\n=== ТЕСТ 5: Ехать ВПЕРЁД обоими ===")
motors.set_speed(40, 40)
time.sleep(2)
motors.stop()

print("\n=== ТЕСТ 6: Ехать НАЗАД обоими ===")
motors.set_speed(-40, -40)
time.sleep(2)
motors.stop()

print("\n=== ТЕСТ 7: Вращение РОБОТА ВЛЕВО ===")
motors.set_speed(-40, 40)  
time.sleep(2)
motors.stop()

print("\n=== ТЕСТ 8: Вращение РОБОТА ВПРАВО ===")
motors.set_speed(40, -40)
time.sleep(2)
motors.stop()

motors.cleanup()
