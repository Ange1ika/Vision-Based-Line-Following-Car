import tensorflow as tf
import numpy as np

interpreter = tf.lite.Interpreter(model_path="checkpoints/last_model/best_saved_model/best_float16.tflite")
interpreter.allocate_tensors()
outputs = interpreter.get_output_details()
for i, out in enumerate(outputs):
    print(f"[{i}] shape={out['shape']}, dtype={out['dtype']}")
