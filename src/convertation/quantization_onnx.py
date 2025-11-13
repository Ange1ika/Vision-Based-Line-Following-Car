"""optional"""

from onnxruntime.quantization import quantize_dynamic, QuantType

quantize_dynamic(
    model_input="checkpoints/last_model/best.onnx",
    model_output="checkpoints/last_model/best_int8.onnx",
    weight_type=QuantType.QInt8
)
print("✅ Сохранено: best_int8.onnx")
