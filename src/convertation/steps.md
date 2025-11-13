https://discuss.roboflow.com/t/interpreting-yolov8-tflite-output/1562/4



convert yolo.pth to onnx:

```
from ultralytics import YOLO

model = YOLO("best.pt")

model.export(format='onnx',
simplify=True, # Let YOLO do the simplification
opset=11, # Use older opset
int8=True,
dynamic=False) # Static shapes

```
create separate env for convert onnx->tf->tflite 
```
conda create -n tf_convert python=3.9
conda activate tf_convert

pip install tensorflow==2.10.1
pip install numpy==1.26.4
pip install protobuf==3.19.6
pip install onnx2tf==1.17.0
pip install onnx==1.14.1  #
```

 ```
onnx2tf -i "checkpoints/last_model/best.onnx" -o checkpoints/last_model/tflite_export/ -ois 1,3,320,320

  ```
saved_model output started ==========================================================
saved_model output complete!
Estimated count of arithmetic ops: 3.126 G  ops, equivalently 1.563 G  MACs
Float32 tflite output complete!
Estimated count of arithmetic ops: 3.126 G  ops, equivalently 1.563 G  MACs
Float16 tflite output complete!


check model after pth to onnx: CHECK_ONNX.PY
DEBUD:

TypeError: Unable to convert function return value to a Python type! The signature was
        () -> handle

pip install "numpy<1.24" --force-reinstall




onnx2tf -i checkpoints/last_model/best.onnx -o tflite_export/ -ois 1,3,320,320 -prf checkpoints/last_model/parameter_replacement.json

onnx2tf -i checkpoints/last_model/best.pt -b 1,3,320,320 -onwdt

onnx2tf   -i "checkpoints/best_500ep.pt"   -o tflite_export/   -ois 1,3,320,320 -onwdt