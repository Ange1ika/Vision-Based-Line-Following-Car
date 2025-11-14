from ultralytics import YOLO

model = YOLO("/home/angelika/Desktop/Seoul/Vision-Based-Line-Following-Car/checkpoints/yolov8n_seg_last/best.pt")
"""it is not so good way to convert yolo pth to tflite directly, better to convert pth to onnx first, then onnx to tflite
see src/convertation"""

model.export(format="onnx", imgsz=320) #imgsz important!

# model.export(format='tflite',
#              simplify=True, # Let YOLO do the simplification
#              opset=11,
#              half=True,
#              nms=False, # Use older opset
#              dynamic=False,
#              agnostic_nms=False) # Static shapes

#model.export(format='onnx', imgsz=320)
#simplify=True, # Let YOLO do the simplification
#nms=False,
#opset=11, # Use older opset
#dynamic=False) # Static shapes  

# model.export(
#     format="onnx",
#     opset=12,
#     simplify=True,
#     imgsz=320,
#     dynamic=False,      # ОБЯЗАТЕЛЬНО отключить
# )
