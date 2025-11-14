from ultralytics import YOLO

model = YOLO("/home/angelika/Desktop/Seoul/Vision-Based-Line-Following-Car/checkpoints/yolov8s_seg/yolov8s_550ep.pt")
model.export(format="openvino", imgsz=640, half=True)   # FP16 для Raspberry
