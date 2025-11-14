from ultralytics import YOLO

model = YOLO("checkpoints/yolov8n_seg_last/best.pt")
model.export(format="openvino", imgsz=320, half=True)   # FP16 для Raspberry
