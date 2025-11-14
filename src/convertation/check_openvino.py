import cv2
import numpy as np
import time
from openvino import Core

MODEL_PATH = "/home/angelika/Desktop/Seoul/Vision-Based-Line-Following-Car/checkpoints/yolov8s_seg/yolov8s_550ep_openvino_model/yolov8s_550ep.xml"
TEST_IMAGE = "data/test_images/45.jpg"
IMG_SIZE = 640

ie = Core()
model = ie.read_model(MODEL_PATH)
compiled = ie.compile_model(model, "CPU")

input_layer = compiled.input(0)
infer = compiled.create_infer_request()

print("=== MODEL LOADED ===")
print(f"Input: {input_layer.shape}")

# -------------------------
# Load and preprocess image
# -------------------------
frame = cv2.imread(TEST_IMAGE)
orig_h, orig_w = frame.shape[:2]

img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
blob = img.transpose(2, 0, 1)[None].astype(np.float32) / 255.0

# -------------------------
# Inference + FPS
# -------------------------
t0 = time.time()
outputs = infer.infer({input_layer: blob})

# -------------------------
# Parse outputs
# -------------------------
det = outputs[list(outputs.keys())[0]]
proto = outputs[list(outputs.keys())[1]]

# det = (1, 37, 2100)
det = det[0]
boxes = det[:4].T
scores = det[4]
mask_coef = det[5:].T

# filter confidence
keep = scores > 0.03
if np.sum(keep) == 0:
    print("No masks detected")
    exit()

boxes = boxes[keep]
mask_coef = mask_coef[keep]

# proto = (1, 32, 80, 80)
proto = proto[0]  # (32, 80, 80)

# -------------------------
# Reconstruct masks
# -------------------------
proto_flat = proto.reshape(32, -1).T    # (6400, 32)
masks = 1 / (1 + np.exp(-(proto_flat @ mask_coef.T)))
masks = masks.reshape(80, 80, -1)       # (80, 80, N)

# take 1st mask only (line detector usually gives 1)
mask = masks[..., 0]
mask = cv2.resize(mask, (orig_w, orig_h))
mask_bin = (mask > 0.5).astype(np.uint8) * 255

# -------------------------
# Visualize
# -------------------------
color_mask = cv2.applyColorMap(mask_bin, cv2.COLORMAP_JET)
vis = cv2.addWeighted(frame, 0.7, color_mask, 0.3, 0)
dt = (time.time() - t0)*1000
fps = 1000.0 / dt

print(f"Inference time: {dt:.2f} ms ({fps:.1f} FPS)")

cv2.putText(vis, f"OpenVINO {fps:.1f} FPS  {dt:.1f} ms",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            (0, 255, 255), 1)

cv2.imshow("OpenVINO Segmentation", vis)
cv2.waitKey(0)
