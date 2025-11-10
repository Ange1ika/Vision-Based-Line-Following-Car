# import cv2
# import numpy as np
# from openvino.runtime import Core
# from openvino.runtime import Tensor

# class YOLOLineDetectorOpenVINO:
#     def __init__(self,
#                  ov_xml="/home/raspberry/Desktop/data_mining/Vision-Based-Line-Following-Car/checkpoints/yolov8n_seg_last/best_openvino_model/best.xml",
#                  img_size=320,
#                  conf_thresh=0.7,
#                  iou_thresh=0.45,
#                  min_contour_area=60):

#         self.img_size = img_size
#         self.conf_thresh = conf_thresh
#         self.iou_thresh = iou_thresh
#         self.min_contour_area = min_contour_area

#         core = Core()
#         self.model = core.read_model(ov_xml)
#         self.compiled = core.compile_model(self.model, "CPU")
#         self.infer = self.compiled.create_infer_request()

#         # ✅ Берём вход не по имени, а просто первый input
#         self.input_idx = 0  # единственный вход
#         # ✅ Выходы будем брать по индексам (0, 1)
#         self.num_outputs = len(self.model.outputs)

#         print(f"✓ OpenVINO YOLOv8-seg loaded: {ov_xml}")
#         print(f"  Inputs: {len(self.model.inputs)}, Outputs: {self.num_outputs}")

#     def threshold(self, frame_bgr):
#         h0, w0 = frame_bgr.shape[:2]

#         # --- Preprocess: BGR -> NCHW, [0..1]
#         img_resized = cv2.resize(frame_bgr, (self.img_size, self.img_size))
#         inp = img_resized.astype(np.float32) / 255.0
#         # NCHW
#         inp = np.expand_dims(inp.transpose(2, 0, 1), 0)

#         # --- Inference ---
#         tensor = Tensor(inp)    
#         self.infer.set_input_tensor(self.input_idx, tensor)
#         #self.infer.set_input_tensor(self.input_idx, inp)
#         self.infer.infer()

#         # ✅ Берём выходы по индексам, без имён
#         det_out = self.infer.get_output_tensor(0).data[0]  # (37, N)
#         proto   = self.infer.get_output_tensor(1).data[0]  # (80, 80, 32) или подобное

#         # ===== Дальше всё как раньше =====
#         boxes = det_out[:4].T
#         scores = det_out[4]
#         mask_coef = det_out[5:].T

#         keep = scores > self.conf_thresh
#         boxes = boxes[keep]
#         scores = scores[keep]
#         mask_coef = mask_coef[keep]

#         if len(scores) == 0:
#             return np.zeros((h0, w0), np.uint8)

#         # NMS
#         boxes_xywh = []
#         for cx, cy, w, h in boxes:
#             x1 = int((cx - w / 2) * w0 / self.img_size)
#             y1 = int((cy - h / 2) * h0 / self.img_size)
#             ww = int(w * w0 / self.img_size)
#             hh = int(h * h0 / self.img_size)
#             boxes_xywh.append([x1, y1, ww, hh])

#         idxs = cv2.dnn.NMSBoxes(boxes_xywh, scores.tolist(),
#                                  self.conf_thresh, self.iou_thresh)
#         if len(idxs) == 0:
#             return np.zeros((h0, w0), np.uint8)

#         idxs = np.array(idxs).flatten()
#         boxes = boxes[idxs]
#         mask_coef = mask_coef[idxs]

#         # Mask reconstruction
#         proto_flat = proto.reshape(-1, proto.shape[-1])      # [H*W, C]
#         masks = 1.0 / (1.0 + np.exp(-proto_flat @ mask_coef.T))
#         Hm, Wm, _ = proto.shape
#         masks = masks.reshape(Hm, Wm, -1)                    # [H, W, N]

#         final_mask = np.zeros((h0, w0), np.uint8)

#         for i, (cx, cy, w, h) in enumerate(boxes):
#             x1 = max(0, min(w0 - 1, int((cx - w / 2) * w0 / self.img_size)))
#             y1 = max(0, min(h0 - 1, int((cy - h / 2) * h0 / self.img_size)))
#             x2 = max(0, min(w0, int((cx + w / 2) * w0 / self.img_size)))
#             y2 = max(0, min(h0, int((cy + h / 2) * h0 / self.img_size)))

#             mask = cv2.resize(masks[..., i], (w0, h0))
#             mask_bin = (mask > 0.5).astype(np.uint8) * 255

#             crop = np.zeros_like(mask_bin)
#             crop[y1:y2, x1:x2] = mask_bin[y1:y2, x1:x2]
#             final_mask |= crop

#         kernel = np.ones((3, 3), np.uint8)
#         final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel, 1)
#         final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, kernel, 1)

#         return final_mask
