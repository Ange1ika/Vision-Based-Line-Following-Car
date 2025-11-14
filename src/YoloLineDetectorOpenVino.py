import cv2
import numpy as np
from openvino.runtime import Core

class YOLOLineDetectorOpenVINO:
    def __init__(self,
                 ov_xml="checkpoints/yolov8n_seg_last/best_openvino_model/best.xmls",
                 img_size=320,
                 conf_thresh=0.7,
                 iou_thresh=0.45,
                 min_contour_area=60):
        
        self.img_size = img_size
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.min_contour_area = min_contour_area

        # ==== Load OpenVINO model ====
        core = Core()
        self.model = core.read_model(ov_xml)
        self.compiled = core.compile_model(self.model, "CPU")
        self.infer = self.compiled.create_infer_request()

        print("✓ OpenVINO YOLOv8-seg loaded:", ov_xml)

        self.input_name = self.model.inputs[0].get_any_name()
        self.output_names = [o.get_any_name() for o in self.model.outputs]

    def threshold(self, frame_bgr):
        h0, w0 = frame_bgr.shape[:2]

        # --- Preprocess ---
        img_resized = cv2.resize(frame_bgr, (self.img_size, self.img_size))
        inp = img_resized.astype(np.float32) / 255.0
        inp = np.expand_dims(inp.transpose(2, 0, 1), 0)  # NCHW

        # --- Inference ---
        self.infer.set_tensor(self.input_name, inp)
        self.infer.infer()

        det_out = self.infer.get_tensor(self.output_names[0]).data[0]   # (37, N)
        proto   = self.infer.get_tensor(self.output_names[1]).data[0]   # (80,80,32)

        boxes = det_out[:4].T
        scores = det_out[4]
        mask_coef = det_out[5:].T

        keep = scores > self.conf_thresh
        boxes, scores, mask_coef = boxes[keep], scores[keep], mask_coef[keep]

        if len(scores) == 0:
            return np.zeros((h0, w0), np.uint8)

        # ==== NMS ====
        boxes_xywh = []
        for cx, cy, w, h in boxes:
            x1 = int((cx - w/2) * w0 / self.img_size)
            y1 = int((cy - h/2) * h0 / self.img_size)
            boxes_xywh.append([x1, y1, int(w*w0/self.img_size), int(h*h0/self.img_size)])

        idxs = cv2.dnn.NMSBoxes(boxes_xywh, scores.tolist(),
                                 self.conf_thresh, self.iou_thresh)
        if len(idxs) == 0:
            return np.zeros((h0, w0), np.uint8)

        idxs = np.array(idxs).flatten()
        boxes = boxes[idxs]
        mask_coef = mask_coef[idxs]

        # ==== Mask reconstruction ====
        proto_flat = proto.reshape(-1, proto.shape[-1])
        masks = 1/(1+np.exp(-proto_flat @ mask_coef.T))
        masks = masks.reshape(80, 80, -1)

        final_mask = np.zeros((h0, w0), np.uint8)

        for i, (cx, cy, w, h) in enumerate(boxes):
            x1 = int((cx - w/2) * w0 / self.img_size)
            y1 = int((cy - h/2) * h0 / self.img_size)
            x2 = int((cx + w/2) * w0 / self.img_size)
            y2 = int((cy + h/2) * h0 / self.img_size)

            mask = cv2.resize(masks[..., i], (w0, h0))
            mask_bin = (mask > 0.5).astype(np.uint8) * 255

            crop = np.zeros_like(mask_bin)
            crop[y1:y2, x1:x2] = mask_bin[y1:y2, x1:x2]
            final_mask |= crop

        # Morphology
        kernel = np.ones((3, 3), np.uint8)
        final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel, 1)

        return final_mask
