from ultralytics import YOLO
import cv2
import numpy as np


class HatDetector:
    def __init__(self, model_path="models/best_model.pt", conf_threshold=0.5):
        self.conf_threshold = conf_threshold
        self.model = YOLO(model_path)
        print(f"hat detector loaded, classes: {self.model.names}")

        # warmup to avoid slow first inference
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        self.model(dummy, imgsz=320, verbose=False)


    def detect_in_frame(self, frame, face_bbox):
        if frame is None or frame.size == 0:
            return False, 0.0, None

        x, y, fw, fh = face_bbox
        H, W = frame.shape[:2]

        head_x1 = max(0, x - int(fw * 0.15))
        head_y1 = max(0, y - int(fh * 1.2))
        head_x2 = min(W, x + fw + int(fw * 0.15))
        head_y2 = min(H, y + int(fh * 0.2))

        try:
            results = self.model(frame, imgsz=320, verbose=False)

            best_conf = 0.0
            best_bbox = None

            for result in results:
                for box in result.boxes:
                    conf = float(box.conf[0])
                    class_id = int(box.cls[0])
                    name = self.model.names[class_id]

                    if name != 'cap':
                        continue
                    if conf < self.conf_threshold:
                        continue

                    bx1, by1, bx2, by2 = map(int, box.xyxy[0])

                    cap_cx = (bx1 + bx2) / 2
                    cap_cy = (by1 + by2) / 2

                    center_in_head = (head_x1 <= cap_cx <= head_x2 and
                                      head_y1 <= cap_cy <= head_y2)

                    ox1 = max(bx1, head_x1)
                    oy1 = max(by1, head_y1)
                    ox2 = min(bx2, head_x2)
                    oy2 = min(by2, head_y2)

                    overlap_w = max(0, ox2 - ox1)
                    overlap_h = max(0, oy2 - oy1)
                    overlap_area = overlap_w * overlap_h
                    cap_area = (bx2 - bx1) * (by2 - by1)

                    overlap_ok = (cap_area > 0 and
                                  (overlap_area / cap_area) >= 0.30)

                    if center_in_head or overlap_ok:
                        if conf > best_conf:
                            best_conf = conf
                            best_bbox = [bx1, by1, bx2 - bx1, by2 - by1]

            if best_bbox is not None:
                return True, best_conf, best_bbox

            return False, 0.0, None

        except Exception as e:
            print(f"hat detector error: {e}")
            return False, 0.0, None
