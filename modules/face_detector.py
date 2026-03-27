import cv2
import numpy as np


class FaceDetector:
    def __init__(self, model_path="models/face_detection_yunet_2023mar.onnx",
                 conf_threshold=0.6, nms_threshold=0.3, top_k=10):
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.top_k = top_k
        self.detector = cv2.FaceDetectorYN_create(
            model_path,
            "",
            (320, 320),
            conf_threshold,
            nms_threshold,
            top_k
        )


    def detect(self, frame):
        h, w = frame.shape[:2]
        self.detector.setInputSize((w, h))
        _, faces = self.detector.detect(frame)

        results = []
        if faces is None:
            return results

        for face in faces:
            x, y, fw, fh = int(face[0]), int(face[1]), int(face[2]), int(face[3])
            confidence = float(face[14])
            landmarks = face[4:14].reshape(5, 2).astype(int)
            results.append({
                "bbox": [x, y, fw, fh],
                "confidence": confidence,
                "landmarks": landmarks
            })

        return results


    def get_head_roi(self, frame, bbox, expand_up=1.0, expand_side=0.15):
        h, w = frame.shape[:2]
        x, y, fw, fh = bbox

        expand_up_px = int(fh * expand_up)
        expand_side_px = int(fw * expand_side)

        x1 = max(0, x - expand_side_px)
        y1 = max(0, y - expand_up_px)
        x2 = min(w, x + fw + expand_side_px)
        y2 = min(h, y + fh)

        if y2 <= y1 or x2 <= x1:
            return None, None

        roi = frame[y1:y2, x1:x2]
        return roi, (x1, y1, x2, y2)
