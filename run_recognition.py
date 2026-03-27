# run_recognition.py — clean version of main.py, NO debug overlay
import cv2
import numpy as np
import time
import os
import argparse

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from modules.face_detector    import FaceDetector
from modules.hat_detector     import HatDetector
from modules.color_checker    import RedCapChecker
from modules.arcface_recognizer import ArcFaceRecognizer
from modules.tracker          import FaceTracker

parser = argparse.ArgumentParser()
parser.add_argument("--threshold", type=float, default=0.35)
args = parser.parse_args()


def main():
    face_detector = FaceDetector("models/face_detection_yunet_2023mar.onnx")
    hat_detector  = HatDetector("models/best_model.pt", conf_threshold=0.5)
    color_checker = RedCapChecker()
    recognizer    = ArcFaceRecognizer(
                        model_path="models/w600k_r50.onnx",
                        embeddings_path="embeddings/known_faces.pkl",
                        similarity_threshold=args.threshold
                    )
    tracker       = FaceTracker()

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("[ERROR] Cannot open webcam.")
        return

    fps_start   = time.time()
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        faces   = face_detector.detect(frame)
        bboxes  = [f["bbox"] for f in faces]
        tracked = tracker.update(bboxes)

        for track_id, bbox in tracked:
            state        = tracker.get_state(track_id)
            x, y, fw, fh = bbox

            if fw < 40 or fh < 40:
                continue

            if state.needs_hat_check():
                hat_found, hat_conf, hat_bbox = hat_detector.detect_in_frame(
                    frame, bbox
                )

                is_red    = False
                red_ratio = 0.0

                if hat_found and hat_bbox is not None:
                    hx, hy, hw2, hh2 = hat_bbox
                    hx  = max(0, hx)
                    hy  = max(0, hy)
                    hx2 = min(frame.shape[1], hx + hw2)
                    hy2 = min(frame.shape[0], hy + hh2)
                    hat_crop = frame[hy:hy2, hx:hx2]
                    if hat_crop.size > 0:
                        is_red, red_ratio = color_checker.is_red(hat_crop)

                state.update_hat(hat_found, is_red, red_ratio)

            if state.is_red_cap and state.needs_recognition():
                face_crop = frame[y:y + fh, x:x + fw]
                if face_crop.size > 0:
                    identity, similarity = recognizer.recognize(face_crop)
                    state.update_identity(identity, similarity)

            _draw_result(frame, state, bbox)

        # FPS counter — clean, top left
        elapsed = time.time() - fps_start
        fps     = frame_count / elapsed if elapsed > 0 else 0
        cv2.putText(frame, f"FPS: {fps:.1f}",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                    0.60, (0, 255, 255), 2)

        cv2.imshow("Red Cap Recognition  |  Q to quit", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def _draw_result(frame, state, bbox):
    x, y, fw, fh = bbox

    if state.is_red_cap:
        box_color = (0, 255, 0)
        cv2.rectangle(frame, (x, y), (x + fw, y + fh), box_color, 3)
        name_part = "Not Registered" if state.identity == "Unknown" else state.identity
        label     = f"{name_part}  |  Red Cap"
        _draw_label(frame, label, x, y, box_color)
    else:
        cv2.rectangle(frame, (x, y), (x + fw, y + fh), (255, 255, 0), 2)


def _draw_label(frame, text, x, y, color):
    font       = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.65
    thickness  = 2
    padding    = 6

    (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)

    label_x1 = x
    label_y1 = max(0, y - text_h - padding * 2)
    label_x2 = x + text_w + padding * 2
    label_y2 = y

    cv2.rectangle(frame, (label_x1, label_y1), (label_x2, label_y2), (0, 0, 0), -1)
    cv2.rectangle(frame, (label_x1, label_y1), (label_x2, label_y1 + 3), color, -1)
    cv2.putText(frame, text, (x + padding, y - padding),
                font, font_scale, color, thickness)


if __name__ == "__main__":
    main()
