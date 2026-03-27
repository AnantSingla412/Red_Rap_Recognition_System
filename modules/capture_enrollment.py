import cv2
import os
import numpy as np
import time


def capture_face_id_style(person_name, num_images=8):
    save_dir = f"dataset/{person_name}"
    os.makedirs(save_dir, exist_ok=True)

    detector = cv2.FaceDetectorYN_create(
        "models/face_detection_yunet_2023mar.onnx",
        "", (640, 480), 0.6, 0.3, 10
    )

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("cannot open webcam")
        return 0

    captured = 0
    last_center = None
    last_capture_time = 0
    last_saved_crop = None
    saved_centers = []

    COOLDOWN_SEC = 1.0
    MIN_DIVERSITY_PX = 20
    STABLE_FRAMES = 5
    stable_count = 0
    last_stable_center = None

    print(f"\nenrolling: {person_name}")
    print("move your head: centre, left, right, up, down, farther, closer — camera captures automatically\n")

    while captured < num_images:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        detector.setInputSize((w, h))
        _, faces = detector.detect(frame)

        display = frame.copy()
        now = time.time()

        cv2.rectangle(display, (0, 0), (w, 85), (0, 0, 0), -1)
        cv2.rectangle(display, (0, h - 75), (w, h), (0, 0, 0), -1)

        face_detected = faces is not None and len(faces) > 0

        if face_detected:
            face = faces[0]
            x, y = int(face[0]), int(face[1])
            fw, fh = int(face[2]), int(face[3])
            cx, cy = x + fw // 2, y + fh // 2

            if last_stable_center is None:
                last_stable_center = (cx, cy)
                stable_count = 0
            else:
                dist_stable = np.hypot(cx - last_stable_center[0],
                                       cy - last_stable_center[1])
                if dist_stable < 8:
                    stable_count += 1
                else:
                    stable_count = 0
                    last_stable_center = (cx, cy)

            diverse_enough = all(
                np.hypot(cx - sc[0], cy - sc[1]) >= MIN_DIVERSITY_PX
                for sc in saved_centers
            ) if saved_centers else True

            time_ok = (now - last_capture_time) >= COOLDOWN_SEC
            face_stable = stable_count >= STABLE_FRAMES
            ready = diverse_enough and time_ok and face_stable

            if ready:
                pad = int(fh * 0.3)
                fx1 = max(0, x - pad)
                fy1 = max(0, y - pad)
                fx2 = min(w, x + fw + pad)
                fy2 = min(h, y + fh + pad)
                face_crop = frame[fy1:fy2, fx1:fx2].copy()

                img_path = os.path.join(save_dir, f"{captured + 1}.jpg")
                cv2.imwrite(img_path, face_crop)
                last_saved_crop = face_crop.copy()
                saved_centers.append((cx, cy))

                last_center = (cx, cy)
                last_capture_time = now
                stable_count = 0
                captured += 1
                print(f"  [OK] Captured {captured}/{num_images}  ->  {img_path}")

                flash = np.ones_like(display) * 255
                cv2.addWeighted(flash, 0.35, display, 0.65, 0, display)

            for sc in saved_centers:
                cv2.circle(display, sc, 6, (0, 210, 90), -1)

            box_color = (0, 255, 0) if ready else (0, 200, 255)
            cv2.rectangle(display, (x, y), (x + fw, y + fh), box_color, 2)

            for i in range(STABLE_FRAMES):
                dot_color = (0, 255, 0) if i < stable_count else (70, 70, 70)
                cv2.circle(display, (x + 4 + i * 14, y - 12), 5, dot_color, -1)

            if not diverse_enough:
                status = "move to a new position"
                status_color = (0, 180, 255)
            elif not face_stable:
                status = f"hold still... ({stable_count}/{STABLE_FRAMES})"
                status_color = (0, 220, 255)
            elif not time_ok:
                remaining = COOLDOWN_SEC - (now - last_capture_time)
                status = f"wait {remaining:.1f}s..."
                status_color = (0, 180, 255)
            else:
                status = "capturing!"
                status_color = (0, 255, 100)

            cv2.putText(display, status, (10, 68),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, status_color, 2)

        else:
            cv2.putText(display,
                        "no face detected — move closer or improve lighting",
                        (10, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.putText(display,
                    "move head: centre, left, right, up, down, farther, closer",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

        bar_x1, bar_y1 = 20, h - 50
        bar_x2, bar_y2 = w - 20, h - 25
        bar_filled = int((captured / num_images) * (bar_x2 - bar_x1))
        cv2.rectangle(display, (bar_x1, bar_y1), (bar_x2, bar_y2), (60, 60, 60), -1)
        if bar_filled > 0:
            cv2.rectangle(display, (bar_x1, bar_y1),
                          (bar_x1 + bar_filled, bar_y2), (0, 210, 90), -1)
        cv2.putText(display, f"{captured}/{num_images} captured",
                    (bar_x1, bar_y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

        if last_saved_crop is not None:
            try:
                thumb = cv2.resize(last_saved_crop, (80, 80))
                display[5:85, w - 90:w - 10] = thumb
                cv2.rectangle(display, (w - 90, 5), (w - 10, 85), (0, 210, 90), 2)
                cv2.putText(display, "last saved", (w - 90, 98),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1)
            except Exception:
                pass

        cv2.putText(display, "Q = cancel", (w - 100, h - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (140, 140, 140), 1)

        cv2.imshow(f"Enrolling: {person_name}", display)

        if cv2.waitKey(30) & 0xFF == ord('q'):
            print("cancelled.")
            break

    cap.release()
    cv2.destroyAllWindows()

    print()
    if captured == num_images:
        print(f"done. {captured} images saved to dataset/{person_name}/")
    else:
        print(f"only {captured}/{num_images} captured, run again for better accuracy")

    return captured


if __name__ == "__main__":
    while True:
        name = input("\nenter person name (or 'done' to finish): ").strip()
        if not name or name.lower() == "done":
            break
        count = capture_face_id_style(name, num_images=8)
        if count > 0:
            print("next step -> run: python enroll.py")

    print("\nall done. run enroll.py to build embeddings.")
