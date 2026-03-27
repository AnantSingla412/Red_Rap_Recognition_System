import cv2
import numpy as np


class RedCapChecker:
    def __init__(self,
                 hsv_red_ratio_threshold=0.20,
                 min_saturation=120,
                 min_value=70):
        self.hsv_red_ratio_threshold = hsv_red_ratio_threshold
        self.min_saturation = min_saturation
        self.min_value = min_value


    def is_red(self, roi):
        if roi is None or roi.size == 0:
            return False, 0.0

        roi = cv2.resize(roi, (64, 64))
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # red wraps around H=0/180 in OpenCV
        lower_red1 = np.array([0,   self.min_saturation, self.min_value])
        upper_red1 = np.array([10,  255, 255])
        lower_red2 = np.array([160, self.min_saturation, self.min_value])
        upper_red2 = np.array([180, 255, 255])

        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = cv2.bitwise_or(mask1, mask2)

        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        total_pixels = mask.shape[0] * mask.shape[1]
        red_pixels = np.count_nonzero(mask)
        red_ratio = red_pixels / total_pixels if total_pixels > 0 else 0.0

        is_red = red_ratio >= self.hsv_red_ratio_threshold
        return is_red, red_ratio


    def draw_debug(self, frame, roi_coords, is_red, red_ratio):
        if roi_coords is None:
            return
        x1, y1, x2, y2 = roi_coords
        color = (0, 0, 255) if is_red else (128, 128, 128)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
        label = f"RedCap:{red_ratio:.2f}" if is_red else f"NoRed:{red_ratio:.2f}"
        cv2.putText(frame, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
