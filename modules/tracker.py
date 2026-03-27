# modules/tracker.py
import time


class FaceState:
    def __init__(self, track_id):
        self.track_id = track_id
        self.bbox = None
        self.identity = "Unknown"
        self.similarity = 0.0

        self.is_red_cap = False
        self.hat_found = False
        self._cap_confirm_count = 0
        self._miss_counter = 0

        self.CAP_CONFIRM_NEEDED = 3
        self.NO_CAP_CONFIRM_NEEDED = 2

        self.last_recognition_time = 0
        self.last_hat_check_time = 0
        self.recognition_interval = 2.0
        self.hat_check_interval = 0.2

        self.debug_last_hat_found = False
        self.debug_last_is_red = False
        self.debug_last_ratio = 0.0
        self.debug_last_status = "INIT"


    def needs_hat_check(self):
        return (time.time() - self.last_hat_check_time) > self.hat_check_interval


    def needs_recognition(self):
        return (time.time() - self.last_recognition_time) > self.recognition_interval


    def update_hat(self, hat_found, is_red, red_ratio=0.0):
        self.hat_found = hat_found
        self.last_hat_check_time = time.time()

        self.debug_last_hat_found = hat_found
        self.debug_last_is_red = is_red
        self.debug_last_ratio = red_ratio

        if hat_found and is_red:
            self._cap_confirm_count += 1
            self._miss_counter = 0

            if self._cap_confirm_count >= self.CAP_CONFIRM_NEEDED:
                self.is_red_cap = True
                self.debug_last_status = f"CONFIRMED (hits={self._cap_confirm_count})"
            else:
                self.debug_last_status = f"BUILDING ({self._cap_confirm_count}/{self.CAP_CONFIRM_NEEDED})"

        elif hat_found and not is_red:
            # cap found but color check failed — likely motion blur or lighting
            self.debug_last_status = f"BLUR/COLOR_FAIL ratio={red_ratio:.2f} (ignored)"

        else:
            self._miss_counter += 1

            if self.is_red_cap:
                remaining = self.NO_CAP_CONFIRM_NEEDED - self._miss_counter
                self.debug_last_status = f"HOLDING (miss={self._miss_counter}, lose_in={remaining})"
            else:
                self.debug_last_status = f"NO_CAP (miss={self._miss_counter})"

            if self._miss_counter >= self.NO_CAP_CONFIRM_NEEDED:
                self._cap_confirm_count = 0
                self._miss_counter = 0
                self.is_red_cap = False
                self.identity = "Unknown"
                self.similarity = 0.0
                self.debug_last_status = "LOST — reset"


    def update_identity(self, identity, similarity):
        if identity != "Unknown" and similarity > self.similarity:
            self.identity = identity
            self.similarity = similarity
        elif self.identity == "Unknown" and identity != "Unknown":
            self.identity = identity
            self.similarity = similarity
        self.last_recognition_time = time.time()


class FaceTracker:
    def __init__(self, iou_threshold=0.4):
        self.states = {}
        self.next_id = 0
        self.iou_threshold = iou_threshold


    def update(self, detected_bboxes):
        matched = []
        matched_ids = set()

        for bbox in detected_bboxes:
            best_id, best_iou = None, 0.0
            for tid, state in self.states.items():
                if tid in matched_ids or state.bbox is None:
                    continue
                iou = self._iou(bbox, state.bbox)
                if iou > best_iou and iou >= self.iou_threshold:
                    best_iou = iou
                    best_id = tid

            if best_id is not None:
                self.states[best_id].bbox = bbox
                matched.append((best_id, bbox))
                matched_ids.add(best_id)
            else:
                new_state = FaceState(self.next_id)
                new_state.bbox = bbox
                self.states[self.next_id] = new_state
                matched.append((self.next_id, bbox))
                self.next_id += 1

        active_ids = {t[0] for t in matched}
        for tid in list(self.states.keys()):
            if tid not in active_ids:
                del self.states[tid]

        return matched


    def get_state(self, track_id):
        return self.states.get(track_id)


    @staticmethod
    def _iou(b1, b2):
        x1, y1, w1, h1 = b1
        x2, y2, w2, h2 = b2
        ax1, ay1, ax2, ay2 = x1, y1, x1+w1, y1+h1
        bx1, by1, bx2, by2 = x2, y2, x2+w2, y2+h2
        ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
        inter = max(0, ix2-ix1) * max(0, iy2-iy1)
        union = w1*h1 + w2*h2 - inter
        return inter / union if union > 0 else 0.0
