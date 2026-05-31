# pylint: disable=all

import cv2
import numpy as np
from pathlib import Path

# -----------------------------
# Config
# -----------------------------
INPUT_PATH = "negrito_cat_2.mp4"
OUTPUT_PATH = "kalman_tracking_output.mp4"

FRAME_STEP = 3  # sample every N frames
DIFF_THRESHOLD = 30  # pixel difference threshold
MIN_AREA = 300  # ignore tiny motion blobs
DILATE_ITER = 2
ERODE_ITER = 1
SLOW_FACTOR = 2.0  # >1.0 slows the output (e.g. 2.0 -> twice slower)


# -----------------------------
# Kalman Filter Wrapper
# -----------------------------
class BBoxKalmanTracker:
    """
    State:
        [cx, cy, vx, vy, w, h]^T

    Measurement:
        [cx, cy, w, h]^T
    """

    def __init__(self):
        self.kf = cv2.KalmanFilter(6, 4)

        # State transition matrix A
        # cx' = cx + vx
        # cy' = cy + vy
        # vx' = vx
        # vy' = vy
        # w'  = w
        # h'  = h
        self.kf.transitionMatrix = np.array(
            [
                [1, 0, 1, 0, 0, 0],
                [0, 1, 0, 1, 0, 0],
                [0, 0, 1, 0, 0, 0],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1],
            ],
            dtype=np.float32,
        )

        # Observation matrix H
        # We observe cx, cy, w, h
        self.kf.measurementMatrix = np.array(
            [
                [1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1],
            ],
            dtype=np.float32,
        )

        # Process noise: how much we allow motion model to vary
        self.kf.processNoiseCov = np.eye(6, dtype=np.float32) * 1e-2

        # Measurement noise: how noisy the bbox detector is
        self.kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * 5e-1

        # Initial uncertainty
        self.kf.errorCovPost = np.eye(6, dtype=np.float32)

        self.initialized = False

    def initialize(self, bbox):
        x, y, w, h = bbox
        cx = x + w / 2.0
        cy = y + h / 2.0

        self.kf.statePost = np.array(
            [
                [cx],
                [cy],
                [0],
                [0],
                [w],
                [h],
            ],
            dtype=np.float32,
        )

        self.initialized = True

    def predict(self):
        pred = self.kf.predict()
        return self._state_to_bbox(pred)

    def correct(self, bbox):
        x, y, w, h = bbox
        cx = x + w / 2.0
        cy = y + h / 2.0

        measurement = np.array(
            [
                [cx],
                [cy],
                [w],
                [h],
            ],
            dtype=np.float32,
        )

        corrected = self.kf.correct(measurement)
        return self._state_to_bbox(corrected)

    @staticmethod
    def _state_to_bbox(state):
        cx, cy, vx, vy, w, h = state.flatten()

        w = max(1, float(w))
        h = max(1, float(h))

        x = int(cx - w / 2.0)
        y = int(cy - h / 2.0)

        return int(x), int(y), int(w), int(h)


# -----------------------------
# Motion detector
# -----------------------------
def detect_motion_bbox(prev_gray, curr_gray):
    """
    Returns:
        bbox: (x, y, w, h) or None
        mask: binary motion mask
    """

    diff = cv2.absdiff(prev_gray, curr_gray)
    _, mask = cv2.threshold(diff, DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)

    # Clean noise
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=ERODE_ITER)
    mask = cv2.dilate(mask, kernel, iterations=DILATE_ITER)

    ys, xs = np.where(mask > 0)

    if len(xs) == 0 or len(ys) == 0:
        return None, mask

    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()

    w = x2 - x1 + 1
    h = y2 - y1 + 1

    if w * h < MIN_AREA:
        return None, mask

    return (int(x1), int(y1), int(w), int(h)), mask


# -----------------------------
# Visualization helpers
# -----------------------------
def draw_bbox(frame, bbox, color, label):
    x, y, w, h = bbox

    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(frame.shape[1] - 1, x + w)
    y2 = min(frame.shape[0] - 1, y + h)

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(
        frame,
        label,
        (x1, max(20, y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2,
        cv2.LINE_AA,
    )


def overlay_mask(frame, mask):
    colored_mask = np.zeros_like(frame)
    colored_mask[:, :, 2] = mask  # red channel
    return cv2.addWeighted(frame, 0.85, colored_mask, 0.35, 0)


# -----------------------------
# Main pipeline
# -----------------------------
def main():
    input_path = Path(INPUT_PATH)

    if not input_path.exists():
        raise FileNotFoundError(f"Could not find {INPUT_PATH}")

    cap = cv2.VideoCapture(str(input_path))

    if not cap.isOpened():
        raise RuntimeError(f"Could not open {INPUT_PATH}")

    original_fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Compute output FPS: reduce by FRAME_STEP sampling, and divide by SLOW_FACTOR
    # Example: original 30fps, FRAME_STEP=5 -> base 6fps. SLOW_FACTOR=2 -> output 3fps (twice slower)
    output_fps = max(0.5, (original_fps / FRAME_STEP) / max(1.0, SLOW_FACTOR))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        OUTPUT_PATH,
        fourcc,
        output_fps,
        (width, height),
    )

    tracker = BBoxKalmanTracker()

    prev_gray = None
    frame_idx = 0
    sampled_idx = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        if frame_idx % FRAME_STEP != 0:
            frame_idx += 1
            continue

        display = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        if prev_gray is None:
            prev_gray = gray
            frame_idx += 1
            sampled_idx += 1
            writer.write(display)
            continue

        detected_bbox, mask = detect_motion_bbox(prev_gray, gray)

        # Always predict after tracker has been initialized
        predicted_bbox = None
        if tracker.initialized:
            predicted_bbox = tracker.predict()
            draw_bbox(display, predicted_bbox, (255, 0, 0), "Kalman prediction")

        if detected_bbox is not None:
            if not tracker.initialized:
                tracker.initialize(detected_bbox)

            corrected_bbox = tracker.correct(detected_bbox)

            draw_bbox(display, detected_bbox, (0, 255, 255), "Sensor bbox")
            draw_bbox(display, corrected_bbox, (0, 255, 0), "Kalman corrected")
        else:
            corrected_bbox = predicted_bbox

        display = overlay_mask(display, mask)

        # Dibujar puntos: predicción (azul) y corrección por Kalman (rojo)
        if predicted_bbox is not None:
            px, py, pw, ph = predicted_bbox
            p_cx = int(px + pw / 2.0)
            p_cy = int(py + ph / 2.0)
            cv2.circle(display, (p_cx, p_cy), 4, (255, 0, 0), -1)  # azul

        if corrected_bbox is not None:
            kx, ky, kw, kh = corrected_bbox
            k_cx = int(kx + kw / 2.0)
            k_cy = int(ky + kh / 2.0)
            cv2.circle(display, (k_cx, k_cy), 4, (0, 0, 255), -1)  # rojo

        cv2.putText(
            display,
            f"sampled_frame={sampled_idx} | raw_frame={frame_idx}",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        writer.write(display)

        prev_gray = gray
        frame_idx += 1
        sampled_idx += 1

    cap.release()
    writer.release()

    print(f"Saved output video to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
