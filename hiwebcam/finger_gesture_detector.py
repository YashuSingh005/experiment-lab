import cv2
import numpy as np
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# -----------------------------
# LOAD MODEL (auto required)
# -----------------------------
MODEL_PATH = "hand_landmarker.task"

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = vision.HandLandmarker
HandLandmarkerOptions = vision.HandLandmarkerOptions
VisionRunningMode = vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=1
)

detector = HandLandmarker.create_from_options(options)

# -----------------------------
# FINGER LOGIC
# -----------------------------
def get_fingers(lm):
    tips = [8, 12, 16, 20]
    dips = [6, 10, 14, 18]

    fingers = {}

    # thumb (simple rule)
    fingers["thumb"] = lm[4].x > lm[3].x

    names = ["index", "middle", "ring", "pinky"]

    for tip, dip, name in zip(tips, dips, names):
        fingers[name] = lm[tip].y < lm[dip].y

    return fingers


def get_gesture(fingers):
    active = [k for k, v in fingers.items() if v]

    if len(active) == 0:
        return "FIST ✊"
    if len(active) == 5:
        return "OPEN HAND 🖐️"
    if len(active) == 1:
        return active[0].upper()

    return "MIXED GESTURE"

# -----------------------------
# CAMERA
# -----------------------------
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    result = detector.detect(mp_image)

    gesture_text = "NO HAND"

    if result.hand_landmarks:
        lm = result.hand_landmarks[0][0]  # first hand

        fingers = get_fingers(lm)
        gesture_text = get_gesture(fingers)

        cv2.putText(frame, str(fingers), (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 0), 2)

    cv2.putText(frame, gesture_text, (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                (0, 0, 255), 3)

    cv2.imshow("Finger Detector (New MediaPipe API)", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()