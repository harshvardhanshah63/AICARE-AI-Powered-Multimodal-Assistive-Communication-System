import cv2
import mediapipe as mp
import numpy as np
import time
import tkinter as tk

# ============================================================
# 1. SCREEN SIZE
# ============================================================

root = tk.Tk()
SCREEN_WIDTH = root.winfo_screenwidth()
SCREEN_HEIGHT = root.winfo_screenheight()
root.destroy()


# ============================================================
# 2. LOAD CALIBRATION
# ============================================================

data = np.load("gaze_calibration.npz")

coef_x = data["coef_x"]
coef_y = data["coef_y"]


# ============================================================
# 3. MEDIAPIPE
# ============================================================

MODEL_PATH = "face_landmarker.task"

BaseOptions = mp.tasks.BaseOptions
RunningMode = mp.tasks.vision.RunningMode
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions

options = FaceLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=MODEL_PATH
    ),
    running_mode=RunningMode.VIDEO,
    num_faces=1
)

landmarker = FaceLandmarker.create_from_options(options)


# ============================================================
# 4. CAMERA
# ============================================================

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Camera error")
    exit()


# ============================================================
# 5. LANDMARKS
# ============================================================

LEFT_EYE_LEFT = 33
LEFT_EYE_RIGHT = 133
LEFT_EYE_TOP = 159
LEFT_EYE_BOTTOM = 145

RIGHT_EYE_LEFT = 362
RIGHT_EYE_RIGHT = 263
RIGHT_EYE_TOP = 386
RIGHT_EYE_BOTTOM = 374

LEFT_IRIS = [468, 469, 470, 471, 472]
RIGHT_IRIS = [473, 474, 475, 476, 477]


# ============================================================
# 6. GET POINT
# ============================================================

def get_point(landmarks, index, width, height):

    p = landmarks[index]

    return np.array([
        p.x * width,
        p.y * height
    ])


# ============================================================
# 7. NORMALIZED GAZE
# ============================================================

def get_gaze(landmarks, width, height):

    # ---------------- LEFT EYE ----------------

    left_corner = get_point(
        landmarks,
        LEFT_EYE_LEFT,
        width,
        height
    )

    right_corner = get_point(
        landmarks,
        LEFT_EYE_RIGHT,
        width,
        height
    )

    top = get_point(
        landmarks,
        LEFT_EYE_TOP,
        width,
        height
    )

    bottom = get_point(
        landmarks,
        LEFT_EYE_BOTTOM,
        width,
        height
    )

    iris = np.mean(
        [
            get_point(landmarks, i, width, height)
            for i in LEFT_IRIS
        ],
        axis=0
    )

    width_eye = right_corner[0] - left_corner[0]
    height_eye = bottom[1] - top[1]

    if abs(width_eye) < 1 or abs(height_eye) < 1:
        return None

    lx = (iris[0] - left_corner[0]) / width_eye
    ly = (iris[1] - top[1]) / height_eye


    # ---------------- RIGHT EYE ----------------

    left_corner = get_point(
        landmarks,
        RIGHT_EYE_LEFT,
        width,
        height
    )

    right_corner = get_point(
        landmarks,
        RIGHT_EYE_RIGHT,
        width,
        height
    )

    top = get_point(
        landmarks,
        RIGHT_EYE_TOP,
        width,
        height
    )

    bottom = get_point(
        landmarks,
        RIGHT_EYE_BOTTOM,
        width,
        height
    )

    iris = np.mean(
        [
            get_point(landmarks, i, width, height)
            for i in RIGHT_IRIS
        ],
        axis=0
    )

    width_eye = right_corner[0] - left_corner[0]
    height_eye = bottom[1] - top[1]

    if abs(width_eye) < 1 or abs(height_eye) < 1:
        return None

    rx = (iris[0] - left_corner[0]) / width_eye
    ry = (iris[1] - top[1]) / height_eye


    # Average both eyes

    x = (lx + rx) / 2
    y = (ly + ry) / 2

    return x, y


# ============================================================
# 8. POLYNOMIAL FEATURES
# ============================================================

def features(x, y):

    return np.array([
        1,
        x,
        y,
        x*x,
        x*y,
        y*y,
        x*x*x,
        x*x*y,
        x*y*y,
        y*y*y
    ])


# ============================================================
# 9. CREATE TEST WINDOW
# ============================================================

window_name = "Gaze Accuracy Test"

cv2.namedWindow(
    window_name,
    cv2.WINDOW_NORMAL
)

cv2.setWindowProperty(
    window_name,
    cv2.WND_PROP_FULLSCREEN,
    cv2.WINDOW_FULLSCREEN
)


# ============================================================
# 10. SMOOTHING
# ============================================================

smooth_x = None
smooth_y = None

alpha = 0.20


# ============================================================
# 11. MAIN LOOP
# ============================================================

timestamp = 0

while True:

    ret, frame = cap.read()

    if not ret:
        continue

    frame = cv2.flip(frame, 1)

    h, w = frame.shape[:2]

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    timestamp += 33

    result = landmarker.detect_for_video(
        mp_image,
        timestamp
    )

    # Blank screen
    screen = np.zeros(
        (SCREEN_HEIGHT, SCREEN_WIDTH, 3),
        dtype=np.uint8
    )

    # --------------------------------------------------------
    # Detect face
    # --------------------------------------------------------

    if len(result.face_landmarks) > 0:

        landmarks = result.face_landmarks[0]

        gaze = get_gaze(
            landmarks,
            w,
            h
        )

        if gaze is not None:

            gx, gy = gaze

            # Polynomial mapping
            f = features(gx, gy)

            target_x = float(
                np.dot(coef_x, f)
            )

            target_y = float(
                np.dot(coef_y, f)
            )


            # Keep inside screen
            target_x = np.clip(
                target_x,
                0,
                SCREEN_WIDTH - 1
            )

            target_y = np.clip(
                target_y,
                0,
                SCREEN_HEIGHT - 1
            )


            # ------------------------------------------------
            # Smooth cursor
            # ------------------------------------------------

            if smooth_x is None:

                smooth_x = target_x
                smooth_y = target_y

            else:

                smooth_x = (
                    alpha * target_x
                    + (1 - alpha) * smooth_x
                )

                smooth_y = (
                    alpha * target_y
                    + (1 - alpha) * smooth_y
                )


            # ------------------------------------------------
            # Draw gaze cursor
            # ------------------------------------------------

            cv2.circle(
                screen,
                (
                    int(smooth_x),
                    int(smooth_y)
                ),
                25,
                (0, 255, 0),
                -1
            )

            cv2.circle(
                screen,
                (
                    int(smooth_x),
                    int(smooth_y)
                ),
                32,
                (255, 255, 255),
                3
            )


    # Instructions

    cv2.putText(
        screen,
        "Move your eyes and watch the GREEN dot",
        (40, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2
    )

    cv2.putText(
        screen,
        "Press ESC to exit",
        (40, 105),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (200, 200, 200),
        2
    )


    cv2.imshow(
        window_name,
        screen
    )

    key = cv2.waitKey(1) & 0xFF

    if key == 27:
        break


# ============================================================
# 12. CLEANUP
# ============================================================

cap.release()
cv2.destroyAllWindows()
landmarker.close()

print("Gaze test finished.")