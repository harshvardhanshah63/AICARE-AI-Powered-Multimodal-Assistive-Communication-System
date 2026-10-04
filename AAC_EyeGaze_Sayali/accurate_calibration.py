import cv2
import mediapipe as mp
import numpy as np
import time
import tkinter as tk

# ============================================================
# 1. GET SCREEN SIZE
# ============================================================

root = tk.Tk()
SCREEN_WIDTH = root.winfo_screenwidth()
SCREEN_HEIGHT = root.winfo_screenheight()
root.destroy()

print("Screen size:", SCREEN_WIDTH, "x", SCREEN_HEIGHT)


# ============================================================
# 2. MEDIAPIPE FACE LANDMARKER
# ============================================================

MODEL_PATH = "face_landmarker.task"

BaseOptions = mp.tasks.BaseOptions
VisionRunningMode = mp.tasks.vision.RunningMode
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.VIDEO,
    num_faces=1
)

landmarker = FaceLandmarker.create_from_options(options)


# ============================================================
# 3. CAMERA
# ============================================================

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("ERROR: Camera could not be opened.")
    exit()

print("Camera started.")


# ============================================================
# 4. EYE LANDMARKS
# ============================================================

# Left eye
LEFT_EYE_LEFT = 33
LEFT_EYE_RIGHT = 133
LEFT_EYE_TOP = 159
LEFT_EYE_BOTTOM = 145

# Right eye
RIGHT_EYE_LEFT = 362
RIGHT_EYE_RIGHT = 263
RIGHT_EYE_TOP = 386
RIGHT_EYE_BOTTOM = 374

# Iris
LEFT_IRIS = [468, 469, 470, 471, 472]
RIGHT_IRIS = [473, 474, 475, 476, 477]


# ============================================================
# 5. GET LANDMARK POSITION
# ============================================================

def get_point(landmarks, index, width, height):

    p = landmarks[index]

    return np.array([
        p.x * width,
        p.y * height
    ])


# ============================================================
# 6. CALCULATE NORMALIZED IRIS POSITION
# ============================================================

def get_normalized_gaze(landmarks, width, height):

    # ---------- LEFT EYE ----------

    left_corner = get_point(
        landmarks, LEFT_EYE_LEFT, width, height
    )

    right_corner = get_point(
        landmarks, LEFT_EYE_RIGHT, width, height
    )

    top = get_point(
        landmarks, LEFT_EYE_TOP, width, height
    )

    bottom = get_point(
        landmarks, LEFT_EYE_BOTTOM, width, height
    )

    left_iris_points = [
        get_point(landmarks, i, width, height)
        for i in LEFT_IRIS
    ]

    left_iris = np.mean(left_iris_points, axis=0)

    eye_width = right_corner[0] - left_corner[0]
    eye_height = bottom[1] - top[1]

    if abs(eye_width) < 1 or abs(eye_height) < 1:
        return None

    left_x = (
        left_iris[0] - left_corner[0]
    ) / eye_width

    left_y = (
        left_iris[1] - top[1]
    ) / eye_height


    # ---------- RIGHT EYE ----------

    left_corner = get_point(
        landmarks, RIGHT_EYE_LEFT, width, height
    )

    right_corner = get_point(
        landmarks, RIGHT_EYE_RIGHT, width, height
    )

    top = get_point(
        landmarks, RIGHT_EYE_TOP, width, height
    )

    bottom = get_point(
        landmarks, RIGHT_EYE_BOTTOM, width, height
    )

    right_iris_points = [
        get_point(landmarks, i, width, height)
        for i in RIGHT_IRIS
    ]

    right_iris = np.mean(right_iris_points, axis=0)

    eye_width = right_corner[0] - left_corner[0]
    eye_height = bottom[1] - top[1]

    if abs(eye_width) < 1 or abs(eye_height) < 1:
        return None

    right_x = (
        right_iris[0] - left_corner[0]
    ) / eye_width

    right_y = (
        right_iris[1] - top[1]
    ) / eye_height


    # Average both eyes

    gaze_x = (left_x + right_x) / 2
    gaze_y = (left_y + right_y) / 2

    return gaze_x, gaze_y


# ============================================================
# 7. 2D POLYNOMIAL FEATURES
# ============================================================

def make_features(x, y):

    return np.array([
        1,
        x,
        y,
        x * x,
        x * y,
        y * y,
        x * x * x,
        x * x * y,
        x * y * y,
        y * y * y
    ])


# ============================================================
# 8. CALIBRATION POINTS
# ============================================================

positions = [0.08, 0.36, 0.64, 0.92]

calibration_points = []

for y in positions:

    for x in positions:

        screen_x = int(x * SCREEN_WIDTH)
        screen_y = int(y * SCREEN_HEIGHT)

        calibration_points.append(
            (screen_x, screen_y)
        )


print("\nTotal calibration points:", len(calibration_points))


# ============================================================
# 9. FULLSCREEN CALIBRATION WINDOW
# ============================================================

window_name = "Eye Gaze Calibration"

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
# 10. CALIBRATION
# ============================================================

raw_data = []
screen_data = []

print("\nCalibration starting...")
print("Look directly at the red dot.")
print("Keep your head as still as possible.")
print("Blink normally.")
print()


for point_number, (target_x, target_y) in enumerate(
        calibration_points):

    print(
        f"Point {point_number + 1}/16:"
        f" ({target_x}, {target_y})"
    )

    # Give user time to look at new point
    start_time = time.time()

    while time.time() - start_time < 1.2:

        ret, frame = cap.read()

        if not ret:
            continue

        frame = cv2.flip(frame, 1)

        display = np.zeros(
            (SCREEN_HEIGHT, SCREEN_WIDTH, 3),
            dtype=np.uint8
        )

        # Draw target
        cv2.circle(
            display,
            (target_x, target_y),
            18,
            (0, 0, 255),
            -1
        )

        cv2.putText(
            display,
            f"Calibration {point_number + 1}/16",
            (40, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2
        )

        cv2.imshow(window_name, display)

        if cv2.waitKey(1) & 0xFF == 27:
            cap.release()
            cv2.destroyAllWindows()
            landmarker.close()
            exit()


    # --------------------------------------------------------
    # Collect samples
    # --------------------------------------------------------

    samples = []

    collection_start = time.time()

    while time.time() - collection_start < 1.0:

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

        timestamp = int(time.time() * 1000)

        result = landmarker.detect_for_video(
            mp_image,
            timestamp
        )

        if len(result.face_landmarks) == 0:
            continue

        landmarks = result.face_landmarks[0]

        gaze = get_normalized_gaze(
            landmarks,
            w,
            h
        )

        if gaze is None:
            continue

        gx, gy = gaze

        # Reject obviously bad measurements
        if (
            0.0 < gx < 1.0 and
            0.0 < gy < 1.0
        ):

            samples.append(
                [gx, gy]
            )


        # Display calibration point

        display = np.zeros(
            (SCREEN_HEIGHT, SCREEN_WIDTH, 3),
            dtype=np.uint8
        )

        cv2.circle(
            display,
            (target_x, target_y),
            18,
            (0, 0, 255),
            -1
        )

        cv2.putText(
            display,
            f"Point {point_number + 1}/16",
            (40, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2
        )

        cv2.imshow(
            window_name,
            display
        )

        if cv2.waitKey(1) & 0xFF == 27:
            cap.release()
            cv2.destroyAllWindows()
            landmarker.close()
            exit()


    # --------------------------------------------------------
    # Save median value
    # --------------------------------------------------------

    if len(samples) > 5:

        samples = np.array(samples)

        median_gaze = np.median(
            samples,
            axis=0
        )

        raw_data.append(
            median_gaze
        )

        screen_data.append(
            [target_x, target_y]
        )

        print(
            "   collected:",
            len(samples),
            "samples"
        )

        print(
            "   normalized:",
            median_gaze
        )

    else:

        print(
            "   WARNING: Not enough samples!"
        )

        # Still store approximate value
        if len(samples) > 0:

            samples = np.array(samples)

            raw_data.append(
                np.median(samples, axis=0)
            )

            screen_data.append(
                [target_x, target_y]
            )


# ============================================================
# 11. CONVERT DATA TO NUMPY
# ============================================================

raw_data = np.array(raw_data)
screen_data = np.array(screen_data)

print("\nCalibration samples:", len(raw_data))


# ============================================================
# 12. BUILD POLYNOMIAL MATRIX
# ============================================================

X = []

for x, y in raw_data:

    X.append(
        make_features(x, y)
    )

X = np.array(X)


# ============================================================
# 13. FIT SCREEN X AND SCREEN Y
# ============================================================

coef_x, _, _, _ = np.linalg.lstsq(
    X,
    screen_data[:, 0],
    rcond=None
)

coef_y, _, _, _ = np.linalg.lstsq(
    X,
    screen_data[:, 1],
    rcond=None
)


# ============================================================
# 14. SAVE CALIBRATION
# ============================================================

np.savez(
    "gaze_calibration.npz",

    coef_x=coef_x,
    coef_y=coef_y,

    screen_width=SCREEN_WIDTH,
    screen_height=SCREEN_HEIGHT
)


print("\n===================================")
print("CALIBRATION COMPLETE")
print("===================================")

print("Saved file:")
print("gaze_calibration.npz")

print("\nYou can now close the camera window.")

time.sleep(2)

cap.release()
cv2.destroyAllWindows()
landmarker.close()