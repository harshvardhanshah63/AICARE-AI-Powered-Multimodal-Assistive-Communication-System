
import cv2
import mediapipe as mp
import numpy as np
import time


# ============================================================
# 1. MEDIAPIPE SETUP
# ============================================================

MODEL_PATH = "face_landmarker.task"

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


options = FaceLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=MODEL_PATH
    ),
    running_mode=VisionRunningMode.VIDEO,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

landmarker = FaceLandmarker.create_from_options(options)


# ============================================================
# 2. CAMERA
# ============================================================

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("ERROR: Camera could not be opened")
    exit()


# ============================================================
# 3. IRIS LANDMARKS
# ============================================================

LEFT_IRIS = [468, 469, 470, 471, 472]
RIGHT_IRIS = [473, 474, 475, 476, 477]


# ============================================================
# 4. FUNCTION TO GET IRIS CENTER
# ============================================================

def get_iris_center(landmarks, indices):

    points = []

    for i in indices:

        x = landmarks[i].x
        y = landmarks[i].y

        points.append((x, y))

    points = np.array(points)

    x = np.mean(points[:, 0])
    y = np.mean(points[:, 1])

    return x, y


# ============================================================
# 5. CALIBRATION POINTS
# ============================================================

# These are relative screen positions

calibration_points = [

    (0.10, 0.10),   # top-left
    (0.50, 0.10),   # top-center
    (0.90, 0.10),   # top-right

    (0.10, 0.50),   # middle-left
    (0.50, 0.50),   # center
    (0.90, 0.50),   # middle-right

    (0.10, 0.90),   # bottom-left
    (0.50, 0.90),   # bottom-center
    (0.90, 0.90)    # bottom-right
]


# ============================================================
# 6. CREATE FULL SCREEN
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


screen_width = 1280
screen_height = 720


# ============================================================
# 7. DATA STORAGE
# ============================================================

iris_data = []
screen_data = []


# ============================================================
# 8. CALIBRATION PROCESS
# ============================================================

print()
print("======================================")
print("      EYE GAZE CALIBRATION")
print("======================================")
print()
print("Look directly at each red dot.")
print("Keep your head relatively still.")
print("The system will automatically collect data.")
print()


for point_number, (px, py) in enumerate(calibration_points):

    target_x = int(px * screen_width)
    target_y = int(py * screen_height)


    # --------------------------------------------------------
    # Wait before collecting
    # --------------------------------------------------------

    start_time = time.time()


    while time.time() - start_time < 1.0:

        frame = np.zeros(
            (screen_height, screen_width, 3),
            dtype=np.uint8
        )


        cv2.circle(
            frame,
            (target_x, target_y),
            25,
            (0, 0, 255),
            -1
        )


        cv2.putText(
            frame,
            f"Point {point_number + 1} / 9",
            (40, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2
        )


        cv2.putText(
            frame,
            "Look at the RED dot",
            (40, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )


        cv2.imshow(
            window_name,
            frame
        )


        if cv2.waitKey(1) & 0xFF == ord("q"):

            cap.release()
            cv2.destroyAllWindows()
            landmarker.close()
            exit()


    # --------------------------------------------------------
    # Collect samples
    # --------------------------------------------------------

    samples = []

    collection_start = time.time()


    while time.time() - collection_start < 1.5:

        ret, frame = cap.read()

        if not ret:
            continue


        frame = cv2.flip(frame, 1)


        height, width, _ = frame.shape


        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )


        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )


        timestamp = int(
            time.time() * 1000
        )


        result = landmarker.detect_for_video(
            mp_image,
            timestamp
        )


        if result.face_landmarks:

            landmarks = result.face_landmarks[0]


            left_x, left_y = get_iris_center(
                landmarks,
                LEFT_IRIS
            )


            right_x, right_y = get_iris_center(
                landmarks,
                RIGHT_IRIS
            )


            iris_x = (
                left_x + right_x
            ) / 2


            iris_y = (
                left_y + right_y
            ) / 2


            samples.append(
                [iris_x, iris_y]
            )


        # Show target

        display = np.zeros(
            (screen_height, screen_width, 3),
            dtype=np.uint8
        )


        cv2.circle(
            display,
            (target_x, target_y),
            25,
            (0, 0, 255),
            -1
        )


        cv2.putText(
            display,
            f"Point {point_number + 1} / 9",
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


        if cv2.waitKey(1) & 0xFF == ord("q"):

            cap.release()
            cv2.destroyAllWindows()
            landmarker.close()
            exit()


    # --------------------------------------------------------
    # Average samples
    # --------------------------------------------------------

    if len(samples) > 0:

        average_iris = np.mean(
            samples,
            axis=0
        )


        iris_data.append(
            average_iris
        )


        screen_data.append(
            [px, py]
        )


        print(
            f"Point {point_number + 1}: "
            f"Iris = {average_iris}"
        )


# ============================================================
# 9. CONVERT DATA
# ============================================================

iris_data = np.array(
    iris_data
)

screen_data = np.array(
    screen_data
)


# ============================================================
# 10. CALCULATE LINEAR MAPPING
# ============================================================

# X mapping

x_coefficients = np.polyfit(
    iris_data[:, 0],
    screen_data[:, 0],
    2
)


# Y mapping

y_coefficients = np.polyfit(
    iris_data[:, 1],
    screen_data[:, 1],
    2
)


# ============================================================
# 11. SAVE CALIBRATION
# ============================================================

np.savez(
    "gaze_calibration.npz",
    x_coefficients=x_coefficients,
    y_coefficients=y_coefficients
)


# ============================================================
# 12. FINISHED
# ============================================================

print()
print("======================================")
print("       CALIBRATION COMPLETE")
print("======================================")
print()
print("Saved as:")
print("gaze_calibration.npz")
print()
print("You can now use this calibration")
print("for accurate gaze-based selection.")
print()


# Show completion screen

frame = np.zeros(
    (screen_height, screen_width, 3),
    dtype=np.uint8
)


cv2.putText(
    frame,
    "CALIBRATION COMPLETE!",
    (350, 300),
    cv2.FONT_HERSHEY_SIMPLEX,
    1.5,
    (0, 255, 0),
    3
)


cv2.putText(
    frame,
    "Press Q to continue",
    (430, 360),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.8,
    (255, 255, 255),
    2
)


while True:

    cv2.imshow(
        window_name,
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):

        break


# ============================================================
# 13. CLEANUP
# ============================================================

cap.release()
cv2.destroyAllWindows()
landmarker.close()

