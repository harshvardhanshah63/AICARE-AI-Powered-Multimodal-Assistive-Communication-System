
import cv2
import mediapipe as mp
import numpy as np


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

print("Camera started")
print("Look at different directions.")
print("Press Q to quit.")


# ============================================================
# 3. IRIS LANDMARKS
# ============================================================

# MediaPipe iris landmarks

LEFT_IRIS = [468, 469, 470, 471, 472]

RIGHT_IRIS = [473, 474, 475, 476, 477]


# Eye corner / eyelid landmarks

LEFT_EYE_LEFT = 33
LEFT_EYE_RIGHT = 133
LEFT_EYE_TOP = 159
LEFT_EYE_BOTTOM = 145

RIGHT_EYE_LEFT = 362
RIGHT_EYE_RIGHT = 263
RIGHT_EYE_TOP = 386
RIGHT_EYE_BOTTOM = 374


# ============================================================
# 4. HELPER FUNCTION
# ============================================================

def get_point(landmark, width, height):

    x = int(landmark.x * width)
    y = int(landmark.y * height)

    return x, y


def iris_center(landmarks, indices, width, height):

    points = []

    for i in indices:

        x, y = get_point(
            landmarks[i],
            width,
            height
        )

        points.append((x, y))

    points = np.array(points)

    x = int(np.mean(points[:, 0]))
    y = int(np.mean(points[:, 1]))

    return x, y


# ============================================================
# 5. MAIN LOOP
# ============================================================

timestamp = 0


while True:

    ret, frame = cap.read()

    if not ret:
        print("Camera frame error")
        break


    # Mirror image
    frame = cv2.flip(frame, 1)


    height, width, _ = frame.shape


    # OpenCV BGR → RGB
    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )


    # MediaPipe image
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )


    timestamp += 1


    result = landmarker.detect_for_video(
        mp_image,
        timestamp
    )


    # ========================================================
    # 6. FACE DETECTED
    # ========================================================

    if result.face_landmarks:

        landmarks = result.face_landmarks[0]


        # ----------------------------------------------------
        # LEFT IRIS
        # ----------------------------------------------------

        left_iris_x, left_iris_y = iris_center(
            landmarks,
            LEFT_IRIS,
            width,
            height
        )


        # ----------------------------------------------------
        # RIGHT IRIS
        # ----------------------------------------------------

        right_iris_x, right_iris_y = iris_center(
            landmarks,
            RIGHT_IRIS,
            width,
            height
        )


        # ----------------------------------------------------
        # AVERAGE BOTH IRIS
        # ----------------------------------------------------

        iris_x = int(
            (left_iris_x + right_iris_x) / 2
        )

        iris_y = int(
            (left_iris_y + right_iris_y) / 2
        )


        # ====================================================
        # 7. DRAW IRIS
        # ====================================================

        cv2.circle(
            frame,
            (left_iris_x, left_iris_y),
            5,
            (0, 255, 0),
            -1
        )

        cv2.circle(
            frame,
            (right_iris_x, right_iris_y),
            5,
            (0, 255, 0),
            -1
        )


        # ====================================================
        # 8. DISPLAY IRIS POSITION
        # ====================================================

        cv2.putText(
            frame,
            f"Iris X: {iris_x}",
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Iris Y: {iris_y}",
            (30, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )


        # ====================================================
        # 9. NORMALIZE IRIS POSITION
        # ====================================================

        # Convert camera coordinates to 0-1

        gaze_x = iris_x / width
        gaze_y = iris_y / height


        # ====================================================
        # 10. CREATE GAZE POINT
        # ====================================================

        gaze_screen_x = int(
            gaze_x * width
        )

        gaze_screen_y = int(
            gaze_y * height
        )


        cv2.circle(
            frame,
            (gaze_screen_x, gaze_screen_y),
            12,
            (0, 0, 255),
            -1
        )


        cv2.putText(
            frame,
            "RED DOT = IRIS POSITION",
            (30, 115),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2
        )


    else:

        cv2.putText(
            frame,
            "Face not detected",
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )


    # ========================================================
    # 11. SHOW WINDOW
    # ========================================================

    cv2.imshow(
        "Iris Gaze Tracking",
        frame
    )


    # Q = quit

    if cv2.waitKey(1) & 0xFF == ord("q"):

        break


# ============================================================
# 12. CLEANUP
# ============================================================

cap.release()

cv2.destroyAllWindows()

landmarker.close()

print("Program stopped.")
