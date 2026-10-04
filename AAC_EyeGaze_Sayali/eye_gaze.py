
import cv2
import mediapipe as mp
import numpy as np
import time


# ============================================================
# 1. CHECK MODEL
# ============================================================

MODEL_PATH = "face_landmarker.task"


# ============================================================
# 2. CREATE MEDIAPIPE FACE LANDMARKER
# ============================================================

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
# 3. OPEN CAMERA
# ============================================================

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("ERROR: Camera could not be opened!")
    exit()

print("Camera started successfully.")
print("Press Q to quit.")


# ============================================================
# 4. MAIN LOOP
# ============================================================

frame_timestamp = 0


while True:

    ret, frame = cap.read()

    if not ret:
        print("ERROR: Could not read camera frame.")
        break


    # Mirror camera
    frame = cv2.flip(frame, 1)


    # Get image size
    height, width, _ = frame.shape


    # Convert OpenCV BGR → RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


    # Convert to MediaPipe Image
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )


    # Timestamp must increase
    frame_timestamp += 1


    # Detect face landmarks
    result = landmarker.detect_for_video(
        mp_image,
        frame_timestamp
    )


    # ========================================================
    # 5. IF FACE DETECTED
    # ========================================================

    if result.face_landmarks:

        landmarks = result.face_landmarks[0]


        # ----------------------------------------------------
        # LEFT EYE LANDMARKS
        # ----------------------------------------------------

        left_eye_indices = [
            33, 133, 159, 145, 160, 144
        ]


        # ----------------------------------------------------
        # RIGHT EYE LANDMARKS
        # ----------------------------------------------------

        right_eye_indices = [
            362, 263, 386, 374, 387, 373
        ]


        # ----------------------------------------------------
        # LEFT EYE CENTER
        # ----------------------------------------------------

        left_points = []

        for index in left_eye_indices:

            x = int(landmarks[index].x * width)
            y = int(landmarks[index].y * height)

            left_points.append((x, y))


        left_points = np.array(left_points)

        left_x = int(np.mean(left_points[:, 0]))
        left_y = int(np.mean(left_points[:, 1]))


        # ----------------------------------------------------
        # RIGHT EYE CENTER
        # ----------------------------------------------------

        right_points = []

        for index in right_eye_indices:

            x = int(landmarks[index].x * width)
            y = int(landmarks[index].y * height)

            right_points.append((x, y))


        right_points = np.array(right_points)

        right_x = int(np.mean(right_points[:, 0]))
        right_y = int(np.mean(right_points[:, 1]))


        # ----------------------------------------------------
        # DRAW EYE CENTERS
        # ----------------------------------------------------

        cv2.circle(
            frame,
            (left_x, left_y),
            5,
            (0, 255, 0),
            -1
        )


        cv2.circle(
            frame,
            (right_x, right_y),
            5,
            (0, 255, 0),
            -1
        )


        # ----------------------------------------------------
        # AVERAGE EYE POSITION
        # ----------------------------------------------------

        eye_x = int((left_x + right_x) / 2)
        eye_y = int((left_y + right_y) / 2)


        cv2.circle(
            frame,
            (eye_x, eye_y),
            8,
            (255, 0, 0),
            -1
        )


        # ====================================================
        # 6. SIMPLE GAZE DIRECTION
        # ====================================================

        screen_center_x = width // 2
        screen_center_y = height // 2


        # Horizontal direction

        if eye_x < screen_center_x - 80:

            horizontal = "LEFT"

        elif eye_x > screen_center_x + 80:

            horizontal = "RIGHT"

        else:

            horizontal = "CENTER"


        # Vertical direction

        if eye_y < screen_center_y - 80:

            vertical = "UP"

        elif eye_y > screen_center_y + 80:

            vertical = "DOWN"

        else:

            vertical = "CENTER"


        # ----------------------------------------------------
        # Select final direction
        # ----------------------------------------------------

        if horizontal != "CENTER":

            gaze = horizontal

        elif vertical != "CENTER":

            gaze = vertical

        else:

            gaze = "CENTER"


        # ====================================================
        # 7. DISPLAY GAZE
        # ====================================================

        cv2.putText(
            frame,
            "Gaze: " + gaze,
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )


        # Display eye coordinates

        cv2.putText(
            frame,
            f"Eye X: {eye_x}",
            (30, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )


        cv2.putText(
            frame,
            f"Eye Y: {eye_y}",
            (30, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )


    else:

        # No face

        cv2.putText(
            frame,
            "Face not detected",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )


    # ========================================================
    # 8. SHOW CAMERA
    # ========================================================

    cv2.imshow(
        "Eye Gaze Detection",
        frame
    )


    # Press Q to exit

    if cv2.waitKey(1) & 0xFF == ord("q"):

        break


# ============================================================
# 9. CLEANUP
# ============================================================

cap.release()

cv2.destroyAllWindows()

landmarker.close()

print("Eye gaze program stopped.")

