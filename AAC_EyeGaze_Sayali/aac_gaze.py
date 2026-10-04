
import cv2
import mediapipe as mp
import numpy as np
import time


# ============================================================
# 1. LOAD CALIBRATION
# ============================================================

calibration = np.load("gaze_calibration.npz")

x_coefficients = calibration["x_coefficients"]
y_coefficients = calibration["y_coefficients"]


# ============================================================
# 2. MEDIAPIPE SETUP
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
# 3. CAMERA
# ============================================================

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():

    print("ERROR: Camera could not be opened.")
    exit()

print("Camera started.")
print("Look at the buttons.")
print("Press Q to quit.")


# ============================================================
# 4. IRIS LANDMARKS
# ============================================================

LEFT_IRIS = [468, 469, 470, 471, 472]
RIGHT_IRIS = [473, 474, 475, 476, 477]


def get_iris_center(landmarks, indices):

    points = []

    for index in indices:

        points.append([
            landmarks[index].x,
            landmarks[index].y
        ])

    points = np.array(points)

    return (
        np.mean(points[:, 0]),
        np.mean(points[:, 1])
    )


# ============================================================
# 5. AAC BUTTONS
# ============================================================

buttons = [

    {
        "name": "WATER",
        "x1": 100,
        "y1": 150,
        "x2": 550,
        "y2": 280
    },

    {
        "name": "FOOD",
        "x1": 730,
        "y1": 150,
        "x2": 1180,
        "y2": 280
    },

    {
        "name": "YES",
        "x1": 100,
        "y1": 330,
        "x2": 550,
        "y2": 460
    },

    {
        "name": "NO",
        "x1": 730,
        "y1": 330,
        "x2": 1180,
        "y2": 460
    },

    {
        "name": "HELP",
        "x1": 100,
        "y1": 510,
        "x2": 550,
        "y2": 640
    },

    {
        "name": "GAME",
        "x1": 730,
        "y1": 510,
        "x2": 1180,
        "y2": 640
    }
]


# ============================================================
# 6. WINDOW
# ============================================================

screen_width = 1280
screen_height = 720

window_name = "AAC Eye Gaze"

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
# 7. DWELL SETTINGS
# ============================================================

DWELL_TIME = 1.2

current_button = None
dwell_start = None


# ============================================================
# 8. MAIN LOOP
# ============================================================

timestamp = 0


while True:

    ret, frame = cap.read()

    if not ret:
        continue


    frame = cv2.flip(frame, 1)

    height, width, _ = frame.shape


    # ========================================================
    # MEDIAPIPE
    # ========================================================

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )


    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )


    timestamp += 1


    result = landmarker.detect_for_video(
        mp_image,
        timestamp
    )


    gaze_x = None
    gaze_y = None


    # ========================================================
    # GET IRIS
    # ========================================================

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


        # ====================================================
        # CALIBRATION MAPPING
        # ====================================================

        gaze_x = np.polyval(
            x_coefficients,
            iris_x
        )


        gaze_y = np.polyval(
            y_coefficients,
            iris_y
        )


        # Keep inside screen

        gaze_x = np.clip(
            gaze_x,
            0.0,
            1.0
        )


        gaze_y = np.clip(
            gaze_y,
            0.0,
            1.0
        )


        # Convert to screen pixels

        gaze_screen_x = int(
            gaze_x * screen_width
        )


        gaze_screen_y = int(
            gaze_y * screen_height
        )


    # ========================================================
    # DRAW AAC SCREEN
    # ========================================================

    screen = np.zeros(
        (screen_height, screen_width, 3),
        dtype=np.uint8
    )


    # Background

    screen[:] = (40, 40, 40)


    # Title

    cv2.putText(
        screen,
        "LOOK AT AN OPTION",
        (390, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (255, 255, 255),
        3
    )


    # ========================================================
    # FIND BUTTON UNDER GAZE
    # ========================================================

    selected_button = None


    if gaze_x is not None:

        for button in buttons:

            if (
                button["x1"] <= gaze_screen_x <= button["x2"]
                and
                button["y1"] <= gaze_screen_y <= button["y2"]
            ):

                selected_button = button["name"]

                break


    # ========================================================
    # DWELL LOGIC
    # ========================================================

    if selected_button != current_button:

        current_button = selected_button

        dwell_start = time.time()


    if selected_button is not None:

        elapsed = time.time() - dwell_start

        progress = min(
            elapsed / DWELL_TIME,
            1.0
        )


        # Selection

        if elapsed >= DWELL_TIME:

            print(
                "SELECTED:",
                selected_button
            )

            # Reset dwell timer

            dwell_start = time.time()


    else:

        progress = 0


    # ========================================================
    # DRAW BUTTONS
    # ========================================================

    for button in buttons:

        name = button["name"]


        if name == selected_button:

            thickness = 8

            border_color = (0, 255, 0)

        else:

            thickness = 3

            border_color = (255, 255, 255)


        cv2.rectangle(
            screen,
            (button["x1"], button["y1"]),
            (button["x2"], button["y2"]),
            border_color,
            thickness
        )


        # Text

        text_size = cv2.getTextSize(
            name,
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            3
        )[0]


        text_x = int(
            (
                button["x1"]
                + button["x2"]
                - text_size[0]
            ) / 2
        )


        text_y = int(
            (
                button["y1"]
                + button["y2"]
                + text_size[1]
            ) / 2
        )


        cv2.putText(
            screen,
            name,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (255, 255, 255),
            3
        )


    # ========================================================
    # DRAW DWELL PROGRESS
    # ========================================================

    if selected_button is not None:

        button = next(
            b for b in buttons
            if b["name"] == selected_button
        )


        bar_width = int(
            (button["x2"] - button["x1"])
            * progress
        )


        cv2.rectangle(
            screen,
            (
                button["x1"],
                button["y2"] - 12
            ),
            (
                button["x1"] + bar_width,
                button["y2"]
            ),
            (0, 255, 0),
            -1
        )


    # ========================================================
    # DRAW GAZE CURSOR
    # ========================================================

    if gaze_x is not None:

        cv2.circle(
            screen,
            (
                gaze_screen_x,
                gaze_screen_y
            ),
            15,
            (0, 0, 255),
            -1
        )


        cv2.circle(
            screen,
            (
                gaze_screen_x,
                gaze_screen_y
            ),
            25,
            (255, 255, 255),
            2
        )


    # ========================================================
    # SHOW
    # ========================================================

    cv2.imshow(
        window_name,
        screen
    )


    # ========================================================
    # EXIT
    # ========================================================

    if cv2.waitKey(1) & 0xFF == ord("q"):

        break


# ============================================================
# CLEANUP
# ============================================================

cap.release()

cv2.destroyAllWindows()

landmarker.close()

print("AAC system stopped.")

