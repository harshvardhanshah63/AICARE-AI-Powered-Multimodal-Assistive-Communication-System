import cv2
import mediapipe as mp
import numpy as np
import tkinter as tk
import pyttsx3
import time


# ============================================================
# SCREEN
# ============================================================

root = tk.Tk()

SCREEN_WIDTH = root.winfo_screenwidth()
SCREEN_HEIGHT = root.winfo_screenheight()

root.destroy()


# ============================================================
# CALIBRATION
# ============================================================

data = np.load("gaze_calibration.npz")

coef_x = data["coef_x"]
coef_y = data["coef_y"]


# ============================================================
# VOICE
# ============================================================

engine = pyttsx3.init()

engine.setProperty("rate", 150)


def speak(text):
    print("VOICE:", text)

    try:
        engine = pyttsx3.init('sapi5')

        voices = engine.getProperty('voices')

        if voices:
            engine.setProperty('voice', voices[0].id)

        engine.setProperty('rate', 150)
        engine.setProperty('volume', 1.0)

        engine.say(text)
        engine.runAndWait()

        engine.stop()

        print("VOICE FINISHED")

    except Exception as e:
        print("SPEECH ERROR:", e)


# ============================================================
# MEDIAPIPE
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


landmarker = FaceLandmarker.create_from_options(
    options
)


# ============================================================
# CAMERA
# ============================================================

cap = cv2.VideoCapture(
    0,
    cv2.CAP_DSHOW
)


if not cap.isOpened():

    print("Camera error")
    exit()


# ============================================================
# GAZE LANDMARKS
# ============================================================

LEFT_EYE_LEFT = 33
LEFT_EYE_RIGHT = 133

RIGHT_EYE_LEFT = 362
RIGHT_EYE_RIGHT = 263

LEFT_IRIS = [
    468,
    469,
    470,
    471,
    472
]

RIGHT_IRIS = [
    473,
    474,
    475,
    476,
    477
]


# ============================================================
# BLINK LANDMARKS
# ============================================================

L_TOP_1 = 159
L_BOTTOM_1 = 145

L_TOP_2 = 158
L_BOTTOM_2 = 153

R_TOP_1 = 386
R_BOTTOM_1 = 374

R_TOP_2 = 385
R_BOTTOM_2 = 380


# ============================================================
# GET POINT
# ============================================================

def point(landmarks, index, w, h):

    p = landmarks[index]

    return np.array([
        p.x * w,
        p.y * h
    ])


# ============================================================
# GAZE CALCULATION
# ============================================================

def get_gaze(landmarks, w, h):

    # --------------------------------------------------------
    # LEFT EYE
    # --------------------------------------------------------

    left_corner = point(
        landmarks,
        LEFT_EYE_LEFT,
        w,
        h
    )

    right_corner = point(
        landmarks,
        LEFT_EYE_RIGHT,
        w,
        h
    )

    left_iris = np.mean(
        [
            point(
                landmarks,
                i,
                w,
                h
            )
            for i in LEFT_IRIS
        ],
        axis=0
    )

    left_eye_width = (
        right_corner[0]
        -
        left_corner[0]
    )

    if left_eye_width < 1:

        return None


    left_x = (
        left_iris[0]
        -
        left_corner[0]
    ) / left_eye_width


    # --------------------------------------------------------
    # RIGHT EYE
    # --------------------------------------------------------

    left_corner = point(
        landmarks,
        RIGHT_EYE_LEFT,
        w,
        h
    )

    right_corner = point(
        landmarks,
        RIGHT_EYE_RIGHT,
        w,
        h
    )

    right_iris = np.mean(
        [
            point(
                landmarks,
                i,
                w,
                h
            )
            for i in RIGHT_IRIS
        ],
        axis=0
    )

    right_eye_width = (
        right_corner[0]
        -
        left_corner[0]
    )

    if right_eye_width < 1:

        return None


    right_x = (
        right_iris[0]
        -
        left_corner[0]
    ) / right_eye_width


    # --------------------------------------------------------
    # LEFT EYE VERTICAL
    # --------------------------------------------------------

    l_top = point(
        landmarks,
        159,
        w,
        h
    )

    l_bottom = point(
        landmarks,
        145,
        w,
        h
    )

    left_height = (
        l_bottom[1]
        -
        l_top[1]
    )

    if abs(left_height) < 1:

        return None


    left_y = (
        left_iris[1]
        -
        l_top[1]
    ) / left_height


    # --------------------------------------------------------
    # RIGHT EYE VERTICAL
    # --------------------------------------------------------

    r_top = point(
        landmarks,
        386,
        w,
        h
    )

    r_bottom = point(
        landmarks,
        374,
        w,
        h
    )

    right_height = (
        r_bottom[1]
        -
        r_top[1]
    )

    if abs(right_height) < 1:

        return None


    right_y = (
        right_iris[1]
        -
        r_top[1]
    ) / right_height


    # --------------------------------------------------------
    # AVERAGE BOTH EYES
    # --------------------------------------------------------

    gaze_x = (
        left_x
        +
        right_x
    ) / 2

    gaze_y = (
        left_y
        +
        right_y
    ) / 2


    return gaze_x, gaze_y


# ============================================================
# BLINK / EYE OPENING
# ============================================================

def get_eye_opening(landmarks, w, h):

    # --------------------------------------------------------
    # LEFT EYE
    # --------------------------------------------------------

    left_corner = point(
        landmarks,
        LEFT_EYE_LEFT,
        w,
        h
    )

    right_corner = point(
        landmarks,
        LEFT_EYE_RIGHT,
        w,
        h
    )

    top1 = point(
        landmarks,
        L_TOP_1,
        w,
        h
    )

    bottom1 = point(
        landmarks,
        L_BOTTOM_1,
        w,
        h
    )

    top2 = point(
        landmarks,
        L_TOP_2,
        w,
        h
    )

    bottom2 = point(
        landmarks,
        L_BOTTOM_2,
        w,
        h
    )


    eye_width = np.linalg.norm(
        right_corner - left_corner
    )

    if eye_width < 1:

        return None


    height1 = np.linalg.norm(
        top1 - bottom1
    )

    height2 = np.linalg.norm(
        top2 - bottom2
    )

    left_height = (
        height1 +
        height2
    ) / 2


    left_ratio = (
        left_height /
        eye_width
    )


    # --------------------------------------------------------
    # RIGHT EYE
    # --------------------------------------------------------

    left_corner = point(
        landmarks,
        RIGHT_EYE_LEFT,
        w,
        h
    )

    right_corner = point(
        landmarks,
        RIGHT_EYE_RIGHT,
        w,
        h
    )

    top1 = point(
        landmarks,
        R_TOP_1,
        w,
        h
    )

    bottom1 = point(
        landmarks,
        R_BOTTOM_1,
        w,
        h
    )

    top2 = point(
        landmarks,
        R_TOP_2,
        w,
        h
    )

    bottom2 = point(
        landmarks,
        R_BOTTOM_2,
        w,
        h
    )


    eye_width = np.linalg.norm(
        right_corner - left_corner
    )

    if eye_width < 1:

        return None


    height1 = np.linalg.norm(
        top1 - bottom1
    )

    height2 = np.linalg.norm(
        top2 - bottom2
    )

    right_height = (
        height1 +
        height2
    ) / 2


    right_ratio = (
        right_height /
        eye_width
    )


    # --------------------------------------------------------
    # AVERAGE BOTH EYES
    # --------------------------------------------------------

    return (
        left_ratio +
        right_ratio
    ) / 2


# ============================================================
# POLYNOMIAL FEATURES
# ============================================================

def features(x, y):

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
# MENUS
# ============================================================

MAIN_MENU = [

    ("FOOD", 0.25, 0.30),
    ("WATER", 0.50, 0.30),
    ("HELP", 0.75, 0.30),

    ("Music", 0.25, 0.65),
    ("YES", 0.50, 0.65),
    ("NO", 0.75, 0.65)

]


FOOD_MENU = [

    ("RICE", 0.25, 0.30),
    ("PIZZA", 0.50, 0.30),
    ("ROTI", 0.75, 0.30),

    ("FRUIT", 0.25, 0.65),
    ("MILK", 0.50, 0.65),
    ("NOODLES", 0.75, 0.65)

]


# ============================================================
# SENTENCES
# ============================================================

SENTENCES = {

    "WATER":
        "I want water.",

    "HELP":
        "I need help.",

    "MUSIC":
        "I want to listen Music.",

    "YES":
        "Yes.",

    "NO":
        "No.",

    "RICE":
        "I want rice.",

    "PIZZA":
        "I want pizza.",

    "ROTI":
        "I want roti.",

    "FRUIT":
        "I want fruit.",

    "MILK":
        "I want milk.",

    "NOODLES":
        "I want noodles."

}


# ============================================================
# CURRENT MENU
# ============================================================

current_menu = "MAIN"

current_options = MAIN_MENU


# ============================================================
# GAZE SMOOTHING
# ============================================================

smooth_x = None
smooth_y = None

SMOOTHING = 0.10


# ============================================================
# BLINK SETTINGS
# ============================================================

eyes_closed = False

blink_target = None

blink_start_time = 0

last_blink = 0


# IMPORTANT:
#
# Eye OPEN:
# approximately 0.20 - 0.35
#
# Eye CLOSED:
# approximately 0.05 - 0.15
#
# These can be adjusted later if necessary.

BLINK_THRESHOLD = 0.16

BLINK_END_THRESHOLD = 0.21

BLINK_MIN_TIME = 0.08

BLINK_MAX_TIME = 0.80

BLINK_COOLDOWN = 0.80


# ============================================================
# CURRENT HIGHLIGHT
# ============================================================

highlighted = None


# ============================================================
# BUTTON SIZE
# ============================================================

def button_rect(cx, cy):

    bw = int(
        SCREEN_WIDTH * 0.25
    )

    bh = int(
        SCREEN_HEIGHT * 0.22
    )

    center_x = int(
        cx * SCREEN_WIDTH
    )

    center_y = int(
        cy * SCREEN_HEIGHT
    )

    x1 = (
        center_x
        -
        bw // 2
    )

    y1 = (
        center_y
        -
        bh // 2
    )

    x2 = (
        center_x
        +
        bw // 2
    )

    y2 = (
        center_y
        +
        bh // 2
    )

    return x1, y1, x2, y2


# ============================================================
# FIND BUTTON
# ============================================================

def find_button(x, y):

    for name, cx, cy in current_options:

        x1, y1, x2, y2 = button_rect(
            cx,
            cy
        )

        if (
            x1 <= x <= x2
            and
            y1 <= y <= y2
        ):

            return name

    return None


# ============================================================
# DRAW BUTTONS
# ============================================================

def draw_buttons(
    screen,
    highlighted
):

    for name, cx, cy in current_options:

        x1, y1, x2, y2 = button_rect(
            cx,
            cy
        )


        # ----------------------------------------------------
        # HIGHLIGHTED
        # ----------------------------------------------------

        if name == highlighted:

            cv2.rectangle(
                screen,
                (x1, y1),
                (x2, y2),
                (0, 200, 0),
                -1
            )

            color = (
                0,
                0,
                0
            )


        # ----------------------------------------------------
        # NORMAL
        # ----------------------------------------------------

        else:

            cv2.rectangle(
                screen,
                (x1, y1),
                (x2, y2),
                (60, 60, 60),
                -1
            )

            cv2.rectangle(
                screen,
                (x1, y1),
                (x2, y2),
                (255, 255, 255),
                3
            )

            color = (
                255,
                255,
                255
            )


        # ----------------------------------------------------
        # TEXT
        # ----------------------------------------------------

        size = cv2.getTextSize(
            name,
            cv2.FONT_HERSHEY_SIMPLEX,
            1.3,
            3
        )[0]


        text_x = int(
            (
                x1
                +
                x2
                -
                size[0]
            ) / 2
        )


        text_y = int(
            (
                y1
                +
                y2
                +
                size[1]
            ) / 2
        )


        cv2.putText(
            screen,
            name,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.3,
            color,
            3
        )


# ============================================================
# OPEN FOOD MENU
# ============================================================

def open_food():

    global current_menu
    global current_options
    global highlighted

    current_menu = "FOOD"

    current_options = FOOD_MENU

    highlighted = None

    print()
    print("==============================")
    print("FOOD MENU OPENED")
    print("==============================")

    speak("I want Food")


# ============================================================
# SELECT ITEM
# ============================================================

def select_item(name):

    global current_menu
    global current_options
    global highlighted


    print()
    print("==============================")
    print("SELECTED:", name)
    print("==============================")


    # ========================================================
    # MAIN MENU
    # ========================================================

    if current_menu == "MAIN":

        # ----------------------------------------------------
        # FOOD
        # ----------------------------------------------------

        if name == "FOOD":

            open_food()

            return


        # ----------------------------------------------------
        # OTHER MAIN OPTIONS
        # ----------------------------------------------------

        sentence = SENTENCES.get(
            name
        )

        if sentence:

            speak(sentence)


    # ========================================================
    # FOOD MENU
    # ========================================================

    elif current_menu == "FOOD":

        sentence = SENTENCES.get(
            name
        )

        if sentence:

            speak(sentence)


        # ----------------------------------------------------
        # RETURN TO MAIN MENU
        # ----------------------------------------------------

        current_menu = "MAIN"

        current_options = MAIN_MENU

        highlighted = None

        print()
        print("RETURNED TO MAIN MENU")


# ============================================================
# WINDOW
# ============================================================

window_name = "AAC"


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
# MAIN LOOP
# ============================================================

timestamp = 0


while True:

    # ========================================================
    # CAMERA FRAME
    # ========================================================

    ret, frame = cap.read()


    if not ret:

        continue


    # Mirror image

    frame = cv2.flip(
        frame,
        1
    )


    h, w = frame.shape[:2]


    # ========================================================
    # MEDIAPIPE IMAGE
    # ========================================================

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )


    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )


    timestamp += 33


    # ========================================================
    # FACE LANDMARK DETECTION
    # ========================================================

    result = landmarker.detect_for_video(
        mp_image,
        timestamp
    )


    # ========================================================
    # CREATE SCREEN
    # ========================================================

    screen = np.zeros(
        (
            SCREEN_HEIGHT,
            SCREEN_WIDTH,
            3
        ),
        dtype=np.uint8
    )


    gaze_x = None
    gaze_y = None


    # ========================================================
    # FACE DETECTED
    # ========================================================

    if len(result.face_landmarks) > 0:

        landmarks = result.face_landmarks[0]


        # ====================================================
        # GAZE
        # ====================================================

        gaze = get_gaze(
            landmarks,
            w,
            h
        )


        if gaze is not None:

            gx, gy = gaze


            if (
                0 < gx < 1
                and
                0 < gy < 1
            ):

                # --------------------------------------------
                # POLYNOMIAL CALIBRATION
                # --------------------------------------------

                f = features(
                    gx,
                    gy
                )


                target_x = float(
                    np.dot(
                        coef_x,
                        f
                    )
                )


                target_y = float(
                    np.dot(
                        coef_y,
                        f
                    )
                )


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


                # --------------------------------------------
                # SMOOTHING
                # --------------------------------------------

                if smooth_x is None:

                    smooth_x = target_x

                    smooth_y = target_y


                else:

                    smooth_x = (
                        SMOOTHING
                        *
                        target_x
                        +
                        (
                            1 - SMOOTHING
                        )
                        *
                        smooth_x
                    )


                    smooth_y = (
                        SMOOTHING
                        *
                        target_y
                        +
                        (
                            1 - SMOOTHING
                        )
                        *
                        smooth_y
                    )


                gaze_x = smooth_x

                gaze_y = smooth_y


                # --------------------------------------------
                # HIGHLIGHT
                #
                # IMPORTANT:
                # During blink, don't change target.
                # --------------------------------------------

                if not eyes_closed:

                    highlighted = find_button(
                        gaze_x,
                        gaze_y
                    )


        # ====================================================
        # BLINK DETECTION
        # ====================================================

        eye_opening = get_eye_opening(
            landmarks,
            w,
            h
        )


        if eye_opening is not None:

            current_time = time.time()


            # =================================================
            # SHOW EYE VALUE
            # =================================================

            cv2.putText(
                screen,
                f"Eye: {eye_opening:.2f}",
                (
                    SCREEN_WIDTH - 180,
                    50
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )


            # =================================================
            # BLINK START
            # =================================================

            if eye_opening < BLINK_THRESHOLD:

                if not eyes_closed:

                    eyes_closed = True

                    blink_start_time = current_time

                    # ----------------------------------------
                    # LOCK CURRENT BUTTON
                    # ----------------------------------------

                    blink_target = highlighted


                    print()
                    print(
                        "========== BLINK START =========="
                    )

                    print(
                        "Eye opening:",
                        round(
                            eye_opening,
                            3
                        )
                    )

                    print(
                        "LOCKED TARGET:",
                        blink_target
                    )


            # =================================================
            # BLINK END
            # =================================================

            elif eye_opening > BLINK_END_THRESHOLD:

                if eyes_closed:

                    eyes_closed = False


                    blink_duration = (
                        current_time
                        -
                        blink_start_time
                    )


                    print(
                        "BLINK END | Eye:",
                        round(
                            eye_opening,
                            3
                        )
                    )


                    print(
                        "Blink duration:",
                        round(
                            blink_duration,
                            2
                        ),
                        "seconds"
                    )


                    # =================================================
                    # CHECK VALID BLINK
                    # =================================================

                    valid_blink = (
                        blink_target is not None
                        and
                        blink_duration >= BLINK_MIN_TIME
                        and
                        blink_duration <= BLINK_MAX_TIME
                        and
                        (
                            current_time
                            -
                            last_blink
                        ) > BLINK_COOLDOWN
                    )


                    if valid_blink:

                        last_blink = current_time


                        # --------------------------------------------
                        # SAVE TARGET
                        # --------------------------------------------

                        selected = blink_target


                        # Clear target BEFORE speaking
                        blink_target = None


                        print()
                        print(
                            "******** BLINK SELECT ********"
                        )

                        print(
                            "SELECTING:",
                            selected
                        )


                        # --------------------------------------------
                        # SELECT + SPEAK
                        # --------------------------------------------

                        select_item(
                            selected
                        )


                    else:

                        print(
                            "BLINK IGNORED"
                        )

                        print(
                            "Target:",
                            blink_target
                        )


                        blink_target = None


    # ========================================================
    # DRAW BUTTONS
    # ========================================================

    draw_buttons(
        screen,
        highlighted
    )


    # ========================================================
    # TITLE
    # ========================================================

    if current_menu == "MAIN":

        title = (
            "AAC - WHAT DO YOU NEED?"
        )

    else:

        title = "FOOD"


    cv2.putText(
        screen,
        title,
        (
            50,
            70
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (255, 255, 255),
        3
    )


    # ========================================================
    # INSTRUCTION
    # ========================================================

    cv2.putText(
        screen,
        "LOOK at an option + BLINK once",
        (
            50,
            SCREEN_HEIGHT - 40
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (220, 220, 220),
        2
    )


    # ========================================================
    # GAZE DOT
    # ========================================================

    if gaze_x is not None:

        cv2.circle(
            screen,
            (
                int(gaze_x),
                int(gaze_y)
            ),
            18,
            (0, 255, 255),
            -1
        )


    # ========================================================
    # BLINK STATUS
    # ========================================================

    if eyes_closed:

        cv2.putText(
            screen,
            "BLINK",
            (
                SCREEN_WIDTH - 180,
                100
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
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
    # KEYBOARD
    # ========================================================

    key = cv2.waitKey(1) & 0xFF


    # ESC = EXIT

    if key == 27:

        break


# ============================================================
# CLEANUP
# ============================================================

cap.release()

cv2.destroyAllWindows()

landmarker.close()

print("AAC stopped.")