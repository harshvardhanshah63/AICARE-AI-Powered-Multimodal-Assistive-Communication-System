# AAC EyeGaze Sayali
### Eye-Gaze Based Augmentative and Assistive Communication

**AAC EyeGaze Sayali** is an eye-gaze-based Augmentative and Alternative Communication (AAC) prototype included in **AICARE – AI-Powered Multimodal Assistive Communication System**. It uses a webcam and facial/iris landmarks to estimate where a user is looking, then lets the user select on-screen communication options using gaze and a blink-based selection mechanism. Selected phrases are spoken using text-to-speech.


## Problem Statement

People who cannot reliably communicate through speech or conventional physical input may find it difficult to express everyday needs. AAC EyeGaze Sayali explores webcam-based eye-gaze interaction as an alternative input method, helping a user select simple, pre-defined messages without requiring a mouse or keyboard for each selection.

## Features

- **Webcam-based face and eye tracking:** Uses MediaPipe Face Landmarker to obtain facial and iris landmarks.
- **Gaze estimation:** Calculates normalized gaze features from both eyes and maps them to screen coordinates using calibration coefficients.
- **Personalized calibration:** Calibration scripts collect gaze samples against known screen targets and save the fitted mapping to an `.npz` file.
- **Gaze-based menu selection:** The main AAC interface highlights the on-screen option that corresponds to the estimated gaze position.
- **Blink-based activation:** The main interface uses eye-opening measurements and blink timing thresholds to activate a highlighted option.
- **Text-to-speech output:** Uses `pyttsx3` with the Windows SAPI5 speech engine to vocalize supported phrases.
- **Predefined communication choices:** Includes common messages such as FOOD, WATER, HELP, MUSIC, YES and NO, plus a food submenu.
- **Supporting utilities:** Includes camera checks, basic gaze visualization, calibration, and gaze accuracy test scripts.

## How It Works

1. The webcam captures video frames.
2. MediaPipe Face Landmarker detects facial landmarks, including eye and iris landmarks.
3. The program estimates horizontal and vertical gaze features from the eye and iris positions.
4. The calibration coefficients map those features to screen coordinates.
5. The AAC interface highlights the menu item under the estimated gaze position.
6. In the main application, a detected blink that meets the configured duration and cooldown thresholds selects the highlighted item.
7. The selected phrase is spoken through the system's text-to-speech engine.

## Technology Stack

| Component | Technology |
|---|---|
| Language | Python |
| Webcam and image processing | OpenCV (`cv2`) |
| Face and iris landmarks | MediaPipe Tasks – Face Landmarker |
| Numerical processing and calibration | NumPy |
| Screen dimensions / window support | Tkinter (Python standard library) |
| Text-to-speech | `pyttsx3` / Windows SAPI5 |
| Calibration data | NumPy `.npz` archive |
| Face landmark model | `face_landmarker.task` |

## Repository Structure

```text
AAC_EyeGaze_Sayali/
├── aac_final.py                 # Main AAC interface with gaze, blink selection and speech
├── aac_gaze.py                  # Gaze cursor and menu interaction prototype
├── accurate_calibration.py      # Screen-target calibration; saves coef_x / coef_y
├── accurate_gaze_test.py        # Tests gaze mapping using saved calibration
├── calibration.py               # Earlier iris-based calibration prototype
├── camera_test.py               # Checks available camera indices
├── eye_gaze.py                  # Basic gaze direction visualization
├── iris_gaze.py                 # Iris-centre visualization prototype
├── face_landmarker.task         # MediaPipe face landmark model used by scripts
├── face_landmarker (1).task     # Duplicate model asset in repository
├── gaze_calibration.npz         # Saved calibration data (may be user/device specific)
└── models/
    ├── camera_test.py           # Additional camera test utility
    └── face_landmarker.task     # Model asset copy
```

## Requirements

- Windows is the expected environment for the current camera setup (`cv2.CAP_DSHOW`) and SAPI5 voice output.
- Python 3 with `pip`.
- A working webcam.
- A display for the calibration targets and AAC interface.
- MediaPipe-compatible runtime and the required Python packages.

### Python Packages

Install the packages used by the scripts:

```bash
python -m pip install opencv-python mediapipe numpy pyttsx3
```

`tkinter` is part of many standard Python distributions. On Windows, install Python with the Tcl/Tk support option enabled if Tkinter is unavailable.

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/harshvardhanshah63/AICARE-AI-Powered-Multimodal-Assistive-Communication-System.git
```

### 2. Open the project folder

```bash
cd AICARE-AI-Powered-Multimodal-Assistive-Communication-System/AAC_EyeGaze_Sayali
```

### 3. (Recommended) Create and activate a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, you can use Command Prompt activation instead:

```bat
.venv\Scripts\activate.bat
```

### 4. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install opencv-python mediapipe numpy pyttsx3
```

### 5. Check the webcam

```bash
python camera_test.py
```

The utility checks camera indices and displays a frame when a camera is found. Close the preview window or wait for it to finish.

## Calibration

The main `aac_final.py` application loads `gaze_calibration.npz` and expects the arrays `coef_x` and `coef_y`. The matching calibration workflow in the repository is `accurate_calibration.py`, which fits and saves these coefficients.

Run calibration from inside the `AAC_EyeGaze_Sayali` directory so the relative model and output paths resolve:

```bash
python accurate_calibration.py
```

- Sit in a stable, comfortable position facing the webcam.
- Keep the face visible and lighting reasonably even.
- Follow the calibration targets shown on the screen and look at each target as instructed by the program.
- Complete the calibration without moving the camera or changing the screen setup.
- Confirm that `gaze_calibration.npz` has been created or updated in this project directory.

The calibration file is tied to the user's position, camera placement, and display setup. Recalibrate if these change or gaze selection is inaccurate.

### Test the calibration

```bash
python accurate_gaze_test.py
```

This opens the gaze test visualization using the saved coefficients. Review whether the displayed gaze point follows the user's gaze across the screen.

## Run the AAC Application

After the model file and compatible calibration data are present in the project directory:

```bash
python aac_final.py
```

The current main interface contains a primary menu with **FOOD, WATER, HELP, MUSIC, YES, and NO**. FOOD opens a submenu with **RICE, PIZZA, ROTI, FRUIT, MILK, and NOODLES**. Selecting a supported item triggers its associated spoken phrase.

Keep the webcam connected and ensure the application can access it. Run calibration first if the saved calibration does not match the current user and setup.

## Other Utilities

Run these from the `AAC_EyeGaze_Sayali` directory:

| Script | Purpose | Command |
|---|---|---|
| `camera_test.py` | Checks camera availability | `python camera_test.py` |
| `eye_gaze.py` | Basic gaze-direction visualization | `python eye_gaze.py` |
| `iris_gaze.py` | Iris-centre visualization | `python iris_gaze.py` |
| `calibration.py` | Earlier iris calibration implementation | `python calibration.py` |
| `accurate_calibration.py` | Calibrates screen mapping for the main AAC app | `python accurate_calibration.py` |
| `accurate_gaze_test.py` | Tests calibrated gaze tracking | `python accurate_gaze_test.py` |
| `aac_gaze.py` | Gaze cursor / menu interaction prototype | `python aac_gaze.py` |
| `aac_final.py` | Main AAC application | `python aac_final.py` |

Some scripts are development and testing alternatives rather than steps that must all be run. Use `accurate_calibration.py` followed by `aac_final.py` for the calibration-to-main-app workflow.

## Troubleshooting

**Camera could not be opened**
- Close other applications that may be using the webcam.
- Check Windows camera privacy permissions.
- Try another camera index if the default camera is not index `0`. Update the script only if necessary.

**Face landmark model not found**
- Run commands from the `AAC_EyeGaze_Sayali` directory.
- Ensure `face_landmarker.task` is present beside the script. The repository also contains a copy under `models/`, but scripts using the relative path `face_landmarker.task` expect it in the current working directory.

**Calibration file missing or incompatible**
- Run `python accurate_calibration.py` from the project directory.
- The main app expects `coef_x` and `coef_y` keys. Calibration files produced by other prototype scripts may use different key names and may not be compatible with `aac_final.py`.

**Gaze point is inaccurate**
- Re-run calibration with the camera and display in their intended positions.
- Maintain consistent seating distance, face visibility, and lighting.
- Avoid changing camera angle after calibration.

**No speech output**
- Confirm that Windows has an available speech voice and audio output.
- Check that `pyttsx3` is installed and that the system's SAPI5 voice service is available.

**MediaPipe installation or runtime error**
- Use a supported Python version for the installed MediaPipe release.
- Install dependencies in the active virtual environment and check the full error message before changing package versions.

## Limitations

- The interface currently uses a finite set of predefined communication phrases.
- Tracking quality can be affected by lighting, camera position, head movement, glasses, and individual eye characteristics.
- Calibration is required for reliable screen mapping and may need to be repeated for a different setup.
- The implementation is a prototype and has not been established as a clinically validated assistive or medical device.
- The saved calibration data may be specific to an individual user and computer configuration.

## Future Scope

- Customizable communication boards and user-defined phrases.
- Additional languages and configurable speech voices.
- More robust calibration and gaze stabilization.
- Configurable dwell selection and alternative accessible selection methods.
- Integration with other input modalities as part of the wider AICARE system.
- Usability testing with intended users and accessibility specialists.

## Responsible Use

This project is an assistive-technology prototype intended for experimentation and development. It should not be treated as a replacement for clinically prescribed AAC solutions or professional assessment. Provide users with a reliable alternative communication method during testing.

## Contributing

Contributions that improve accessibility, usability, documentation, calibration, and reliability are welcome.

1. Fork the repository.
2. Create a feature branch.
3. Make and test your changes.
4. Commit with a clear message.
5. Open a pull request describing the change.

## Project Link

[AICARE – AI-Powered Multimodal Assistive Communication System](https://github.com/harshvardhanshah63/AICARE-AI-Powered-Multimodal-Assistive-Communication-System/tree/main/AAC_EyeGaze_Sayali)

## License

No license is currently specified for this project. Add a `LICENSE` file before redistributing or reusing the code under an open-source license.
