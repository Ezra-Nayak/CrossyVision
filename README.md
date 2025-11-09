# CrossyVision  Gesture Controller

> A real-time, gesture-controlled interface for games like Crossy Road, built with Python, MediaPipe, and a rich terminal-based UI.

![CrossyVision Screenshot](CrossyVision.gif)

### Check out a video of this project in action: https://youtu.be/JWum0gnZ4T4

---

## ✨ Features

-   **Intuitive Gesture Control**: Use your index and middle fingers on both hands to control UP, DOWN, LEFT, and RIGHT actions.
-   **Real-Time Hand Tracking**: Powered by Google's MediaPipe for high-performance, low-latency hand and finger detection.
-   **Advanced Terminal UI**: A beautiful and responsive dashboard built with `rich` that provides real-time visual feedback on finger positions and actions.
-   **Dynamic Calibration**: The system intelligently calibrates to your hand's position and finger lengths, requiring no manual setup.
-   **Face Masking**: An optional feature that automatically detects and masks your face in the camera feed.
-   **Multi-Threaded Performance**: The camera feed, MediaPipe processing, and face detection each run in separate threads to ensure a smooth, high-framerate experience.

## 🛠️ How It Works

The application uses a multi-threaded architecture to process video and gestures efficiently:

1.  **Webcam Input**: A dedicated thread reads frames from the webcam at the highest possible framerate.
2.  **MediaPipe Processing**: A second thread takes the latest frame and runs it through the MediaPipe Hands model to detect hand landmarks.
3.  **Face Masking (Optional)**: A third thread runs face detection to calculate a smoothed bounding box for applying a privacy mask.
4.  **Main Thread**: The main thread orchestrates everything:
    -   It analyzes the landmark data to interpret gestures.
    -   It calculates finger extension "progress" based on a dynamic threshold.
    -   It simulates key presses (`pyKey`) when a gesture is completed.
    -   It updates the `rich` terminal UI with the animated arrows and action history.
    -   It displays the processed video feed with visual feedback overlays.

## 🚀 Getting Started

### Prerequisites

-   Python 3.8+
-   A webcam connected to your computer.

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/CrossyVision.git
    cd CrossyVision
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    .\venv\Scripts\activate
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install opencv-python mediapipe rich pyKey
    ```

## 🎮 How to Use

1.  **Run the application:**
    ```bash
    python main.py
    ```
    The script will start, and you will see two windows: your webcam feed and the terminal UI.

2.  **Activation Gesture:**
    -   To activate the controls, hold **both hands** up in a **peace sign** (✌️) gesture.
    -   The header in the terminal UI will turn green, indicating that controls are active.

3.  **Control State:**
    -   Once active, curl your **ring and pinky fingers** down on both hands. This is the "ready" position for controlling the game.
    -   Your index and middle fingers should be extended.

4.  **Performing Actions:**
    -   To trigger an action, simply lower the corresponding extended finger past its activation threshold. The animated arrows in the UI will fill up as your finger approaches the threshold.

| Hand  | Finger | Action |
| :---- | :----- | :----- |
| **Left**  | Index  | **Right** (→) |
| **Left**  | Middle | **Left** (←)  |
| **Right** | Index  | **Up** (↑)    |
| **Right** | Middle | **Down** (↓)  |

5.  **Reset Gesture (Spacebar):**
    -   To press the `SPACEBAR` (useful for starting a game or jumping), make a **peace sign** with one hand and a **high-five** (all five fingers extended) with the other.

6.  **Deactivation:**
    -   To deactivate controls, simply move your hands out of the "control state" (e.g., open your hands or lower them). After a brief moment, the system will deactivate.

## ⚙️ Configuration

You can tweak the performance and feel of the controls by modifying the constants at the top of `main.py`:

-   `COOLDOWN_SECONDS`: Adjusts the time before the same key can be pressed again.
-   `FORCE_CALIBRATION_SECONDS`: How long a finger must be held down before its calibration baseline is reset.
-   `RELATIVE_THRESHOLD_PERCENT`: Changes the sensitivity of the finger press. Lower is more sensitive.
-   `ACTIVATION_CONFIDENCE_FRAMES`: The number of consecutive frames the activation gesture must be held.
-   `MASK_FACE`: Set to `False` to disable the face masking feature.

## 📄 License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## 🙏 Acknowledgements

-   **Google MediaPipe** for their incredible framework for hand and face landmark detection.
-   **Rich** by Will McGugan for making it possible to create such beautiful terminal interfaces.
