import cv2
import mediapipe as mp
import time
from pyKey import press
from collections import deque
import threading
from threading import Thread
import numpy as np

# --- Rich TUI Imports ---
from collections import deque
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import Progress, BarColumn, TextColumn
from rich.columns import Columns
from rich.text import Text
import logging
import time # Required for custom log timestamp

# --- Rich TUI Setup ---
console = Console()
log_deque = deque(maxlen=8) # Store the last 8 log messages

# Custom handler to append logs to our deque
class DequeHandler(logging.Handler):
    def __init__(self, deque_instance):
        super().__init__()
        self.deque_instance = deque_instance

    def emit(self, record):
        # Format the message with a timestamp
        log_entry = f"[{time.strftime('%X')}] {record.getMessage()}"
        self.deque_instance.append(log_entry)

# Configure the specific logger we will use to send messages to the TUI
log = logging.getLogger("tui_logger")
log.setLevel(logging.INFO)
log.addHandler(DequeHandler(log_deque))
log.propagate = False # Prevent TUI messages from appearing on stderr

# Configure basicConfig for any other module's logging (like MediaPipe's warnings)
# This will print them to stderr without interfering with the Rich Live display.
logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(name)s: %(message)s")


class WebcamVideoStream:
    """
    A class to read frames from a webcam in a dedicated thread.
    This prevents the main processing loop from being blocked by camera I/O.
    """

    def __init__(self, src=0):
        # Initialize the video camera stream and read the first frame
        self.stream = cv2.VideoCapture(src, cv2.CAP_DSHOW)
        self.stream.set(cv2.CAP_PROP_FPS, 60)
        (self.grabbed, self.frame) = self.stream.read()

        # A flag to indicate if the thread should be stopped
        self.stopped = False

    def start(self):
        # Start the thread to read frames from the video stream
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        # Keep looping infinitely until the thread is stopped
        while True:
            # If the thread indicator variable is set, stop the thread
            if self.stopped:
                return

            # Otherwise, read the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        # Return the frame most recently read
        return self.frame

    def stop(self):
        # Indicate that the thread should be stopped
        self.stopped = True


class MediaPipeProcessor:
    """
    A class to run MediaPipe hand processing in a dedicated thread.
    It consumes frames from a WebcamVideoStream and produces annotated frames.
    """

    def __init__(self, stream, hands_instance):
        self.stream = stream
        self.hands = hands_instance
        self.stopped = False

        # These will hold the latest processed results
        self.results = None
        self.output_frame = None

    def start(self):
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        while True:
            if self.stopped:
                return

            # Get the latest frame from the camera thread
            frame = self.stream.read()
            if frame is None:
                continue

            # Process the frame
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.results = self.hands.process(rgb_frame)

            # Store the frame that was used for this result
            self.output_frame = frame.copy()

    def read(self):
        # Return the latest processed frame and its results
        return self.output_frame, self.results

    def stop(self):
        self.stopped = True


class FaceMasker:
    """
    Runs face detection and smoothing in a dedicated thread at a fixed rate (60 UPS)
    to provide a stable, jitter-free overlay, independent of the main loop's frame rate.
    """

    def __init__(self, stream, mask_image):
        self.stream = stream  # The WebcamVideoStream instance
        self.mask_image = mask_image
        self.stopped = False

        # MediaPipe Face Detection setup
        self.face_detection = mp.solutions.face_detection.FaceDetection(model_selection=0,
                                                                        min_detection_confidence=0.5)

        # State variables for smoothing and storing the final bbox
        self.smoothed_bbox = None
        self.latest_bbox_for_overlay = None  # This will be read by the main thread

        # Threading lock to prevent race conditions when accessing the bbox
        self.lock = threading.Lock()

    def start(self):
        Thread(target=self.update, args=(), daemon=True).start()
        return self

    def update(self):
        """
        This loop runs in its own thread, constantly detecting the face
        and updating the smoothed bounding box at a fixed rate.
        """
        updates_per_second = 60
        sleep_duration = 1.0 / updates_per_second

        while not self.stopped:
            start_time = time.time()

            frame = self.stream.read()
            if frame is None:
                time.sleep(sleep_duration)
                continue

            # Process for faces
            face_frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_results = self.face_detection.process(face_frame_rgb)
            ih, iw, _ = frame.shape

            current_bbox = None
            if face_results.detections:
                detection = face_results.detections[0]
                bboxC = detection.location_data.relative_bounding_box
                current_bbox = (int(bboxC.xmin * iw), int(bboxC.ymin * ih),
                                int(bboxC.width * iw), int(bboxC.height * ih))

            # --- Bounding Box Smoothing ---
            if current_bbox:
                if self.smoothed_bbox is None:
                    self.smoothed_bbox = current_bbox
                else:
                    x1, y1, w1, h1 = self.smoothed_bbox
                    x2, y2, w2, h2 = current_bbox
                    # A more aggressive smoothing factor works better in a fixed-rate loop
                    smoothing = 0.1
                    new_x = int(x1 * (1 - smoothing) + x2 * smoothing)
                    new_y = int(y1 * (1 - smoothing) + y2 * smoothing)
                    new_w = int(w1 * (1 - smoothing) + w2 * smoothing)
                    new_h = int(h1 * (1 - smoothing) + h2 * smoothing)
                    self.smoothed_bbox = (new_x, new_y, new_w, new_h)

            # Safely update the bbox that the main thread will read
            with self.lock:
                self.latest_bbox_for_overlay = self.smoothed_bbox

            # Maintain the 60 UPS rate
            elapsed_time = time.time() - start_time
            time.sleep(max(0, sleep_duration - elapsed_time))

    def apply_mask(self, frame):
        """
        Called from the main loop to draw the mask on the frame
        using the latest calculated bounding box.
        """
        local_bbox = None
        with self.lock:
            if self.latest_bbox_for_overlay:
                local_bbox = self.latest_bbox_for_overlay

        if local_bbox:
            ih, iw, _ = frame.shape
            x, y, w, h = local_bbox

            # --- Expanded Bounding Box for Full Coverage ---
            # Shift up significantly and expand to cover hair/forehead/chin
            x_new = x - int(w * 0.25)
            y_new = y - int(h * 0.55)  # Shift up by ~55% of face height
            w_new = int(w * 1.5)  # Make mask 50% wider than face
            h_new = int(h * 1.8)  # Make mask 80% taller than face

            y1, y2 = max(0, y_new), min(ih, y_new + h_new)
            x1, x2 = max(0, x_new), min(iw, x_new + w_new)

            roi_h, roi_w = y2 - y1, x2 - x1

            if roi_h > 0 and roi_w > 0:
                mask_resized = cv2.resize(self.mask_image, (roi_w, roi_h))

                # --- Handle Grayscale Masks ---
                # If the mask image is grayscale (shape has only 2 dimensions),
                # convert it to 3-channel BGR to make it compatible with the frame.
                if len(mask_resized.shape) == 2:
                    mask_resized = cv2.cvtColor(mask_resized, cv2.COLOR_GRAY2BGR)

                # --- Blending/Overlay Logic ---
                # Opaque image logic (3 channels)
                if mask_resized.shape[2] != 4:
                    frame[y1:y2, x1:x2] = mask_resized
                # Transparent image logic (4 channels)
                else:
                    roi = frame[y1:y2, x1:x2]
                    mask_bgr = mask_resized[:, :, :3]
                    alpha = (mask_resized[:, :, 3] / 255.0)[:, :, np.newaxis]
                    blended = (roi.astype(float) * (1 - alpha) + mask_bgr.astype(float) * alpha)
                    frame[y1:y2, x1:x2] = blended.astype(np.uint8)
        return frame

    def stop(self):
        self.stopped = True


# --- Constants and Initialization ---

# Control parameters
COOLDOWN_SECONDS = 0.08  # Cooldown for individual finger presses.
FORCE_CALIBRATION_SECONDS = 2.0  # If finger is held down for this long, force recalibration.
CALIBRATION_FRAMES = 1200  # Number of frames to average for the trigger line (~1 second).
RELATIVE_THRESHOLD_PERCENT = 0.85  # Trigger line is 40% of the way down from the MCP. Tune this for comfort.

# State stabilization parameters
ACTIVATION_CONFIDENCE_FRAMES = 10  # Require 5 consecutive frames of correct gesture to activate.
DEACTIVATION_CONFIDENCE_FRAMES = 25  # Require 5 consecutive frames of broken gesture to deactivate.

# MediaPipe Hands setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    model_complexity=1,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_drawing = mp.solutions.drawing_utils

# --- Face Masking Setup ---
MASK_FACE = True  # Set to False to disable face masking

# Load the mask image
if MASK_FACE:
    try:
        mask_image = cv2.imread("mask.png", cv2.IMREAD_UNCHANGED)
        if mask_image is None:
            log.warning("'mask.png' not found or could not be read. Face masking will be disabled.")
            MASK_FACE = False
    except Exception as e:
        log.error(f"Error loading 'mask.png': {e}. Face masking will be disabled.")
        MASK_FACE = False

# --- Threaded Setup ---
log.info("Starting threaded services...")
# Start the camera stream thread
vs = WebcamVideoStream(src=0).start()
# Start the MediaPipe processing thread
processor = MediaPipeProcessor(stream=vs, hands_instance=hands).start()
# Start the Face Masking thread if enabled
face_masker = None
if MASK_FACE:
    face_masker = FaceMasker(stream=vs, mask_image=mask_image).start()

time.sleep(2.0)  # Allow services to warm up

# --- State Variables ---

# FPS calculation
prev_frame_time = 0
new_frame_time = 0
fps = 0

# Dictionaries to hold state information
controls_active = False
last_key_press_time = {}
# For dynamic thresholds
mcp_history = {}  # Stores recent MCP joint Y-positions for averaging
finger_lengths = {}  # Stores recent finger lengths for averaging
trigger_thresholds = {}  # Stores the calculated trigger line Y-position for each finger
previous_finger_is_above = {}  # Stores the last known position relative to the trigger line
time_finger_went_below = {}  # For forced recalibration

# For state stabilization (debouncing)
activation_counter = 0
deactivation_counter = 0

# --- TUI Elements ---
# Create progress bars for each finger
progress_bars = {
    "LEFT_INDEX": Progress(TextColumn("{task.description}"), BarColumn(bar_width=None),
                           TextColumn("{task.percentage:>3.0f}%"), console=console),
    "LEFT_MIDDLE": Progress(TextColumn("{task.description}"), BarColumn(bar_width=None),
                            TextColumn("{task.percentage:>3.0f}%"), console=console),
    "RIGHT_INDEX": Progress(TextColumn("{task.description}"), BarColumn(bar_width=None),
                            TextColumn("{task.percentage:>3.0f}%"), console=console),
    "RIGHT_MIDDLE": Progress(TextColumn("{task.description}"), BarColumn(bar_width=None),
                             TextColumn("{task.percentage:>3.0f}%"), console=console),
}
# Add tasks to the progress bars
progress_tasks = {
    "LEFT_INDEX": progress_bars["LEFT_INDEX"].add_task("[cyan]UP (Index)  ", total=100),
    "LEFT_MIDDLE": progress_bars["LEFT_MIDDLE"].add_task("[cyan]DOWN (Middle)", total=100),
    "RIGHT_INDEX": progress_bars["RIGHT_INDEX"].add_task("[cyan]LEFT (Index) ", total=100),
    "RIGHT_MIDDLE": progress_bars["RIGHT_MIDDLE"].add_task("[cyan]RIGHT (Middle)", total=100),
}

log.info("CrossyVision Controller Initialized. Press 'ESC' to quit.")


# --- Helper Functions ---

def is_finger_extended(landmarks, tip_landmark, pip_landmark):
    """Checks if a specific finger is extended."""
    return landmarks[tip_landmark].y < landmarks[pip_landmark].y


def is_peace_sign(hand_landmarks):
    if not hand_landmarks:
        return False
    landmarks = hand_landmarks.landmark
    index_extended = is_finger_extended(landmarks, mp_hands.HandLandmark.INDEX_FINGER_TIP,
                                        mp_hands.HandLandmark.INDEX_FINGER_PIP)
    middle_extended = is_finger_extended(landmarks, mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
                                         mp_hands.HandLandmark.MIDDLE_FINGER_PIP)
    ring_curled = not is_finger_extended(landmarks, mp_hands.HandLandmark.RING_FINGER_TIP,
                                         mp_hands.HandLandmark.RING_FINGER_PIP)
    pinky_curled = not is_finger_extended(landmarks, mp_hands.HandLandmark.PINKY_TIP, mp_hands.HandLandmark.PINKY_PIP)
    return all([index_extended, middle_extended, ring_curled, pinky_curled])


def is_control_gesture_active(hand_landmarks):
    """A lenient check requiring only the ring and pinky fingers to be curled."""
    if not hand_landmarks:
        return False
    landmarks = hand_landmarks.landmark
    ring_curled = not is_finger_extended(landmarks, mp_hands.HandLandmark.RING_FINGER_TIP,
                                         mp_hands.HandLandmark.RING_FINGER_PIP)
    pinky_curled = not is_finger_extended(landmarks, mp_hands.HandLandmark.PINKY_TIP, mp_hands.HandLandmark.PINKY_PIP)
    return ring_curled and pinky_curled


def is_high_five(hand_landmarks):
    """Checks if a hand is doing a high five (all fingers extended)."""
    if not hand_landmarks:
        return False
    landmarks = hand_landmarks.landmark
    thumb_extended = landmarks[mp_hands.HandLandmark.THUMB_TIP].y < landmarks[mp_hands.HandLandmark.THUMB_IP].y
    index_extended = is_finger_extended(landmarks, mp_hands.HandLandmark.INDEX_FINGER_TIP,
                                        mp_hands.HandLandmark.INDEX_FINGER_PIP)
    middle_extended = is_finger_extended(landmarks, mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
                                         mp_hands.HandLandmark.MIDDLE_FINGER_PIP)
    ring_extended = is_finger_extended(landmarks, mp_hands.HandLandmark.RING_FINGER_TIP,
                                       mp_hands.HandLandmark.RING_FINGER_PIP)
    pinky_extended = is_finger_extended(landmarks, mp_hands.HandLandmark.PINKY_TIP, mp_hands.HandLandmark.PINKY_PIP)
    return all([thumb_extended, index_extended, middle_extended, ring_extended, pinky_extended])


def press_key_threaded(key, duration):
    """
    Presses a key in a separate thread to avoid blocking the main loop.
    """
    press(key, duration)


# --- Main Loop ---

# Define the layout for the TUI
layout = Layout()
layout.split(
    Layout(name="header", size=3),
    Layout(ratio=1, name="main"),
    Layout(size=10, name="footer"),
)
layout["main"].split_row(Layout(name="left"), Layout(name="right"))

# Set up the OpenCV window to be resizable and smaller
WINDOW_NAME = 'CrossyVision Controller'
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW_NAME, 800, 600)  # Width, Height

with Live(layout, screen=True, redirect_stderr=False, console=console) as live:
    while True:
        # Grab the latest processed frame and results from the processor thread
        frame, results = processor.read()

        # If the processor hasn't produced a frame yet, skip the loop
        if frame is None or results is None:
            continue

        # --- Face Masking Logic ---
        if MASK_FACE and face_masker:
            frame = face_masker.apply_mask(frame)

        # --- ALL LOGIC NOW USES THE PROCESSED FRAME AND RESULTS ---
        current_time = time.time()
        frame_height, frame_width, _ = frame.shape

        # Master Gesture Check (Reset)
        reset_gesture_detected = False
        if results.multi_hand_landmarks and len(results.multi_hand_landmarks) == 2:
            hand_states = {}
            for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
                handedness = results.multi_handedness[i].classification[0].label
                is_peace = is_peace_sign(hand_landmarks)
                is_five = is_high_five(hand_landmarks)
                hand_states[handedness] = {"is_peace": is_peace, "is_five": is_five}
            left_peace_right_five = hand_states.get("Left", {}).get("is_peace") and hand_states.get("Right", {}).get(
                "is_five")
            right_peace_left_five = hand_states.get("Right", {}).get("is_peace") and hand_states.get("Left", {}).get(
                "is_five")
            if left_peace_right_five or right_peace_left_five:
                reset_gesture_detected = True
                if (current_time - last_key_press_time.get('SPACEBAR', 0)) > 1.0:
                    threading.Thread(target=press_key_threaded, args=('SPACEBAR', 0.05), daemon=True).start()
                    log.info("--- ACTION: SPACEBAR (Restart) ---")
                    last_key_press_time['SPACEBAR'] = current_time

        if not reset_gesture_detected:
            left_hand_in_position = False
            right_hand_in_position = False
            if results.multi_hand_landmarks and len(results.multi_hand_landmarks) == 2:
                # We need to see both hands to properly manage state
                for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    handedness = results.multi_handedness[i].classification[0].label
                    if handedness == "Left":
                        if controls_active:
                            left_hand_in_position = is_control_gesture_active(hand_landmarks)
                        else:
                            left_hand_in_position = is_peace_sign(hand_landmarks)
                    else:  # Right
                        if controls_active:
                            right_hand_in_position = is_control_gesture_active(hand_landmarks)
                        else:
                            right_hand_in_position = is_peace_sign(hand_landmarks)

            both_hands_in_position = left_hand_in_position and right_hand_in_position

            if not controls_active:
                if both_hands_in_position:
                    activation_counter += 1
                    if activation_counter > ACTIVATION_CONFIDENCE_FRAMES:
                        log.info("CONTROLS ACTIVATED")
                        controls_active = True
                        activation_counter = 0  # Reset counter
                else:
                    activation_counter = 0  # Reset if gesture is broken
            else:  # If controls are currently active
                if not both_hands_in_position:
                    deactivation_counter += 1
                    if deactivation_counter > DEACTIVATION_CONFIDENCE_FRAMES:
                        log.info("CONTROLS DEACTIVATED")
                        controls_active = False
                        deactivation_counter = 0  # Reset counter
                else:
                    deactivation_counter = 0  # Reset if gesture is maintained
            if results.multi_hand_landmarks:
                for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    if is_control_gesture_active(hand_landmarks):
                        handedness = results.multi_handedness[i].classification[0].label
                        landmarks = hand_landmarks.landmark
                        control_fingers = {
                            "INDEX": (mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.INDEX_FINGER_MCP),
                            "MIDDLE": (mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
                                       mp_hands.HandLandmark.MIDDLE_FINGER_MCP)}
                        for finger_name, (tip_id, mcp_id) in control_fingers.items():
                            finger_id = f"{handedness.upper()}_{finger_name}"
                            mcp_pos, tip_pos = landmarks[mcp_id], landmarks[tip_id]
                            current_length = abs(mcp_pos.y - tip_pos.y)
                            if finger_id not in mcp_history:
                                mcp_history[finger_id] = deque(maxlen=CALIBRATION_FRAMES)
                                finger_lengths[finger_id] = deque(maxlen=CALIBRATION_FRAMES)

                            force_recalibration = False
                            if not previous_finger_is_above.get(finger_id, True):  # If finger is below the line
                                if finger_id in time_finger_went_below and time_finger_went_below[
                                    finger_id] is not None:
                                    if (current_time - time_finger_went_below[finger_id]) > FORCE_CALIBRATION_SECONDS:
                                        force_recalibration = True

                            if previous_finger_is_above.get(finger_id, True) or force_recalibration:
                                mcp_history[finger_id].append(mcp_pos.y)
                                finger_lengths[finger_id].append(current_length)

                            if len(mcp_history[finger_id]) > 0:
                                avg_mcp_y = sum(mcp_history[finger_id]) / len(mcp_history[finger_id])
                                avg_length = sum(finger_lengths[finger_id]) / len(finger_lengths[finger_id])
                                dynamic_offset = avg_length * RELATIVE_THRESHOLD_PERCENT
                                threshold_y = avg_mcp_y - dynamic_offset
                                trigger_thresholds[finger_id] = threshold_y
                                if finger_id not in previous_finger_is_above: previous_finger_is_above[
                                    finger_id] = tip_pos.y < threshold_y
            if controls_active and results.multi_hand_landmarks:
                for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    handedness = results.multi_handedness[i].classification[0].label
                    landmarks = hand_landmarks.landmark
                    key_mappings = {"INDEX": 'RIGHT' if handedness == "Left" else 'UP',
                                    "MIDDLE": 'LEFT' if handedness == "Left" else 'DOWN'}
                    for finger_name, key_action in key_mappings.items():
                        finger_id = f"{handedness.upper()}_{finger_name}"
                        if finger_id in trigger_thresholds:
                            tip_pos_y = landmarks[mp_hands.HandLandmark[f"{finger_name}_FINGER_TIP"]].y
                            threshold_y = trigger_thresholds[finger_id]
                            finger_is_currently_above = tip_pos_y < threshold_y
                            finger_was_previously_above = previous_finger_is_above.get(finger_id, True)

                            if finger_was_previously_above and not finger_is_currently_above:
                                time_finger_went_below[finger_id] = current_time
                            elif not finger_was_previously_above and finger_is_currently_above:
                                time_finger_went_below[finger_id] = None

                            if finger_was_previously_above and not finger_is_currently_above:
                                if (current_time - last_key_press_time.get(key_action, 0)) > COOLDOWN_SECONDS:
                                    threading.Thread(target=press_key_threaded, args=(key_action, 0.05),
                                                     daemon=True).start()
                                    log.info(f"--- ACTION: {key_action} ---")
                                    last_key_press_time[key_action] = current_time
                            previous_finger_is_above[finger_id] = finger_is_currently_above
        else:
            controls_active = False

        # --- Visual Feedback (Energy Tether) ---
        overlay = frame.copy()
        FINGER_CONNECTIONS = [
            (mp_hands.HandLandmark.INDEX_FINGER_MCP, mp_hands.HandLandmark.INDEX_FINGER_PIP),
            (mp_hands.HandLandmark.INDEX_FINGER_PIP, mp_hands.HandLandmark.INDEX_FINGER_DIP),
            (mp_hands.HandLandmark.INDEX_FINGER_DIP, mp_hands.HandLandmark.INDEX_FINGER_TIP),
            (mp_hands.HandLandmark.MIDDLE_FINGER_MCP, mp_hands.HandLandmark.MIDDLE_FINGER_PIP),
            (mp_hands.HandLandmark.MIDDLE_FINGER_PIP, mp_hands.HandLandmark.MIDDLE_FINGER_DIP),
            (mp_hands.HandLandmark.MIDDLE_FINGER_DIP, mp_hands.HandLandmark.MIDDLE_FINGER_TIP),
        ]

        if results.multi_hand_landmarks:
            for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
                handedness = results.multi_handedness[i].classification[0].label
                landmarks = hand_landmarks.landmark
                pixel_landmarks = {idx: (int(lm.x * frame_width), int(lm.y * frame_height)) for idx, lm in
                                   enumerate(landmarks)}
                for connection in FINGER_CONNECTIONS:
                    start_idx, end_idx = connection
                    if start_idx in pixel_landmarks and end_idx in pixel_landmarks:
                        cv2.line(frame, pixel_landmarks[start_idx], pixel_landmarks[end_idx], (255, 255, 255), 2)
                for conn in FINGER_CONNECTIONS:
                    for idx in conn:
                        cv2.circle(frame, pixel_landmarks[idx], 3, (0, 0, 255), -1)
                control_fingers_vis = {"INDEX": mp_hands.HandLandmark.INDEX_FINGER_TIP,
                                       "MIDDLE": mp_hands.HandLandmark.MIDDLE_FINGER_TIP}
                for finger_name, tip_id in control_fingers_vis.items():
                    finger_id = f"{handedness.upper()}_{finger_name}"
                    if finger_id in trigger_thresholds:
                        CYAN, YELLOW, RED, GREEN = (255, 255, 0), (0, 255, 255), (0, 0, 255), (0, 255, 0)
                        threshold_y_px = int(trigger_thresholds[finger_id] * frame_height)
                        tip_pos = pixel_landmarks[tip_id]
                        mcp_id = mp_hands.HandLandmark[f"{finger_name}_FINGER_MCP"]
                        mcp_pos = pixel_landmarks[mcp_id]
                        is_above = previous_finger_is_above.get(finger_id, True)
                        if is_above:
                            total_dist = abs(mcp_pos[1] - threshold_y_px)
                            current_dist = abs(tip_pos[1] - threshold_y_px)
                            proximity = max(0, min(1, current_dist / total_dist if total_dist > 0 else 0))
                            bar_color = [int(c1 * proximity + c2 * (1 - proximity)) for c1, c2 in zip(CYAN, YELLOW)]
                            line_color = GREEN
                        else:
                            bar_color = RED
                            line_color = RED
                        cv2.line(frame, (tip_pos[0] - 30, threshold_y_px), (tip_pos[0] + 30, threshold_y_px),
                                 line_color, 2)
                        cv2.rectangle(overlay, (tip_pos[0] - 2, tip_pos[1]), (tip_pos[0] + 2, threshold_y_px),
                                      bar_color,
                                      -1)
                        cv2.circle(frame, tip_pos, 5, bar_color, 2)

        alpha = 0.4
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        # --- TUI Update ---
        new_frame_time = time.time()
        if (new_frame_time - prev_frame_time) > 0:
            fps = 1 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time

        # Update Header
        status_text = Text("CONTROLS ACTIVE", style="bold green") if controls_active else Text("CONTROLS INACTIVE",
                                                                                               style="bold red")
        header_text = Text.assemble(status_text, Text(f" | FPS: {fps:.1f}", justify="right"))
        layout["header"].update(Panel(header_text, title="CrossyVision Controller", border_style="blue"))

        # Update Finger Progress Bars
        if results.multi_hand_landmarks:
            for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
                handedness = results.multi_handedness[i].classification[0].label
                landmarks = hand_landmarks.landmark
                for finger_name in ["INDEX", "MIDDLE"]:
                    finger_id = f"{handedness.upper()}_{finger_name}"
                    if finger_id in trigger_thresholds:
                        mcp_y = landmarks[mp_hands.HandLandmark[f"{finger_name}_FINGER_MCP"]].y
                        tip_y = landmarks[mp_hands.HandLandmark[f"{finger_name}_FINGER_TIP"]].y
                        threshold_y = trigger_thresholds[finger_id]

                        total_dist = abs(mcp_y - threshold_y)
                        current_dist_from_mcp = abs(mcp_y - tip_y)

                        progress_val = 0
                        if total_dist > 1e-6:  # Avoid division by zero
                            progress_val = min(100, max(0, (current_dist_from_mcp / total_dist) * 100))

                        is_above = previous_finger_is_above.get(finger_id, True)
                        bar = progress_bars[finger_id]
                        task_id = progress_tasks[finger_id]

                        # Change bar color based on activation state
                        if not is_above:
                            bar.columns[1].style = "red"  # BarColumn
                        else:
                            bar.columns[1].style = "cyan"

                        bar.update(task_id, completed=progress_val)

        # Update Panels
        left_panel_content = Columns([progress_bars["LEFT_INDEX"], progress_bars["LEFT_MIDDLE"]], expand=True,
                                     equal=True)
        right_panel_content = Columns([progress_bars["RIGHT_INDEX"], progress_bars["RIGHT_MIDDLE"]], expand=True,
                                      equal=True)
        layout["left"].update(Panel(left_panel_content, title="Left Hand"))
        layout["right"].update(Panel(right_panel_content, title="Right Hand"))

        # Update Footer Log
        log_text = Text("\n".join(log_deque), no_wrap=True)
        layout["footer"].update(Panel(log_text, title="Log", border_style="blue"))

        cv2.imshow(WINDOW_NAME, frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

# --- Cleanup ---
log.info("Shutting down services...")
if face_masker:
    face_masker.stop()
processor.stop()
vs.stop()
cv2.destroyAllWindows()