import cv2
import mediapipe as mp
import time
from pyKey import press
from collections import deque
import threading
from threading import Thread
import numpy as np

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

# --- Constants and Initialization ---

# Control parameters
COOLDOWN_SECONDS = 0.12          # Cooldown for individual finger presses.
CALIBRATION_FRAMES = 500         # Number of frames to average for the trigger line (~1 second).
RELATIVE_THRESHOLD_PERCENT = 0.9 # Trigger line is 40% of the way down from the MCP. Tune this for comfort.

# MediaPipe Hands setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode = False,
    model_complexity=1,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_drawing = mp.solutions.drawing_utils

# --- Threaded Setup ---
print("Starting threaded services...")
# Start the camera stream thread
vs = WebcamVideoStream(src=0).start()
# Start the MediaPipe processing thread, passing it the camera stream and hands object
processor = MediaPipeProcessor(stream=vs, hands_instance=hands).start()
time.sleep(2.0) # Allow services to warm up

# --- State Variables ---

# FPS calculation
prev_frame_time = 0

# Dictionaries to hold state information
controls_active = False
last_key_press_time = {}
# For dynamic thresholds
mcp_history = {}            # Stores recent MCP joint Y-positions for averaging
finger_lengths = {}         # Stores recent finger lengths for averaging
trigger_thresholds = {}     # Stores the calculated trigger line Y-position for each finger
previous_finger_is_above = {} # Stores the last known position relative to the trigger line

print("CrossyVision Controller Initialized. Press 'ESC' to quit.")
print("Ensure Crossy Road is the active window!")


# --- Helper Functions ---

def is_finger_extended(landmarks, tip_landmark, pip_landmark):
    """Checks if a specific finger is extended."""
    return landmarks[tip_landmark].y < landmarks[pip_landmark].y

def is_peace_sign(hand_landmarks):
    if not hand_landmarks:
        return False
    landmarks = hand_landmarks.landmark
    index_extended = is_finger_extended(landmarks, mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.INDEX_FINGER_PIP)
    middle_extended = is_finger_extended(landmarks, mp_hands.HandLandmark.MIDDLE_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_PIP)
    ring_curled = not is_finger_extended(landmarks, mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.RING_FINGER_PIP)
    pinky_curled = not is_finger_extended(landmarks, mp_hands.HandLandmark.PINKY_TIP, mp_hands.HandLandmark.PINKY_PIP)
    return all([index_extended, middle_extended, ring_curled, pinky_curled])

def is_control_gesture_active(hand_landmarks):
    """A lenient check requiring only the ring and pinky fingers to be curled."""
    if not hand_landmarks:
        return False
    landmarks = hand_landmarks.landmark
    ring_curled = not is_finger_extended(landmarks, mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.RING_FINGER_PIP)
    pinky_curled = not is_finger_extended(landmarks, mp_hands.HandLandmark.PINKY_TIP, mp_hands.HandLandmark.PINKY_PIP)
    return ring_curled and pinky_curled

def is_high_five(hand_landmarks):
    """Checks if a hand is doing a high five (all fingers extended)."""
    if not hand_landmarks:
        return False
    landmarks = hand_landmarks.landmark
    thumb_extended = landmarks[mp_hands.HandLandmark.THUMB_TIP].y < landmarks[mp_hands.HandLandmark.THUMB_IP].y
    index_extended = is_finger_extended(landmarks, mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.INDEX_FINGER_PIP)
    middle_extended = is_finger_extended(landmarks, mp_hands.HandLandmark.MIDDLE_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_PIP)
    ring_extended = is_finger_extended(landmarks, mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.RING_FINGER_PIP)
    pinky_extended = is_finger_extended(landmarks, mp_hands.HandLandmark.PINKY_TIP, mp_hands.HandLandmark.PINKY_PIP)
    return all([thumb_extended, index_extended, middle_extended, ring_extended, pinky_extended])

def press_key_threaded(key, duration):
    """
    Presses a key in a separate thread to avoid blocking the main loop.
    """
    press(key, duration)

# --- Main Loop ---

while True:
    # Grab the latest processed frame and results from the processor thread
    frame, results = processor.read()

    # If the processor hasn't produced a frame yet, skip the loop
    if frame is None or results is None:
        continue

    # --- ALL LOGIC NOW USES THE PROCESSED FRAME AND RESULTS ---
    current_time = time.time()
    frame_height, frame_width, _ = frame.shape

    # Master Gesture Check (Reset)
    reset_gesture_detected = False
    # (The rest of your existing, working logic goes here, unchanged)
    # ...
    # (The following is your complete, verified logic block, now correctly placed)
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
                print("--- ACTION: SPACEBAR (Restart) ---")
                last_key_press_time['SPACEBAR'] = current_time

    if not reset_gesture_detected:
        left_hand_in_position = False
        right_hand_in_position = False
        if results.multi_hand_landmarks:
            for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
                handedness = results.multi_handedness[i].classification[0].label
                if handedness == "Left":
                    if controls_active:
                        left_hand_in_position = is_control_gesture_active(hand_landmarks)
                    else:
                        left_hand_in_position = is_peace_sign(hand_landmarks)
                else:
                    if controls_active:
                        right_hand_in_position = is_control_gesture_active(hand_landmarks)
                    else:
                        right_hand_in_position = is_peace_sign(hand_landmarks)
        if not controls_active and (left_hand_in_position and right_hand_in_position):
            print("CONTROLS ACTIVATED")
            controls_active = True
        elif controls_active and not (left_hand_in_position and right_hand_in_position):
            print("CONTROLS DEACTIVATED")
            controls_active = False
        if results.multi_hand_landmarks:
            for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
                if is_control_gesture_active(hand_landmarks):
                    handedness = results.multi_handedness[i].classification[0].label
                    landmarks = hand_landmarks.landmark
                    control_fingers = {
                        "INDEX": (mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.INDEX_FINGER_MCP),
                        "MIDDLE": (mp_hands.HandLandmark.MIDDLE_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_MCP)}
                    for finger_name, (tip_id, mcp_id) in control_fingers.items():
                        finger_id = f"{handedness.upper()}_{finger_name}"
                        mcp_pos, tip_pos = landmarks[mcp_id], landmarks[tip_id]
                        current_length = abs(mcp_pos.y - tip_pos.y)
                        if finger_id not in mcp_history:
                            mcp_history[finger_id] = deque(maxlen=CALIBRATION_FRAMES)
                            finger_lengths[finger_id] = deque(maxlen=CALIBRATION_FRAMES)
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
                            if (current_time - last_key_press_time.get(key_action, 0)) > COOLDOWN_SECONDS:
                                threading.Thread(target=press_key_threaded, args=(key_action, 0.05),
                                                 daemon=True).start()
                                print(f"--- ACTION: {key_action} ---")
                                last_key_press_time[key_action] = current_time
                        previous_finger_is_above[finger_id] = finger_is_currently_above
    else:
        controls_active = False

    # --- Visual Feedback (Energy Tether) ---

    # Create a transparent overlay to draw the tether bars on
    overlay = frame.copy()

    # Define the specific connections for the two control fingers
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

            # --- 1. Draw the selective skeleton ---
            # Convert normalized landmarks to pixel coordinates
            pixel_landmarks = {idx: (int(lm.x * frame_width), int(lm.y * frame_height)) for idx, lm in
                               enumerate(landmarks)}

            # Draw connections
            for connection in FINGER_CONNECTIONS:
                start_idx, end_idx = connection
                if start_idx in pixel_landmarks and end_idx in pixel_landmarks:
                    cv2.line(frame, pixel_landmarks[start_idx], pixel_landmarks[end_idx], (255, 255, 255), 2)

            # Draw landmarks (joints) for those fingers
            for conn in FINGER_CONNECTIONS:
                for idx in conn:
                    cv2.circle(frame, pixel_landmarks[idx], 3, (0, 0, 255), -1)

            # --- 2. Draw the Energy Tethers ---
            control_fingers_vis = {"INDEX": mp_hands.HandLandmark.INDEX_FINGER_TIP,
                                   "MIDDLE": mp_hands.HandLandmark.MIDDLE_FINGER_TIP}
            for finger_name, tip_id in control_fingers_vis.items():
                finger_id = f"{handedness.upper()}_{finger_name}"

                if finger_id in trigger_thresholds:
                    # Define colors
                    CYAN = (255, 255, 0)
                    YELLOW = (0, 255, 255)
                    RED = (0, 0, 255)
                    GREEN = (0, 255, 0)

                    # Get key positions in pixels
                    threshold_y_px = int(trigger_thresholds[finger_id] * frame_height)
                    tip_pos = pixel_landmarks[tip_id]
                    mcp_id = mp_hands.HandLandmark[f"{finger_name}_FINGER_MCP"]
                    mcp_pos = pixel_landmarks[mcp_id]

                    is_above = previous_finger_is_above.get(finger_id, True)

                    # Determine colors based on state
                    if is_above:
                        # Calculate proximity for color gradient
                        total_dist = abs(mcp_pos[1] - threshold_y_px)
                        current_dist = abs(tip_pos[1] - threshold_y_px)
                        proximity = max(0, min(1, current_dist / total_dist if total_dist > 0 else 0))

                        # Interpolate color from Cyan (far) to Yellow (close)
                        bar_color = [int(c1 * proximity + c2 * (1 - proximity)) for c1, c2 in zip(CYAN, YELLOW)]
                        line_color = GREEN
                    else:
                        bar_color = RED
                        line_color = RED

                    # Draw the main trigger line
                    cv2.line(frame, (tip_pos[0] - 30, threshold_y_px), (tip_pos[0] + 30, threshold_y_px),
                             line_color, 2)

                    # Draw the proximity bar on the transparent overlay
                    cv2.rectangle(overlay, (tip_pos[0] - 5, tip_pos[1]), (tip_pos[0] + 5, threshold_y_px),
                                  bar_color, -1)

                    # Draw the fingertip aura
                    cv2.circle(frame, tip_pos, 8, bar_color, 2)

    # Blend the overlay with the main frame to create transparency effect
    alpha = 0.4  # Transparency factor
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    # We are no longer drawing any text
    new_frame_time = time.time()
    prev_frame_time = new_frame_time

    cv2.imshow('CrossyVision Controller', frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

# --- Cleanup ---
print("Shutting down services...")
processor.stop()
vs.stop()
cv2.destroyAllWindows()