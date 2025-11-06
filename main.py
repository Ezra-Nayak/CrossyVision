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
                # Opaque image logic: simply replace the region
                if mask_resized.shape[2] != 4:
                    frame[y1:y2, x1:x2] = mask_resized
                # Transparent image logic
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
COOLDOWN_SECONDS = 0.12          # Cooldown for individual finger presses.
CALIBRATION_FRAMES = 500         # Number of frames to average for the trigger line (~1 second).
RELATIVE_THRESHOLD_PERCENT = 0.85 # Trigger line is 40% of the way down from the MCP. Tune this for comfort.

# State stabilization parameters
ACTIVATION_CONFIDENCE_FRAMES = 10  # Require 5 consecutive frames of correct gesture to activate.
DEACTIVATION_CONFIDENCE_FRAMES = 25 # Require 5 consecutive frames of broken gesture to deactivate.

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

# --- Face Masking Setup ---
MASK_FACE = True # Set to False to disable face masking

# Load the mask image
if MASK_FACE:
    try:
        mask_image = cv2.imread("mask.png", cv2.IMREAD_UNCHANGED)
        if mask_image is None:
            print("Warning: 'mask.png' not found or could not be read. Face masking will be disabled.")
            MASK_FACE = False
    except Exception as e:
        print(f"Error loading 'mask.png': {e}. Face masking will be disabled.")
        MASK_FACE = False

# --- Threaded Setup ---
print("Starting threaded services...")
# Start the camera stream thread
vs = WebcamVideoStream(src=0).start()
# Start the MediaPipe processing thread
processor = MediaPipeProcessor(stream=vs, hands_instance=hands).start()
# Start the Face Masking thread if enabled
face_masker = None
if MASK_FACE:
    face_masker = FaceMasker(stream=vs, mask_image=mask_image).start()

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

# For state stabilization (debouncing)
activation_counter = 0
deactivation_counter = 0

print("CrossyVision Controller Initialized. Press 'ESC' to quit.")

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
                print("--- ACTION: SPACEBAR (Restart) ---")
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
                    print("CONTROLS ACTIVATED")
                    controls_active = True
                    activation_counter = 0  # Reset counter
            else:
                activation_counter = 0  # Reset if gesture is broken
        else:  # If controls are currently active
            if not both_hands_in_position:
                deactivation_counter += 1
                if deactivation_counter > DEACTIVATION_CONFIDENCE_FRAMES:
                    print("CONTROLS DEACTIVATED")
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

                    # Draw the proximity bar on the transparent overlay (thinner)
                    cv2.rectangle(overlay, (tip_pos[0] - 2, tip_pos[1]), (tip_pos[0] + 2, threshold_y_px), bar_color,
                                  -1)

                    # Draw the fingertip aura (smaller)
                    cv2.circle(frame, tip_pos, 5, bar_color, 2)

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
if face_masker:
    face_masker.stop()
processor.stop()
vs.stop()
cv2.destroyAllWindows()