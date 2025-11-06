import cv2
import mediapipe as mp
import time
from pyKey import press
from collections import deque
import datetime

# --- Constants and Initialization ---

# Control parameters
COOLDOWN_SECONDS = 0.12          # Cooldown for individual finger presses.
CALIBRATION_FRAMES = 60         # Number of frames to average for the trigger line (~1 second).
RELATIVE_THRESHOLD_PERCENT = 0.9 # Trigger line is 40% of the way down from the MCP. Tune this for comfort.

# MediaPipe Hands setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    model_complexity=1,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_drawing = mp.solutions.drawing_utils

# OpenCV VideoCapture setup
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FPS, 60)

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

# --- Main Loop ---

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        continue

    # --- Frame Processing ---
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    # Define frame-specific variables at the top level of the loop
    current_time = time.time()
    frame_height, frame_width, _ = frame.shape

    # --- Master Gesture Check (Reset) ---
    # This block runs on every frame, regardless of the 'controls_active' state.
    reset_gesture_detected = False
    if results.multi_hand_landmarks and len(results.multi_hand_landmarks) == 2:
        # Determine the state of each hand
        hand_states = {}
        for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
            handedness = results.multi_handedness[i].classification[0].label
            is_peace = is_peace_sign(hand_landmarks)
            is_five = is_high_five(hand_landmarks)
            hand_states[handedness] = {"is_peace": is_peace, "is_five": is_five}

        # Check for the two valid combinations
        left_peace_right_five = hand_states.get("Left", {}).get("is_peace") and hand_states.get("Right", {}).get(
            "is_five")
        right_peace_left_five = hand_states.get("Right", {}).get("is_peace") and hand_states.get("Left", {}).get(
            "is_five")

        if left_peace_right_five or right_peace_left_five:
            reset_gesture_detected = True
            if (current_time - last_key_press_time.get('SPACEBAR', 0)) > 1.0:
                press('SPACEBAR', 0.05)
                print("--- ACTION: SPACEBAR (Restart) ---")
                last_key_press_time['SPACEBAR'] = current_time

    # --- Regular State Management & Control Logic ---
    if not reset_gesture_detected:
        # If a reset was not detected, proceed with normal logic.
        # This 'if' block now contains all the other logic.

        # --- Regular State Management (Activation/Deactivation) ---
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
                else:  # Right
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

        # --- Control Logic (Calibration and Action) ---
        # Part 1: Calibration
        if results.multi_hand_landmarks:
            for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
                if is_control_gesture_active(hand_landmarks):
                    handedness = results.multi_handedness[i].classification[0].label
                    landmarks = hand_landmarks.landmark

                    control_fingers = {
                        "INDEX": (mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.INDEX_FINGER_MCP),
                        "MIDDLE": (mp_hands.HandLandmark.MIDDLE_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_MCP)
                    }

                    for finger_name, (tip_id, mcp_id) in control_fingers.items():
                        finger_id = f"{handedness.upper()}_{finger_name}"
                        mcp_pos = landmarks[mcp_id]
                        tip_pos = landmarks[tip_id]
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
                            if finger_id not in previous_finger_is_above:
                                previous_finger_is_above[finger_id] = tip_pos.y < threshold_y

        # Part 2: Action
        if controls_active and results.multi_hand_landmarks:
            for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
                handedness = results.multi_handedness[i].classification[0].label
                landmarks = hand_landmarks.landmark
                key_mappings = {
                    "INDEX": 'RIGHT' if handedness == "Left" else 'UP',
                    "MIDDLE": 'LEFT' if handedness == "Left" else 'DOWN'
                }
                for finger_name, key_action in key_mappings.items():
                    finger_id = f"{handedness.upper()}_{finger_name}"
                    if finger_id in trigger_thresholds:
                        tip_pos_y = landmarks[mp_hands.HandLandmark[f"{finger_name}_FINGER_TIP"]].y
                        threshold_y = trigger_thresholds[finger_id]
                        finger_is_currently_above = tip_pos_y < threshold_y
                        finger_was_previously_above = previous_finger_is_above.get(finger_id, True)
                        if finger_was_previously_above and not finger_is_currently_above:
                            if (current_time - last_key_press_time.get(key_action, 0)) > COOLDOWN_SECONDS:
                                press(key_action, 0.05)
                                print(f"--- ACTION: {key_action} ---")
                                last_key_press_time[key_action] = current_time
                        previous_finger_is_above[finger_id] = finger_is_currently_above
    else:
        # This else corresponds to 'if not reset_gesture_detected'.
        # If a reset happened, we must ensure controls are off for the next frame.
        controls_active = False

    # --- Visual Feedback ---
    if results.multi_hand_landmarks:
        for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            handedness = results.multi_handedness[i].classification[0].label
            landmarks = hand_landmarks.landmark
            control_fingers_vis = {"INDEX": mp_hands.HandLandmark.INDEX_FINGER_TIP, "MIDDLE": mp_hands.HandLandmark.MIDDLE_FINGER_TIP}
            for finger_name, tip_id in control_fingers_vis.items():
                finger_id = f"{handedness.upper()}_{finger_name}"
                if finger_id in trigger_thresholds:
                    threshold_y_pixels = int(trigger_thresholds[finger_id] * frame_height)
                    tip_pos_x_pixels = int(landmarks[tip_id].x * frame_width)
                    is_above = previous_finger_is_above.get(finger_id, True)
                    line_color = (0, 255, 0) if is_above else (0, 0, 255)
                    cv2.line(frame, (tip_pos_x_pixels - 30, threshold_y_pixels), (tip_pos_x_pixels + 30, threshold_y_pixels), line_color, 2)

    # --- On-Screen Display ---
    status_text = "CONTROLS ACTIVE" if controls_active else "WAITING FOR GESTURE"
    color = (0, 255, 0) if controls_active else (0, 255, 255)
    cv2.putText(frame, status_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
    new_frame_time = time.time()
    fps = 1 / (new_frame_time - prev_frame_time) if (new_frame_time - prev_frame_time) > 0 else 0
    prev_frame_time = new_frame_time
    cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow('CrossyVision Controller', frame)

    if cv2.waitKey(5) & 0xFF == 27:
        break

# --- Cleanup ---
print("Shutting down.")
cap.release()
cv2.destroyAllWindows()