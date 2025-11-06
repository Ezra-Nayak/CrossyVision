import cv2
import mediapipe as mp
import time
from pyKey import press

# --- Constants and Initialization ---

# Control parameters
COOLDOWN_SECONDS = 0.3  # Cooldown for twitch/fold actions. May need tuning.

# MediaPipe Hands setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    model_complexity=0,
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
controls_active = False     # Global state for whether controls are on or off
previous_finger_states = {} # Stores the last known extended state (True/False) for each finger
last_key_press_time = {}    # Stores the timestamp of the last press for each key

print("Starting Milestone 3 (Revised): Finger Fold Detection. Press 'ESC' to quit.")
print("Ensure Crossy Road is the active window!")


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
    """
    A more lenient check to see if the user is still in 'control mode'.
    Requires only the ring and pinky fingers to be curled.
    """
    if not hand_landmarks:
        return False
    landmarks = hand_landmarks.landmark

    ring_curled = not is_finger_extended(landmarks, mp_hands.HandLandmark.RING_FINGER_TIP,
                                         mp_hands.HandLandmark.RING_FINGER_PIP)
    pinky_curled = not is_finger_extended(landmarks, mp_hands.HandLandmark.PINKY_TIP, mp_hands.HandLandmark.PINKY_PIP)

    return ring_curled and pinky_curled

# --- Main Loop ---

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        continue

    # --- Frame Processing ---
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    # --- State Management (NEW LOGIC)---

    # Check the state of each hand
    left_hand_in_position = False
    right_hand_in_position = False
    left_hand_landmarks = None
    right_hand_landmarks = None

    if results.multi_hand_landmarks and results.multi_handedness:
        for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
            handedness = results.multi_handedness[i].classification[0].label
            if handedness == "Left":
                left_hand_landmarks = hand_landmarks
                if controls_active:
                    # Use the lenient check if controls are already on
                    left_hand_in_position = is_control_gesture_active(hand_landmarks)
                else:
                    # Use the strict check to activate
                    left_hand_in_position = is_peace_sign(hand_landmarks)
            else:  # Right
                right_hand_landmarks = hand_landmarks
                if controls_active:
                    right_hand_in_position = is_control_gesture_active(hand_landmarks)
                else:
                    right_hand_in_position = is_peace_sign(hand_landmarks)

    # Update the global controls_active state
    if not controls_active and (left_hand_in_position and right_hand_in_position):
        print("CONTROLS ACTIVATED")
        controls_active = True
        # Clear previous states to avoid false triggers on activation
        previous_finger_states.clear()
    elif controls_active and not (left_hand_in_position and right_hand_in_position):
        print("CONTROLS DEACTIVATED")
        controls_active = False

    # --- Control Logic (NEW FOLD DETECTION) ---
    if controls_active and results.multi_hand_landmarks:
        current_time = time.time()

        for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
            handedness = results.multi_handedness[i].classification[0].label
            landmarks = hand_landmarks.landmark

            # Get current extended state of control fingers
            index_is_extended = is_finger_extended(landmarks, mp_hands.HandLandmark.INDEX_FINGER_TIP,
                                                   mp_hands.HandLandmark.INDEX_FINGER_PIP)
            middle_is_extended = is_finger_extended(landmarks, mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
                                                    mp_hands.HandLandmark.MIDDLE_FINGER_PIP)

            # Define finger IDs for our state dictionary
            right_index_id = 'RIGHT_INDEX'
            right_middle_id = 'RIGHT_MIDDLE'
            left_index_id = 'LEFT_INDEX'
            left_middle_id = 'LEFT_MIDDLE'

            if handedness == "Right":
                # Check for UP action (index finger fold)
                if previous_finger_states.get(right_index_id, True) and not index_is_extended:
                    if (current_time - last_key_press_time.get('UP', 0)) > COOLDOWN_SECONDS:
                        press('UP', 0.1)
                        print("ACTION: UP")
                        last_key_press_time['UP'] = current_time

                # Check for DOWN action (middle finger fold)
                if previous_finger_states.get(right_middle_id, True) and not middle_is_extended:
                    if (current_time - last_key_press_time.get('DOWN', 0)) > COOLDOWN_SECONDS:
                        press('DOWN', 0.1)
                        print("ACTION: DOWN")
                        last_key_press_time['DOWN'] = current_time

                # Update previous states for right hand fingers
                previous_finger_states[right_index_id] = index_is_extended
                previous_finger_states[right_middle_id] = middle_is_extended

            elif handedness == "Left":
                # Check for RIGHT action (index finger fold)
                if previous_finger_states.get(left_index_id, True) and not index_is_extended:
                    if (current_time - last_key_press_time.get('RIGHT', 0)) > COOLDOWN_SECONDS:
                        press('RIGHT', 0.1)
                        print("ACTION: RIGHT")
                        last_key_press_time['RIGHT'] = current_time

                # Check for LEFT action (middle finger fold)
                if previous_finger_states.get(left_middle_id, True) and not middle_is_extended:
                    if (current_time - last_key_press_time.get('LEFT', 0)) > COOLDOWN_SECONDS:
                        press('LEFT', 0.1)
                        print("ACTION: LEFT")
                        last_key_press_time['LEFT'] = current_time

                # Update previous states for left hand fingers
                previous_finger_states[left_index_id] = index_is_extended
                previous_finger_states[left_middle_id] = middle_is_extended
    else:
        # If controls are not active, reset the finger states to avoid false triggers
        previous_finger_states.clear()

    # --- Visual Feedback ---
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

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