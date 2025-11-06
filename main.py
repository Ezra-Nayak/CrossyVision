import cv2
import mediapipe as mp
import time

# --- Constants and Initialization ---

# MediaPipe Hands setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    model_complexity=0,  # Use 0 for the lite model, 1 for the full model
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_drawing = mp.solutions.drawing_utils

# --- OpenCV VideoCapture setup ---
# Explicitly request the DirectShow backend
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
# Then, attempt to set the desired FPS
cap.set(cv2.CAP_PROP_FPS, 60)

# FPS calculation variables
prev_frame_time = 0
new_frame_time = 0

print("Starting Milestone 2: Gesture State Detection. Press 'ESC' to quit.")


# --- Helper Function ---

def is_peace_sign(hand_landmarks):
    """
    Checks if a hand is making a peace sign based on landmark positions.
    """
    if not hand_landmarks:
        return False

    # Landmark indices for fingertips and lower joints
    # See MediaPipe documentation for the landmark map
    landmarks = hand_landmarks.landmark

    # Check if Index and Middle fingers are extended (tip is above the joint below it)
    index_finger_extended = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP].y < landmarks[
        mp_hands.HandLandmark.INDEX_FINGER_PIP].y
    middle_finger_extended = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y < landmarks[
        mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y

    # Check if Ring and Pinky fingers are curled (tip is below the joint below it)
    ring_finger_curled = landmarks[mp_hands.HandLandmark.RING_FINGER_TIP].y > landmarks[
        mp_hands.HandLandmark.RING_FINGER_PIP].y
    pinky_finger_curled = landmarks[mp_hands.HandLandmark.PINKY_TIP].y > landmarks[mp_hands.HandLandmark.PINKY_PIP].y

    # A peace sign is when index and middle are up, and ring and pinky are down.
    return all([index_finger_extended, middle_finger_extended, ring_finger_curled, pinky_finger_curled])


# --- Main Loop ---

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # --- Frame Processing ---

    # Convert BGR to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    # --- Gesture Logic and Drawing ---
    left_hand_status = "LEFT: NOT DETECTED"
    right_hand_status = "RIGHT: NOT DETECTED"
    controls_active = False

    if results.multi_hand_landmarks and results.multi_handedness:
        # Loop through each detected hand
        for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
            # Determine if the hand is left or right
            handedness = results.multi_handedness[i].classification[0].label

            # Check for peace sign
            is_peace = is_peace_sign(hand_landmarks)
            status_text = f"{handedness.upper()}: PEACE" if is_peace else f"{handedness.upper()}: WAITING"

            if handedness == "Left":
                left_hand_status = status_text
            else:  # Right
                right_hand_status = status_text

            # Draw the landmarks on the frame
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=hand_landmarks,
                connections=mp_hands.HAND_CONNECTIONS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2),
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2))

    # --- FPS Calculation and Display ---
    new_frame_time = time.time()
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # --- Status Display ---
    cv2.putText(frame, left_hand_status, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, right_hand_status, (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)

    # --- Display the Frame ---
    cv2.imshow('Milestone 2 - Gesture Detection', frame)

    if cv2.waitKey(5) & 0xFF == 27:
        break

# --- Cleanup ---
print("Shutting down.")
cap.release()
cv2.destroyAllWindows()