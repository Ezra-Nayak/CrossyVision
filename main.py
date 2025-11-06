import cv2
import mediapipe as mp

# --- Constants and Initialization ---

# MediaPipe Hands setup
mp_hands = mp.solutions.hands
# Parameters:
#   max_num_hands: We need to see both hands.
#   min_detection_confidence: A higher value can be more accurate but might miss hands that are less clear.
#   min_tracking_confidence: Similar to detection, but for tracking the hand once it's found.
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# OpenCV VideoCapture setup
# NOTE: The index '0' might need to be changed. If your phone webcam doesn't show up,
# try '1', '2', etc. Run a simple script to list your camera devices if needed.
cap = cv2.VideoCapture(0)

print("Starting Milestone 1: Visual Hand Tracking. Press 'ESC' to quit.")

# --- Main Loop ---

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # --- Frame Processing ---

    # 1. Flip the frame horizontally for a later selfie-view display.
    # This makes it feel like a mirror, which is more intuitive.
    # frame = cv2.flip(frame, 1)

    # 2. Convert the BGR image to RGB. MediaPipe requires RGB.
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 3. Process the frame to find hands.
    results = hands.process(rgb_frame)

    # 4. Draw the hand annotations on the original frame.
    if results.multi_hand_landmarks:
        # Loop through each detected hand
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=hand_landmarks,
                connections=mp_hands.HAND_CONNECTIONS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2),
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2))

    # --- Display the Frame ---
    cv2.imshow('Milestone 1 - Hand Tracking', frame)

    # --- Exit Condition ---
    # Press ESC key to exit the loop
    if cv2.waitKey(5) & 0xFF == 27:
        break

# --- Cleanup ---
print("Shutting down.")
cap.release()
cv2.destroyAllWindows()