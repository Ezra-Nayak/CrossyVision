import cv2
import mediapipe as mp
import time
from pyKey import press
from collections import deque
import threading
from threading import Thread
import numpy as np

# --- Rich TUI Imports ---
from rich.console import Console, Group
from rich.table import Table
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import Progress, BarColumn, TextColumn
from rich.columns import Columns
from rich.text import Text
from rich.box import ROUNDED
from rich.align import Align
import logging
import ascii_arrow

# --- Standard Logging & TUI Setup ---
logging.basicConfig(level=logging.CRITICAL)
log = logging.getLogger(__name__)

# Rich Console for the TUI
console = Console()


class WebcamVideoStream:
    """
    A class to read frames from a webcam in a dedicated thread.
    This prevents the main processing loop from being blocked by camera I/O.
    """

    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src, cv2.CAP_DSHOW)
        self.stream.set(cv2.CAP_PROP_FPS, 60)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False

    def start(self):
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        while True:
            if self.stopped:
                return
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True


class MediaPipeProcessor:
    """
    A class to run MediaPipe hand processing in a dedicated thread.
    """

    def __init__(self, stream, hands_instance):
        self.stream = stream
        self.hands = hands_instance
        self.stopped = False
        self.results = None
        self.output_frame = None

    def start(self):
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        while True:
            if self.stopped:
                return
            frame = self.stream.read()
            if frame is None:
                continue
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.results = self.hands.process(rgb_frame)
            self.output_frame = frame.copy()

    def read(self):
        return self.output_frame, self.results

    def stop(self):
        self.stopped = True


class FaceMasker:
    """
    Runs face detection and smoothing in a dedicated thread at 60 UPS.
    """

    def __init__(self, stream, mask_image):
        self.stream = stream
        self.mask_image = mask_image
        self.stopped = False
        self.face_detection = mp.solutions.face_detection.FaceDetection(model_selection=0,
                                                                        min_detection_confidence=0.5)
        self.smoothed_bbox = None
        self.latest_bbox_for_overlay = None
        self.lock = threading.Lock()

    def start(self):
        Thread(target=self.update, args=(), daemon=True).start()
        return self

    def update(self):
        updates_per_second = 60
        sleep_duration = 1.0 / updates_per_second

        while not self.stopped:
            start_time = time.time()
            frame = self.stream.read()
            if frame is None:
                time.sleep(sleep_duration)
                continue

            face_frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_results = self.face_detection.process(face_frame_rgb)
            ih, iw, _ = frame.shape

            current_bbox = None
            if face_results.detections:
                detection = face_results.detections[0]
                bboxC = detection.location_data.relative_bounding_box
                current_bbox = (int(bboxC.xmin * iw), int(bboxC.ymin * ih),
                                int(bboxC.width * iw), int(bboxC.height * ih))

            if current_bbox:
                if self.smoothed_bbox is None:
                    self.smoothed_bbox = current_bbox
                else:
                    x1, y1, w1, h1 = self.smoothed_bbox
                    x2, y2, w2, h2 = current_bbox
                    smoothing = 0.1
                    new_x = int(x1 * (1 - smoothing) + x2 * smoothing)
                    new_y = int(y1 * (1 - smoothing) + y2 * smoothing)
                    new_w = int(w1 * (1 - smoothing) + w2 * smoothing)
                    new_h = int(h1 * (1 - smoothing) + h2 * smoothing)
                    self.smoothed_bbox = (new_x, new_y, new_w, new_h)

            with self.lock:
                self.latest_bbox_for_overlay = self.smoothed_bbox

            elapsed_time = time.time() - start_time
            time.sleep(max(0, sleep_duration - elapsed_time))

    def apply_mask(self, frame):
        local_bbox = None
        with self.lock:
            if self.latest_bbox_for_overlay:
                local_bbox = self.latest_bbox_for_overlay

        if local_bbox:
            ih, iw, _ = frame.shape
            x, y, w, h = local_bbox
            x_new = x - int(w * 0.25)
            y_new = y - int(h * 0.55)
            w_new = int(w * 1.5)
            h_new = int(h * 1.8)

            y1, y2 = max(0, y_new), min(ih, y_new + h_new)
            x1, x2 = max(0, x_new), min(iw, x_new + w_new)
            roi_h, roi_w = y2 - y1, x2 - x1

            if roi_h > 0 and roi_w > 0:
                mask_resized = cv2.resize(self.mask_image, (roi_w, roi_h))
                if len(mask_resized.shape) == 2:
                    mask_resized = cv2.cvtColor(mask_resized, cv2.COLOR_GRAY2BGR)

                if mask_resized.shape[2] != 4:
                    frame[y1:y2, x1:x2] = mask_resized
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
COOLDOWN_SECONDS = 0.08
FORCE_CALIBRATION_SECONDS = 2.0
CALIBRATION_FRAMES = 1200
RELATIVE_THRESHOLD_PERCENT = 0.85

ACTIVATION_CONFIDENCE_FRAMES = 10
DEACTIVATION_CONFIDENCE_FRAMES = 25

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
MASK_FACE = True

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
vs = WebcamVideoStream(src=0).start()
processor = MediaPipeProcessor(stream=vs, hands_instance=hands).start()
face_masker = None
if MASK_FACE:
    face_masker = FaceMasker(stream=vs, mask_image=mask_image).start()

time.sleep(2.0)

# --- State Variables ---
left_hand_actions = deque()
right_hand_actions = deque()
action_symbols = {
    "UP": "↑", "DOWN": "↓", "LEFT": "←", "RIGHT": "→", "SPACEBAR": "␣"
}
finger_presence = {}

prev_frame_time = 0
new_frame_time = 0
fps = 0

controls_active = False
last_key_press_time = {}
mcp_history = {}
finger_lengths = {}
trigger_thresholds = {}
previous_finger_is_above = {}
time_finger_went_below = {}

activation_counter = 0
deactivation_counter = 0

# --- TUI Elements ---
action_styles = {"UP": "light_salmon1", "RIGHT": "light_salmon1", "DOWN": "light_slate_grey",
                 "LEFT": "light_slate_grey"}
symbol_to_action = {v: k for k, v in action_symbols.items()}

# --- Helper Functions ---

def is_finger_extended(landmarks, tip_landmark, pip_landmark):
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
    if not hand_landmarks:
        return False
    landmarks = hand_landmarks.landmark
    ring_curled = not is_finger_extended(landmarks, mp_hands.HandLandmark.RING_FINGER_TIP,
                                         mp_hands.HandLandmark.RING_FINGER_PIP)
    pinky_curled = not is_finger_extended(landmarks, mp_hands.HandLandmark.PINKY_TIP, mp_hands.HandLandmark.PINKY_PIP)
    return ring_curled and pinky_curled


def is_high_five(hand_landmarks):
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
    press(key, duration)


# --- Main Loop ---

# --- Arrow TUI Initialization ---
base_up_arrow_map = [
    "       XX       ",
    "      XXXX      ",
    "     XXXXXX     ",
    "    XXXXXXXX    ",
    "   XXXXXXXXXX   ",
    "  XXXXXXXXXXXX  ",
    " XXXXXXXXXXXXXX ",
    "     XXXXXX     ",
    "     XXXXXX     ",
    "     XXXXXX     ",
]
base_dots = ascii_arrow.generate_base_dots(base_up_arrow_map)
up_dots, up_dims = ascii_arrow.rotate_dots(base_dots, 'up')
down_dots, down_dims = ascii_arrow.rotate_dots(base_dots, 'down')
left_dots, left_dims = ascii_arrow.rotate_dots(base_dots, 'left')
right_dots, right_dims = ascii_arrow.rotate_dots(base_dots, 'right')
# --- End Arrow TUI Initialization ---

layout = Layout()
layout.split(
    Layout(name="header", size=3),
    Layout(ratio=1, name="main"),
)
layout["main"].split_row(Layout(name="left"), Layout(name="right"))

WINDOW_NAME = 'CrossyVision'
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW_NAME, 768, 432)

# CRITICAL FIX: Set refresh rate to 60 fps
with Live(layout, screen=True, redirect_stderr=False, console=console, refresh_per_second=60) as live:
    while True:
        frame, results = processor.read()

        if frame is None or results is None:
            continue

        if MASK_FACE and face_masker:
            frame = face_masker.apply_mask(frame)

        current_time = time.time()
        frame_height, frame_width, _ = frame.shape

        finger_presence = {"LEFT_INDEX": False, "LEFT_MIDDLE": False, "RIGHT_INDEX": False, "RIGHT_MIDDLE": False}

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
                    left_hand_actions.clear()
                    right_hand_actions.clear()
                    last_key_press_time['SPACEBAR'] = current_time

        if not reset_gesture_detected:
            left_hand_in_position = False
            right_hand_in_position = False
            if results.multi_hand_landmarks and len(results.multi_hand_landmarks) == 2:
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

            both_hands_in_position = left_hand_in_position and right_hand_in_position

            if not controls_active:
                if both_hands_in_position:
                    activation_counter += 1
                    if activation_counter > ACTIVATION_CONFIDENCE_FRAMES:
                        controls_active = True
                        activation_counter = 0
                else:
                    activation_counter = 0
            else:
                if not both_hands_in_position:
                    deactivation_counter += 1
                    if deactivation_counter > DEACTIVATION_CONFIDENCE_FRAMES:
                        controls_active = False
                        deactivation_counter = 0
                else:
                    deactivation_counter = 0

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
                            mcp_pos, tip_pos = landmarks[mcp_id], landmarks[tip_id]
                            current_length = abs(mcp_pos.y - tip_pos.y)
                            if finger_id not in mcp_history:
                                mcp_history[finger_id] = deque(maxlen=CALIBRATION_FRAMES)
                                finger_lengths[finger_id] = deque(maxlen=CALIBRATION_FRAMES)

                            force_recalibration = False
                            if not previous_finger_is_above.get(finger_id, True):
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
                                if finger_id not in previous_finger_is_above:
                                    previous_finger_is_above[finger_id] = tip_pos.y < threshold_y

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
                                time_finger_went_below[finger_id] = current_time
                            elif not finger_was_previously_above and finger_is_currently_above:
                                time_finger_went_below[finger_id] = None

                            if finger_was_previously_above and not finger_is_currently_above:
                                if (current_time - last_key_press_time.get(key_action, 0)) > COOLDOWN_SECONDS:
                                    threading.Thread(target=press_key_threaded, args=(key_action, 0.05),
                                                     daemon=True).start()

                                    symbol = action_symbols.get(key_action, "?")
                                    if handedness == "Left":
                                        left_hand_actions.append(symbol)
                                    else:
                                        right_hand_actions.append(symbol)

                                    last_key_press_time[key_action] = current_time
                            previous_finger_is_above[finger_id] = finger_is_currently_above
        else:
            controls_active = False

        # --- Visual Feedback ---
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
                control_fingers_vis = {
                    "INDEX": mp_hands.HandLandmark.INDEX_FINGER_TIP,
                    "MIDDLE": mp_hands.HandLandmark.MIDDLE_FINGER_TIP
                }
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
                                      bar_color, -1)
                        cv2.circle(frame, tip_pos, 5, bar_color, 2)

        alpha = 0.4
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        # --- TUI Update ---
        finger_progress = {
            "LEFT_INDEX": 0.0, "LEFT_MIDDLE": 0.0,
            "RIGHT_INDEX": 0.0, "RIGHT_MIDDLE": 0.0
        }

        if results.multi_hand_landmarks:
            for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
                handedness = results.multi_handedness[i].classification[0].label
                landmarks = hand_landmarks.landmark
                for finger_name in ["INDEX", "MIDDLE"]:
                    finger_id = f"{handedness.upper()}_{finger_name}"
                    finger_presence[finger_id] = True

                    if finger_id in trigger_thresholds and finger_id in mcp_history and len(mcp_history[finger_id]) > 0:
                        avg_mcp_y = sum(mcp_history[finger_id]) / len(mcp_history[finger_id])
                        avg_length = sum(finger_lengths[finger_id]) / len(finger_lengths[finger_id])
                        y_extended_tip = avg_mcp_y - avg_length
                        y_activation = trigger_thresholds[finger_id]
                        y_current_tip = landmarks[mp_hands.HandLandmark[f"{finger_name}_FINGER_TIP"]].y
                        total_travel_dist = y_activation - y_extended_tip
                        current_travel = y_current_tip - y_extended_tip
                        progress_val = 0
                        if total_travel_dist > 1e-6:
                            progress_val = (current_travel / total_travel_dist) * 100

                        progress_val = min(100, max(0, progress_val))
                        finger_progress[finger_id] = progress_val / 100.0

        # --- Update Header Panel ---
        all_fingers_present = all(finger_presence.values())
        header_border_style = "bold green" if all_fingers_present else "blue"
        status_table = Table.grid(expand=True)
        status_table.add_column(justify="left")
        status_table.add_column(justify="center")
        status_table.add_column(justify="center")
        status_table.add_column(justify="right")
        labels_map = {"LEFT_INDEX": "L.INDEX", "LEFT_MIDDLE": "L.MIDDLE", "RIGHT_INDEX": "R.INDEX",
                      "RIGHT_MIDDLE": "R.MIDDLE"}
        finger_ids_order = ["LEFT_INDEX", "LEFT_MIDDLE", "RIGHT_INDEX", "RIGHT_MIDDLE"]
        styled_labels = []
        for finger_id in finger_ids_order:
            style = "bold green" if finger_presence.get(finger_id, False) else "grey30"
            styled_labels.append(Text(labels_map[finger_id], style=style))
        status_table.add_row(*styled_labels)
        layout["header"].update(
            Panel(status_table, title="CrossyVision", title_align="center", border_style=header_border_style))

        # --- Hand Panels with Improved Layout ---
        available_width = (console.width // 2) - 4
        max_symbols = max(1, available_width // 2)
        while len(left_hand_actions) > max_symbols:
            left_hand_actions.popleft()
        while len(right_hand_actions) > max_symbols:
            right_hand_actions.popleft()

        warm_fade = ["bold light_salmon1", "light_salmon1", "salmon1", "dark_orange3", "grey82", "grey66", "grey50",
                     "grey42", "grey37"]
        cool_fade = ["bold light_slate_grey", "light_slate_grey", "slate_grey1", "slate_grey2", "grey82", "grey66",
                     "grey50", "grey42", "grey37"]
        num_styles = len(warm_fade)

        left_action_text = Text(justify="right", no_wrap=True)
        left_history = list(left_hand_actions)
        for i, symbol in enumerate(left_history):
            action = symbol_to_action.get(symbol)
            fade_list = warm_fade if action in ["UP", "RIGHT"] else cool_fade
            age = len(left_history) - 1 - i
            style_index = min(age, num_styles - 1)
            left_action_text.append(symbol + " ", style=fade_list[style_index])

        right_action_text = Text(justify="left", no_wrap=True)
        right_history = list(right_hand_actions)
        for i, symbol in enumerate(reversed(right_history)):
            action = symbol_to_action.get(symbol)
            fade_list = warm_fade if action in ["UP", "RIGHT"] else cool_fade
            age = i
            style_index = min(age, num_styles - 1)
            right_action_text.append(symbol + " ", style=fade_list[style_index])

        # Create animated braille arrows based on finger progress
        # Left Hand (Note: LEFT_INDEX controls RIGHT key, LEFT_MIDDLE controls LEFT key)
        right_arrow_text = ascii_arrow.create_braille_arrow(right_dots, right_dims, finger_progress["LEFT_INDEX"],
                                                            'right', action_styles['RIGHT'])
        left_arrow_text = ascii_arrow.create_braille_arrow(left_dots, left_dims, finger_progress["LEFT_MIDDLE"], 'left',
                                                           action_styles['LEFT'])

        # Add vertical padding to align horizontal arrows with vertical ones
        vertical_padding = Text("\n\n")
        right_arrow_text = vertical_padding + right_arrow_text
        left_arrow_text = vertical_padding + left_arrow_text

        # Right Hand
        up_arrow_text = ascii_arrow.create_braille_arrow(up_dots, up_dims, finger_progress["RIGHT_INDEX"], 'up',
                                                         action_styles['UP'])
        down_arrow_text = ascii_arrow.create_braille_arrow(down_dots, down_dims, finger_progress["RIGHT_MIDDLE"],
                                                           'down', action_styles['DOWN'])

        # Arrange arrows side-by-side in columns for a cleaner layout
        left_arrow_columns = Columns([
            Align.center(left_arrow_text),
            Align.center(right_arrow_text)
        ], expand=True)
        left_panel_content = Group(
            left_arrow_columns,
            Text("\n"),  # Add a newline for spacing
            left_action_text
        )
        layout["left"].update(Panel(left_panel_content, title="Left Hand", border_style="blue"))

        right_arrow_columns = Columns([
            Align.center(up_arrow_text),
            Align.center(down_arrow_text)
        ], expand=True)
        right_panel_content = Group(
            right_arrow_columns,
            right_action_text
        )
        layout["right"].update(Panel(right_panel_content, title="Right Hand", border_style="blue"))

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