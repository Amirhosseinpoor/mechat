import cv2
import numpy as np
from collections import deque
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time

# ========================
# Use TensorFlow Lite Interpreter
# ========================
try:
    # Preferred: TensorFlow installed (tf 2.x)
    from tensorflow.lite import Interpreter
    print("✅ Using TensorFlow Lite Interpreter (from tensorflow.lite)")
except Exception:
    # Fallback: lightweight tflite_runtime package
    from tflite_runtime.interpreter import Interpreter
    print("✅ Using TensorFlow Lite Interpreter (from tflite_runtime)")

# ========================
# Raspberry Pi GPIO (BCM)
# ========================
import RPi.GPIO as GPIO
GPIO.setmode(GPIO.BCM)

# Gesture -> GPIO mapping (BCM)
LED_PINS = {
    "thumbs up": 17,
    "thumbs down": 27,
    "left swipe": 22,     # "Left swing" synonym handled below
    "right swipe": 23,    # "Right Swing" synonym handled below
    "stop/resume": 24,    # "Stop Gesture" synonym handled below
}

# Prepare pins
ALL_PINS = list(LED_PINS.values())
for p in ALL_PINS:
    GPIO.setup(p, GPIO.OUT, initial=GPIO.LOW)

# Helper to turn only one LED on (others off)
def set_exclusive_led(pin_on):
    for p in ALL_PINS:
        GPIO.output(p, GPIO.HIGH if p == pin_on else GPIO.LOW)

# ========================
# تنظیمات
# ========================
SEQUENCE_LENGTH = 30
MODEL_PATH = "/home/amir/Downloads/LSTM_best_model2.tflite"
HAND_MODEL_PATH = "/home/amir/Downloads/hand_landmarker.task"  # فایل .task
class_names = ["Thumbs Up", "Left Swipe", "Right Swipe", "Stop Gesture", "Thumbs Down"]

# Confidence threshold & debounce
PRED_THRESHOLD = 0.70
DEBOUNCE_SEC = 0.8     # avoid repeated triggers when same gesture is held
TOGGLE_DEBOUNCE_SEC = 1.2  # slightly longer debounce for Stop/Resume toggle

# ========================
# Load TFLite model (TensorFlow Lite)
# ========================
interpreter = Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print("✅ TFLite model loaded!")

# ========================
# MediaPipe HandLandmarker setup
# ========================
base_options = python.BaseOptions(model_asset_path=HAND_MODEL_PATH)
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
hand_landmarker = vision.HandLandmarker.create_from_options(options)

# ========================
# تابع استخراج لندمارک
# ========================
def extract_landmarks(image):
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
    result = hand_landmarker.detect(mp_image)

    if result.hand_landmarks:
        lm_list = []
        for lm in result.hand_landmarks[0]:
            lm_list.extend([lm.x, lm.y, lm.z])
        return lm_list, result.hand_landmarks[0]
    else:
        return [0]*63, None

# ========================
# تابع رسم اسکلت دست
# ========================
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),
    (0,17)
]

def draw_hand(frame, landmarks):
    h, w, _ = frame.shape
    for idx, lm in enumerate(landmarks):
        x, y = int(lm.x * w), int(lm.y * h)
        cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)

    for conn in HAND_CONNECTIONS:
        x1, y1 = int(landmarks[conn[0]].x * w), int(landmarks[conn[0]].y * h)
        x2, y2 = int(landmarks[conn[1]].x * w), int(landmarks[conn[1]].y * h)
        cv2.line(frame, (x1,y1), (x2,y2), (0,255,255), 2)

# ========================
# Label normalization & synonyms
# ========================
def normalize_label(label: str) -> str:
    """Map model labels/synonyms to our LED_PINS keys."""
    l = label.strip().lower()
    if l in ("thumbs up", "thumb up", "thumps up"):
        return "thumbs up"
    if l in ("thumbs down", "thumb down"):
        return "thumbs down"
    if l in ("left swipe", "left swing", "left"):
        return "left swipe"
    if l in ("right swipe", "right swing", "right"):
        return "right swipe"
    if l in ("stop gesture", "stop", "stop/resume", "resume"):
        return "stop/resume"
    return l

# ========================
# وبکم + Inference + GPIO control
# ========================
cap = cv2.VideoCapture(0)
sequence = deque(maxlen=SEQUENCE_LENGTH)

prev_time = time.time()
last_trigger_time = 0.0
last_label_key = None

# Track toggle state for Stop/Resume LED
stop_led_state = False  # False=OFF, True=ON

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        landmarks, hand_landmarks = extract_landmarks(rgb_frame)
        sequence.append(landmarks)

        # رسم دست
        if hand_landmarks:
            draw_hand(frame, hand_landmarks)

        pred_label, confidence, preds = None, None, None
        selected_pin = None

        # پیش‌بینی
        if len(sequence) == SEQUENCE_LENGTH:
            input_data = np.expand_dims(sequence, axis=0).astype(np.float32)  # (1,30,63)
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
            preds = interpreter.get_tensor(output_details[0]['index'])[0]

            pred_class = int(np.argmax(preds))
            pred_label = class_names[pred_class]
            confidence = float(np.max(preds))

            # Draw overlay
            overlay = frame.copy()
            cv2.rectangle(overlay, (5, 5), (360, 140), (0,0,0), -1)
            frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

            cv2.putText(frame, f"{pred_label} ({confidence:.2f})",
                        (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

            # Bar chart
            start_x, start_y = 15, 70
            for i, cls in enumerate(class_names):
                bar_length = int(preds[i] * 200)
                cv2.rectangle(frame, (start_x, start_y + i*20),
                                     (start_x + bar_length, start_y + i*20 + 15),
                                     (255, 0, 0), -1)
                cv2.putText(frame, f"{cls} {preds[i]:.2f}",
                            (start_x + 210, start_y + i*20 + 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

            # ========================
            # GPIO control based on prediction
            # ========================
            now = time.time()
            label_key = normalize_label(pred_label)

            if confidence >= PRED_THRESHOLD and label_key in LED_PINS:
                pin = LED_PINS[label_key]
                selected_pin = pin  # for UI text

                # Debounce timing
                debounce = TOGGLE_DEBOUNCE_SEC if label_key == "stop/resume" else DEBOUNCE_SEC
                should_trigger = (label_key != last_label_key) or ((now - last_trigger_time) >= debounce)

                if should_trigger:
                    if label_key == "stop/resume":
                        # Toggle the stop/resume LED only; leave others as-is (off)
                        stop_led_state = not stop_led_state
                        GPIO.output(pin, GPIO.HIGH if stop_led_state else GPIO.LOW)
                        # Also turn others off to make state obvious:
                        for p in ALL_PINS:
                            if p != pin:
                                GPIO.output(p, GPIO.LOW)
                    else:
                        # Turn on only the LED for this gesture; others off
                        set_exclusive_led(pin)

                    last_trigger_time = now
                    last_label_key = label_key

        # FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0.0
        prev_time = curr_time

        # UI footer
        footer = f"FPS: {fps:.1f}"
        if pred_label is not None:
            footer += " | GPIO: "
            lk = normalize_label(pred_label)
            if lk in LED_PINS:
                footer += f"{LED_PINS[lk]}"
                if lk == "stop/resume":
                    footer += f" (toggle {'ON' if stop_led_state else 'OFF'})"
            else:
                footer += "—"
        cv2.putText(frame, footer, (10, frame.shape[0]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

        cv2.imshow("Hand Gesture Recognition (TensorFlow Lite + Mediapipe)", frame)

        # --- graceful close ---
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if cv2.getWindowProperty("Hand Gesture Recognition (TensorFlow Lite + Mediapipe)", cv2.WND_PROP_VISIBLE) < 1:
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    # Turn everything off and cleanup GPIO
    for p in ALL_PINS:
        GPIO.output(p, GPIO.LOW)
    GPIO.cleanup()
