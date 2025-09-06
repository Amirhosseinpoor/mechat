import cv2
import numpy as np
from collections import deque
from ai_edge_litert.interpreter import Interpreter
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
import RPi.GPIO as GPIO

# ========================
# تنظیمات
# ========================
SEQUENCE_LENGTH = 30
MODEL_PATH = "/home/amir/Downloads/LSTM_best_model2.tflite"
HAND_MODEL_PATH = "/home/amir/Downloads/hand_landmarker.task"  # فایل .task
class_names = ["Thumbs Up", "Left Swipe", "Right Swipe", "Stop Gesture", "Thumbs Down"]

# پین‌های LED (BCM)
LED_PINS = [17, 27, 22, 23, 24]  # LED1..LED5

# نگاشت ژست به ایندکس LED
GESTURE_TO_LED = {
    "Thumbs Up": 0,     # LED1
    "Left Swipe": 1,    # LED2
    "Right Swipe": 2,   # LED3
    "Stop Gesture": 3,  # LED4
    "Thumbs Down": 4    # LED5
}

# آستانه‌ها برای پایداری پیش‌بینی
CONF_THRESH = 0.80          # حداقل اعتماد برای پذیرش ژست
STABLE_FRAMES = 6           # چند فریم متوالی باید یکسان باشند تا تریگر شود
TRIGGER_COOLDOWN = 1.2      # حداقل فاصله زمانی بین دو تریگر (ثانیه)

# ========================
# راه‌اندازی GPIO
# ========================
GPIO.setmode(GPIO.BCM)
for pin in LED_PINS:
    GPIO.setup(pin, GPIO.OUT, initial=GPIO.LOW)
led_states = [False] * len(LED_PINS)

def set_led(index, state: bool):
    led_states[index] = state
    GPIO.output(LED_PINS[index], GPIO.HIGH if state else GPIO.LOW)

def toggle_led(index):
    set_led(index, not led_states[index])

# ========================
# لود مدل TFLite با LiteRT
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
# وبکم + منطق تریگر ژست
# ========================
cap = cv2.VideoCapture(0)
sequence = deque(maxlen=SEQUENCE_LENGTH)

label_history = deque(maxlen=STABLE_FRAMES)
last_trigger_label = None
last_trigger_time = 0.0

prev_time = time.time()

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

        # پیش‌بینی
        if len(sequence) == SEQUENCE_LENGTH:
            input_data = np.expand_dims(sequence, axis=0).astype(np.float32)  # (1,30,63)
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
            preds = interpreter.get_tensor(output_details[0]['index'])[0]

            pred_class = int(np.argmax(preds))
            pred_label = class_names[pred_class]
            confidence = float(np.max(preds))

            # هدر نمایش
            overlay = frame.copy()
            cv2.rectangle(overlay, (5, 5), (380, 160), (0,0,0), -1)
            frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

            cv2.putText(frame, f"{pred_label} ({confidence:.2f})",
                        (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

            # رسم نمودار میله‌ای
            start_x, start_y = 15, 70
            for i, cls in enumerate(class_names):
                bar_length = int(preds[i] * 200)
                cv2.rectangle(frame, (start_x, start_y + i*18),
                                     (start_x + bar_length, start_y + i*18 + 12),
                                     (255, 0, 0), -1)
                cv2.putText(frame, f"{cls} {preds[i]:.2f}",
                            (start_x + 210, start_y + i*18 + 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1)

            # ---------------------------
            # دیبانس و تریگر ژست برای LED
            # ---------------------------
            if confidence is not None and confidence >= CONF_THRESH:
                label_history.append(pred_label)
            else:
                label_history.append(None)

            stable_label = None
            if len(label_history) == label_history.maxlen:
                unique = set(label_history)
                if len(unique) == 1 and None not in unique:
                    stable_label = next(iter(unique))  # تنها عنصر موجود

            now = time.time()
            if stable_label is not None:
                # اگر برچسب پایدار جدید است و کول‌داون رعایت شده
                if (stable_label != last_trigger_label) and ((now - last_trigger_time) >= TRIGGER_COOLDOWN):
                    if stable_label in GESTURE_TO_LED:
                        idx = GESTURE_TO_LED[stable_label]
                        toggle_led(idx)
                        last_trigger_label = stable_label
                        last_trigger_time = now
            else:
                # وقتی پایدار نیست، اجازه تریگر بعدی را آماده کنیم
                # اگر برچسب قبلی دیگر در تاریخچه نیست، بازآرمی کنیم
                if (last_trigger_label is not None) and (last_trigger_label not in label_history):
                    last_trigger_label = None

        # نمایش وضعیت LEDها
        led_overlay = frame.copy()
        # پس‌زمینه کوچک برای وضعیت LEDها
        cv2.rectangle(led_overlay, (5, 170), (200, 170 + 22*len(LED_PINS) + 10), (0,0,0), -1)
        frame = cv2.addWeighted(led_overlay, 0.5, frame, 0.5, 0)
        for i, state in enumerate(led_states):
            txt = f"LED{i+1}: {'ON' if state else 'OFF'}"
            color = (0,255,0) if state else (0,0,255)
            cv2.putText(frame, txt, (15, 190 + i*22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # FPS
        curr_time = time.time()
        fps = 1 / max(1e-6, (curr_time - prev_time))
        prev_time = curr_time
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, frame.shape[0]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

        cv2.imshow("Hand Gesture Recognition (LiteRT + Mediapipe + GPIO)", frame)

        # خروج
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if cv2.getWindowProperty("Hand Gesture Recognition (LiteRT + Mediapipe + GPIO)", cv2.WND_PROP_VISIBLE) < 1:
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    GPIO.cleanup()
    print("GPIO cleaned up. Bye!")
