import cv2
import dlib
import numpy as np
from scipy.spatial import distance as dist
import time
import requests
from datetime import datetime
import RPi.GPIO as GPIO
import drivers  # LCD driver

# ==== Config ====
EAR_THRESHOLD = 0.25
EAR_CONSEC_FRAMES = 15
API_URL = "https://abnormally-intense-grubworm.ngrok-free.app/api/data"  # Change this to your Flask server
driver_id = "driver001"
last_sent_time = time.time()
COOLDOWN_SECONDS = 10

# ==== GPIO Setup ====
BUZZER_PIN = 36
LED_PIN = 29
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BOARD)
GPIO.setup(BUZZER_PIN, GPIO.OUT)
GPIO.setup(LED_PIN, GPIO.OUT)

# ==== LCD Setup ====
display = drivers.Lcd()
display.lcd_display_string("Driver Monitor", 1)

# ==== Dlib Setup ====
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

def get_ear(shape):
    leftEye = np.array([(shape.part(i).x, shape.part(i).y) for i in range(36, 42)])
    rightEye = np.array([(shape.part(i).x, shape.part(i).y) for i in range(42, 48)])
    return (eye_aspect_ratio(leftEye) + eye_aspect_ratio(rightEye)) / 2.0

def send_drowsiness_data(driver_id, duration, frame):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    _, img_encoded = cv2.imencode('.jpg', frame)
    files = {'image': ('drowsy.jpg', img_encoded.tobytes(), 'image/jpeg')}
    data = {
        'driver_id': driver_id,
        'status': 'drowsy',
        'duration': int(duration),
        'timestamp': timestamp
    }
    try:
        response = requests.post(API_URL, files=files, data=data, timeout=3)
        print(f"? Data sent: {response.status_code}")
    except Exception as e:
        print(f"?? Upload failed: {e}")

def trigger_alert():
    GPIO.output(LED_PIN, GPIO.HIGH)
    GPIO.output(BUZZER_PIN, GPIO.HIGH)
    display.lcd_clear()
    display.lcd_display_string("?? Drowsy!", 1)

def stop_alert():
    GPIO.output(LED_PIN, GPIO.LOW)
    GPIO.output(BUZZER_PIN, GPIO.LOW)
    display.lcd_clear()
    display.lcd_display_string("Driver Normal", 1)

# ==== Main Loop ====
cap = cv2.VideoCapture(0)
COUNTER = 0
ALARM_ON = False
start_drowsy = None

print("? Running Drowsiness Detection...")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)
        status = "Normal"

        if len(faces) == 0:
            cv2.putText(frame, "Face Not Detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            display.lcd_clear()
            display.lcd_display_string("No Face Found", 1)
            stop_alert()
            ALARM_ON = False
            COUNTER = 0

        for face in faces:
            shape = predictor(gray, face)
            ear = get_ear(shape)

            cv2.putText(frame, f"EAR: {ear:.2f}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            if ear < EAR_THRESHOLD:
                COUNTER += 1
                if COUNTER >= EAR_CONSEC_FRAMES:
                    if not ALARM_ON:
                        ALARM_ON = True
                        start_drowsy = time.time()
                        trigger_alert()
                    status = "Drowsy"
            else:
                if ALARM_ON:
                    end_drowsy = time.time()
                    duration = end_drowsy - start_drowsy

                    now = time.time()
                    if now - last_sent_time > COOLDOWN_SECONDS:
                        send_drowsiness_data(driver_id, duration, frame)
                        last_sent_time = now

                    stop_alert()

                ALARM_ON = False
                COUNTER = 0

            # Draw rectangle around face
            x, y, w, h = face.left(), face.top(), face.width(), face.height()
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Status text overlay
        cv2.putText(frame, f"Status: {status}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255) if status == "Normal" else (0, 0, 255), 2)

        cv2.imshow("Driver Monitor", frame)
        if cv2.waitKey(1) == 27:  # ESC to quit
            break

except KeyboardInterrupt:
    print("? Interrupted by user")

finally:
    cap.release()
    cv2.destroyAllWindows()
    GPIO.cleanup()
    display.lcd_clear()
    display.lcd_display_string("System Stopped", 1)
