import cv2
import dlib
import numpy as np
import time
import serial

cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("can't access the camera")
    exit()
try:
    arduino = serial.Serial('/dev/ttyUSB0', 9600, timeout=1)
    time.sleep(2)
except serial.SerialException:
    print("can't connect to arduino")
    arduino = None
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("/Users/sanjeevsubramanian/Desktop/CS Work/hacktj/shape_predictor_68_face_landmarks.dat")
DROOP_THRESHOLD = 8
FRAME_COUNT_THRESHOLD = 30
COOLDOWN_TIME = 5
ALERT_SUSTAIN_TIME = 2
droop_counter = 0
last_droop_time = 0
droop_active = False 

def detect_droopiness(frame):
    global droop_counter, last_droop_time, droop_active
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    for face in faces:
        landmarks = predictor(gray, face)
        left_eye_y = landmarks.part(36).y
        right_eye_y = landmarks.part(45).y
        left_mouth_y = landmarks.part(48).y
        right_mouth_y = landmarks.part(54).y
        eye_asymmetry = abs(left_eye_y - right_eye_y)
        mouth_asymmetry = abs(left_mouth_y - right_mouth_y)
        eye_droop_detected = eye_asymmetry > 3
        mouth_droop_detected = mouth_asymmetry > 5
        if time.time() - last_droop_time > COOLDOWN_TIME:
            if eye_droop_detected or mouth_droop_detected:
                droop_counter += 1
            else:
                droop_counter = max(0, droop_counter - 1)
            if droop_counter >= FRAME_COUNT_THRESHOLD:
                print("possible droop")
                cv2.putText(frame, "Facial Droop Detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                if arduino:
                    arduino.write(b'1')

                last_droop_time = time.time()
                droop_counter = 0
                droop_active = True
            else:
                droop_active = False
        if droop_active and (time.time() - last_droop_time < ALERT_SUSTAIN_TIME):
            cv2.putText(frame, "Facial Droop Detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        else:
            droop_active = False
        cv2.circle(frame, (landmarks.part(36).x, left_eye_y), 3, (0, 255, 0), -1)
        cv2.circle(frame, (landmarks.part(45).x, right_eye_y), 3, (0, 255, 0), -1)
        cv2.circle(frame, (landmarks.part(48).x, left_mouth_y), 3, (0, 0, 255), -1)
        cv2.circle(frame, (landmarks.part(54).x, right_mouth_y), 3, (0, 0, 255), -1)
    return frame

def main():
    while True:
        ret, frame = cap.read()
        if not ret:
            print("no frame")
            break
        frame = detect_droopiness(frame)
        cv2.imshow("🩺 Real-Time Health Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()