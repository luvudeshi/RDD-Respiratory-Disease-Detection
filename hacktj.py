import cv2
import dlib
import numpy as np
import serial
import time

arduino = serial.Serial("/dev/cu.usbmodem1101", 9600)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("/Users/sanjeevsubramanian/Desktop/CS Work/hacktj/shape_predictor_68_face_landmarks.dat")

cap = cv2.VideoCapture(1)
screen_width = int(cap.get(3))
screen_height = int(cap.get(4))

MIN_ANGLE = 0
MAX_ANGLE = 180
current_angle_x = 90
current_angle_y = 90

def detect_mouth(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    for face in faces:
        landmarks = predictor(gray, face)
        mouth_points = np.array([(landmarks.part(n).x, landmarks.part(n).y) for n in range(48, 68)])
        x_min, y_min = np.min(mouth_points, axis=0)
        x_max, y_max = np.max(mouth_points, axis=0)
        mouth_center_x = (x_min + x_max) // 2
        mouth_center_y = (y_min + y_max) // 2
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
        return mouth_center_x, mouth_center_y
    return None, None

def adjust_mic(mouth_x, mouth_y):
    global current_angle_x, current_angle_y
    if mouth_x is not None and mouth_y is not None:
        new_angle_x = int((mouth_x / screen_width) * (MAX_ANGLE - MIN_ANGLE)) + MIN_ANGLE
        new_angle_y = int((mouth_y / screen_height) * (MAX_ANGLE - MIN_ANGLE)) + MIN_ANGLE
        new_angle_x = max(MIN_ANGLE, min(MAX_ANGLE, new_angle_x))
        new_angle_y = max(MIN_ANGLE, min(MAX_ANGLE, new_angle_y))
        if abs(new_angle_x - current_angle_x) > 2 or abs(new_angle_y - current_angle_y) > 2:
            print(f"Rotating mic to X: {new_angle_x}°, Y: {new_angle_y}°")
            arduino.write(f"{new_angle_x},{new_angle_y}\n".encode())
            current_angle_x = new_angle_x
            current_angle_y = new_angle_y

def main():
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        mouth_x, mouth_y = detect_mouth(frame)
        adjust_mic(mouth_x, mouth_y)
        cv2.imshow("Mouth Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()