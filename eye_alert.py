import cv2
import mediapipe as mp
import numpy as np
import pygame

# -------- ALARM SETUP --------
pygame.mixer.init()
pygame.mixer.music.load("alarm2.wav")  # loud alarm sound

def start_alarm():
    if not pygame.mixer.music.get_busy():
        pygame.mixer.music.play(-1)  # loop

def stop_alarm():
    if pygame.mixer.music.get_busy():
        pygame.mixer.music.stop()

# -------- MEDIAPIPE --------
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(refine_landmarks=True)

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

def eye_aspect_ratio(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    return (A + B) / (2.0 * C)

cap = cv2.VideoCapture(0)

THRESHOLD = 0.15
FRAMES = 75

counter = 0
alarm_on = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)

    if result.multi_face_landmarks:
        for face in result.multi_face_landmarks:
            h, w, _ = frame.shape

            left_eye = np.array([[int(face.landmark[i].x * w),
                                  int(face.landmark[i].y * h)] for i in LEFT_EYE])
            right_eye = np.array([[int(face.landmark[i].x * w),
                                   int(face.landmark[i].y * h)] for i in RIGHT_EYE])

            ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2

            cv2.putText(frame, f"EAR: {ear:.2f}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

            if ear < THRESHOLD:
                counter += 1
                if counter >= FRAMES:
                    alarm_on = True
            else:
                counter = 0
                alarm_on = False

    # 🔊 ALARM CONTROL
    if alarm_on:
        start_alarm()
        cv2.putText(frame, "WAKE UP !!!", (100, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 5)
    else:
        stop_alarm()

    cv2.imshow("Drowsiness Alert System", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

stop_alarm()
cap.release()
cv2.destroyAllWindows()
