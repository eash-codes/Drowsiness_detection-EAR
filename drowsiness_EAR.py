import cv2 
import dlib
import numpy as np
from scipy.spatial import distance
import pygame
pygame.mixer.init()

def eye_aspect_ratio(eye_points):
    # Calculate distance between vertical eye landmarks 
    vertical_1 = distance.euclidean(eye_points[1], eye_points[5])
    vertical_2 = distance.euclidean(eye_points[2], eye_points[4])
    # Calculate distance between horizontal eye landmarks
    horizontal = distance.euclidean(eye_points[0], eye_points[3])
    
    # Compute EAR value
    ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
    
    return ear

# Initialize dlib's face detector and landmark predictor 
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Indices for eye landmarks from 68 points model 
LEFT_EYE_INDICES = [36, 37, 38, 39, 40, 41]
RIGHT_EYE_INDICES = [42, 43, 44, 45, 46, 47]

# Constants for EAR threshold and consecutive frame count 
EAR_THRESHOLD = 0.25
CONSEC_FRAMES = 20

# Initialize video capture and variables
cap = cv2.VideoCapture(0)
frame_counter = 0

# Load alert sound ONCE before the loop
alert_sound = pygame.mixer.Sound("AlertAudio.wav")
alert_played = False

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    
    for face in faces:
        landmarks = predictor(gray, face)
        landmarks_points = np.array([[p.x, p.y] for p in landmarks.parts()])
        
        left_eye_points = landmarks_points[LEFT_EYE_INDICES]
        right_eye_points = landmarks_points[RIGHT_EYE_INDICES]
        
        # Draw borders around eyes
        cv2.polylines(frame, [left_eye_points], True, (0, 255, 0), 1)
        cv2.polylines(frame, [right_eye_points], True, (0, 255, 0), 1)
        
        left_ear = eye_aspect_ratio(left_eye_points)
        right_ear = eye_aspect_ratio(right_eye_points)
        avg_ear = (left_ear + right_ear) / 2.0
        
        # Display EAR value on screen
        cv2.putText(frame, f"EAR: {avg_ear:.2f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        if avg_ear < EAR_THRESHOLD:
            frame_counter += 1
            if frame_counter >= CONSEC_FRAMES:
                cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                if not alert_played:
                    alert_sound.play(-1)
                    alert_played = True
        else:
            frame_counter = 0  # Reset counter when eyes are open
            if alert_played:
                alert_sound.stop()
                alert_played = False
    
    cv2.imshow("Drowsiness Detection", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit()