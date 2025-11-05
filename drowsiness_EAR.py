import cv2 
import dlib
import numpy as np
from scipy.spatial import distance

def eye_aspect_ratio(eye_points):
    #calculate distance between vertical eyes landmarks 
    vertical_1 = distance.euclidean(eye_points[1],eye_points[5])
    vertical_2 = distance.euclidean(eye_points[2],eyes_points[4])
    # calc distance between horizontal eyes landmarks
    horizontal = distance.euclidean(eye_points[0],eye_points[3])

    #compute EAR value
    ear = (vertical_1 + vertical_2)/(2.0 * horizontal)

    return ear