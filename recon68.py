#!/opt/virtualenv/computer_vision_ORIGINAL/bin/python3
##!/opt/virtualenv/computer_vision/bin/python3

# 20210915
# https://tsurugi-linux.org
# by Visi@n
#
# LICENSE
# THIS SCRIPT USE FACE_RECOGNITION LIBRARY [https://github.com/ageitgey/face_recognition/blob/master/LICENSE]
# THIS SCRIPT HAS BEEN MODIFIED BY Antonio 'Visi@n' Broi [antonio@tsurugi-linux.org] and it's licensed under the MIT License
# Special thanks to Adam Ageitgey [https://adamgeitgey.com] the creator of face_recognition and to all Python community

import os
import numpy as np
import cv2
import time
import imutils
from imutils.video import VideoStream
from imutils import face_utils
import datetime
import argparse
import dlib
import face_recognition

# ============================================================================

def draw_border(frame, pt1, pt2, color, thickness, r, d):
    x1,y1 = pt1
    x2,y2 = pt2

    # Top left
    cv2.line(frame, (x1 + r, y1), (x1 + r + d, y1), color, thickness)
    cv2.line(frame, (x1, y1 + r), (x1, y1 + r + d), color, thickness)
    cv2.ellipse(frame, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)

    # Top right
    cv2.line(frame, (x2 - r, y1), (x2 - r - d, y1), color, thickness)
    cv2.line(frame, (x2, y1 + r), (x2, y1 + r + d), color, thickness)
    cv2.ellipse(frame, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)

    # Bottom left
    cv2.line(frame, (x1 + r, y2), (x1 + r + d, y2), color, thickness)
    cv2.line(frame, (x1, y2 - r), (x1, y2 - r - d), color, thickness)
    cv2.ellipse(frame, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)

    # Bottom right
    cv2.line(frame, (x2 - r, y2), (x2 - r - d, y2), color, thickness)
    cv2.line(frame, (x2, y2 - r), (x2, y2 - r - d), color, thickness)
    cv2.ellipse(frame, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)

# ============================================================================
 
 
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", default="face_recognition_models\modelsshape_predictor_68_face_landmarks.dat",#required=True,
	help="path to facial landmark predictor")
ap.add_argument("-i", "--inputimage", required=True,#required=True,
	help="input image")
ap.add_argument("-o", "--outputimage", required=True,#required=True,
	help="output image")
	
#ap.add_argument("-r", "--picamera", type=int, default=-1,
#	help="whether or not the Raspberry Pi camera should be used")
args = vars(ap.parse_args())
 
# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

printa = (args["inputimage"])
frame = cv2.imread(printa)

##################
# Resize frame of video to 1/4 size for faster face detection processing
small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

# Find all the faces and face encodings in the current frame of video
face_locations = face_recognition.face_locations(small_frame)
	
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# detect faces in the grayscale frame
rects = detector(gray, 0)

# loop over the face detections
for rect in rects:
	# determine the facial landmarks for the face region, then
	# convert the facial landmark (x, y)-coordinates to a NumPy
	# array
	shape = predictor(gray, rect)
	shape = face_utils.shape_to_np(shape)

#################
# loop over the face detections
for rect in rects:
	# determine the facial landmarks for the face region, then
	# convert the facial landmark (x, y)-coordinates to a NumPy
	# array
	shape = predictor(gray, rect)
	shape = face_utils.shape_to_np(shape)

	# loop over the (x, y)-coordinates for the facial landmarks
	# and draw them on the image
	for (x, y) in shape:
		cv2.circle(frame, (x, y), 1, (255, 0, 0), 2)
for top, right, bottom, left in face_locations:
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
		top *= 4
		right *= 4
		bottom *= 4
		left *= 4

        # Extract the region of the image that contains the face
		face_image = frame[top:bottom, left:right]
		draw_border(frame, (left, top), (right, bottom), (255, 0, 0), 3, 10, 20)	  

#fil = cv2.open(file, "rb")
cv2.imshow('recognition key points', frame)
cv2.imwrite((args["outputimage"]), frame)
time.sleep(1.0)
cv2.waitKey(0)
#cv2.destroyAllWindows()
