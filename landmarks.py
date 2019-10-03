from imutils.video import VideoStream
from imutils import face_utils
#from threading import Thread
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2



ap = argparse.ArgumentParser()
ap.add_argument("-w", "--webcam", type=int, default=0,
	help="index of webcam on system")
args = vars(ap.parse_args())

COUNTER = 0
#ALARM_ON = False

print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(sMou,eMou) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]
(sNos,eNos) = face_utils.FACIAL_LANDMARKS_IDXS["nose"]

print("[INFO] starting video stream...")

vs = VideoStream(src=args["webcam"]).start()

print("[INFO] SetUp loading...")
print("[INFO} Please keep ur HEAD STRAIGHT toward's camera")
time.sleep(5)
print("[INFO] Taking values please wait...")


count = 0
count1 = 0
while True:
	frame =  vs.read()
	frame = imutils.resize(frame, width=450)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	rects = detector(gray, 0)

	for rect in rects:
		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)
		leftEye = shape[lStart:lEnd]
		rightEye = shape[rStart:rEnd]
		mouth = shape[sMou:eMou]
		nose = shape[sNos:eNos]
		
		leftEyeHull = cv2.convexHull(leftEye)
		rightEyeHull = cv2.convexHull(rightEye)
		mouthHull = cv2.convexHull(mouth)
		noseHull = cv2.convexHull(nose)
		cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
		cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
		cv2.drawContours(frame,[mouthHull],-1,(0,255,0),1)
		cv2.drawContours(frame, [noseHull], -1, (0, 255, 0), 1)

	cv2.imshow("Frame", frame)
	key = cv2.waitKey(10) & 0xFF
	if key == 27:
		break

cv2.destroyAllWindows()
vs.stop()
exit(0)
