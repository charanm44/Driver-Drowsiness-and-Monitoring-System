from imutils.video import VideoStream
from imutils import face_utils
import cv2
import argparse
import imutils



print("[INFO] loading face classifier...")

face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

ap = argparse.ArgumentParser()
ap.add_argument("-w", "--webcam", type=int, default=0,
	help="index of webcam on system")
args = vars(ap.parse_args())

print("[INFO] starting video stream...")
vs = VideoStream(src=args["webcam"]).start()

while True:
	frame =  vs.read()
	frame = imutils.resize(frame, width=450)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	faces = face_detector.detectMultiScale(gray, 1.3, 5)
	for (x,y,w,h) in faces:
                cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)

	cv2.imshow("Frame", frame)
	key = cv2.waitKey(10) & 0xFF
	if key == 27:
		break

exit(0)
cv2.destroyAllWindows()
vs.stop()
