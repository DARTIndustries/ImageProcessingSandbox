import cv2
import numpy as np

cap = cv2.VideoCapture('line_following_video.MOV')

while(cap.isOpened()):
	ret, frame = cap.read()
	imgray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	ret, thresh = cv2.threshold(imgray, 127, 255, 0)
	contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
	for cnt in contours:
		rect = cv2.minAreaRect(cnt)
		box = cv2.boxPoints(rect)
		box = np.int0(box)
		cv2.drawContours(imgray,[box],0,(0,0,255),2)
	cv2.imshow('frame',imgray)

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()

