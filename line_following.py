import cv2
import numpy as np

cap = cv2.VideoCapture('line_following_video.MOV')
cv2.namedWindow("frame", 0);
cv2.resizeWindow("frame", 800,600);
while(cap.isOpened()):
	ret, frame = cap.read()
	height = len(frame)
	width = len(frame[0])
	imgray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	ret, thresh = cv2.threshold(imgray, 127, 255, 0)
	thresh = cv2.bitwise_not(thresh)
	contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
	contours = sorted(contours, key=cv2.contourArea, reverse=True)[:1]
	cv2.drawContours(frame,contours,0,(0,255,0), thickness=cv2.FILLED)
	cix = width//2
	ciy = height//2
	for cnt in contours:
		M = cv2.moments(cnt)
		cx = int(M["m10"] / M["m00"])
		cy = int(M["m01"] / M["m00"])
		cv2.line(frame,(cix, ciy),(cx,cy),(255,0,0),5)
		rect = cv2.minAreaRect(cnt)
		box = cv2.boxPoints(rect)
		box = np.int0(box)
		cv2.drawContours(frame,[box],0,(255,0,0),2)
	cv2.imshow('frame',frame)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()

