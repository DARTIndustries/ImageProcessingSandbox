import cv2
import numpy as np
import math

cap = cv2.VideoCapture('line_following_video.MOV')  # read from file
cv2.namedWindow("frame", 0)
cv2.resizeWindow("frame", 800,600)
while (cap.isOpened()):
	ret, frame = cap.read()
	height = len(frame)
	width = len(frame[0])
	imgray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # convert to black and white
	ret, thresh = cv2.threshold(imgray, 127, 255, 0)  # mark pixels between color values 127-255
	thresh = cv2.bitwise_not(thresh)  # invert image
	contours, hierarchy = cv2.findContours(thresh,
										   cv2.RETR_TREE,  # allows finding nested contours
										   cv2.CHAIN_APPROX_NONE)  # gets ALL contour points (e.g. doesn't get only corners)
	contours = sorted(contours, key=cv2.contourArea, reverse=True)[:1]  # gets largest contour????
	cv2.drawContours(frame,contours,0,(0,255,0), thickness=cv2.FILLED)
	cix = width//2
	ciy = height//2
	for cnt in contours:
		M = cv2.moments(cnt)

		# find center of mass
		cx = int(M["m10"] / M["m00"])
		cy = int(M["m01"] / M["m00"])

		cv2.line(frame,(cix, ciy),(cx,cy),(255,0,0),5) # draw line from center of image to center of mass

		# draw box around the part of the image with the line (contour) inside
		rect = cv2.minAreaRect(cnt)
		box = cv2.boxPoints(rect)
		box = np.int0(box)
		cv2.drawContours(frame,[box],0,(255,0,0),2)

		print("cx " + str(cx) + ", cy " + str(cy))
		print(math.degrees(math.atan2(cy-ciy, cx-cix))*-1)

		# if cx > 1000 and cy >= 550: #going forwards, turn right
		# 	print("Turn Right!")
		# elif (cx > 950 and cx < 1100) and cy < 550:
		# 	print("Turn Left!")
		# else: #cx < 1000 and cy < 500
		# 	print("On Track!")


		# if cx >= 900:
		# 	print("Turn Left!")
		#
		# if cx >= 1200:
		# 	print("Turn Right")

	cv2.imshow('frame',frame)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()

