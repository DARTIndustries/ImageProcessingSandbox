import cv2
import numpy as np

img = cv2.imread('static_crack.PNG')  # read from file
cv2.namedWindow("frame", 0)
cv2.resizeWindow("frame", 800,600)
while (True):
    frame = img
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

    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

img.release()
cv2.destroyAllWindows()

