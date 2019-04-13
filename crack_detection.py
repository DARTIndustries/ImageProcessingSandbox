# import cv2
# import numpy as np

# frame = cv2.imread('static_crack.PNG')  # read from file
# cv2.namedWindow("frame", 0)
# cv2.resizeWindow("frame", 800,600)
# while True:
#     height = len(frame)
#     width = len(frame[0])
#     imgray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # convert to black and white
#     ret, thresh = cv2.threshold(imgray, 127, 255, 0)  # mark pixels between color values 127-255
#     thresh = cv2.bitwise_not(thresh)  # invert image
#
#     cv2.imshow('frame1', thresh)
#
#     contours, hierarchy = cv2.findContours(thresh,
#                                            cv2.RETR_TREE,  # allows finding nested contours
#                                            cv2.CHAIN_APPROX_NONE)  # gets ALL contour points (e.g. doesn't get only corners)
#     contours = sorted(contours, key=cv2.contourArea, reverse=True)[:1]  # gets largest contour????
#     cv2.drawContours(frame,contours,0,(0,255,0), thickness=cv2.FILLED)
#     cix = width//2
#     ciy = height//2
#
#     cv2.imshow('frame',frame)
#     cv2.imshow('frame0', imgray)
#
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# frame.release()
# cv2.destroyAllWindows()


# Red line: 1.8-1.9 cm wide
#
# Blue line: 1.8-1.9 cm wide
#            8-20 cm long
#            Parallel to red line
#
# Square: 30 x 30 cm

# import the necessary packages
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import numpy as np
import argparse
import imutils
import cv2

img = cv2.imread('crack_horiz_above.PNG')  # read from file
# cv2.namedWindow("crack", 0)
# cv2.resizeWindow("crack", 800,600)

IMG_HEIGHT = len(img)
IMG_WIDTH = len(img[0])

def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

# load the image, convert it to grayscale, and blur it slightly
image = cv2.imread('crack_horiz_above.PNG')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (7, 7), 0)

# perform edge detection, then perform a dilation + erosion to
# close gaps in between object edges
edged = cv2.Canny(gray, 50, 100)
edged = cv2.dilate(edged, None, iterations=1)
edged = cv2.erode(edged, None, iterations=1)

edged = cv2.copyMakeBorder(edged, top=2, bottom=2, left=2, right=2, borderType=cv2.BORDER_CONSTANT, value=[255,255,255])

# find contours in the edge map
cnts = cv2.findContours(edged.copy(), cv2.RETR_TREE,
                        cv2.CHAIN_APPROX_NONE)
cnts = imutils.grab_contours(cnts)

# sort the contours from left-to-right and initialize the
# 'pixels per metric' calibration variable
# (cnts, _) = contours.sort_contours(cnts)

# sort largest to smallest
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

# loop through contours to find a red one
for i, c in enumerate(cnts):
    br = cv2.boundingRect(cnts[i])
    c1 = cv2.mean(br)
    # c1 = cv2.mean(br)
    # mask = np.zeros(gray.shape, np.uint8)
    print(c1)

pixelsPerMetric = 37/1.85 # 37 px = 1.85 cm

# loop over the contours individually
for c in cnts:

    # if the contour is not sufficiently large, ignore it
    if cv2.contourArea(c) > 40*500:
        continue

    # compute the rotated bounding box of the contour
    orig = image.copy()
    box = cv2.minAreaRect(c)
    box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
    box = np.array(box, dtype="int")

    # order the points in the contour such that they appear
    # in top-left, top-right, bottom-right, and bottom-left
    # order, then draw the outline of the rotated bounding
    # box
    box = perspective.order_points(box)
    cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)

    # loop over the original points and draw them
    for (x, y) in box:
        cv2.circle(orig, (int(x), int(y)), 5, (0, 0, 255), -1)

    # unpack the ordered bounding box, then compute the midpoint
    # between the top-left and top-right coordinates, followed by
    # the midpoint between bottom-left and bottom-right coordinates
    (tl, tr, br, bl) = box
    (tltrX, tltrY) = midpoint(tl, tr)
    (blbrX, blbrY) = midpoint(bl, br)

    # compute the midpoint between the top-left and top-right points,
    # followed by the midpoint between the top-righ and bottom-right
    (tlblX, tlblY) = midpoint(tl, bl)
    (trbrX, trbrY) = midpoint(tr, br)

    # draw the midpoints on the image
    cv2.circle(orig, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
    cv2.circle(orig, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
    cv2.circle(orig, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
    cv2.circle(orig, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)

    # draw lines between the midpoints
    cv2.line(orig, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),
             (255, 0, 255), 2)
    cv2.line(orig, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),
             (255, 0, 255), 2)

    # compute the Euclidean distance between the midpoints
    dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
    dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

    # if the pixels per metric has not been initialized, then
    # compute it as the ratio of pixels to supplied metric
    # (in this case, inches)
    if True: #pixelsPerMetric is None:
        #pixelsPerMetric = dB / 500#IMG_WIDTH

        # compute the size of the object
        dimA = dA / pixelsPerMetric
        dimB = dB / pixelsPerMetric

        print(dA, " : ", dB)
        print(dimA, " : ", dimB)

        # draw the object sizes on the image
        cv2.putText(orig, "{:.3f}cm".format(dimA),
                    (int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (0, 0, 0), 2)
        cv2.putText(orig, "{:.3f}cm".format(dimB),
                    (int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (0, 0, 0), 2)

        #cv2.drawContours(orig, cnts, -1, (0, 255, 0), 1)

        # show the output image
        cv2.imshow("Image", orig)
        cv2.waitKey(0)
cv2.waitKey(0)