import cv2
import math
import os

# https://docs.opencv.org/2.4/modules/imgproc/doc/structural_analysis_and_shape_descriptors.html

# load image
photo = "realshapes1.jpg"
imgpath = os.path.join("images", photo)
img = cv2.imread(imgpath)  # benthic species screen shot from manual

# resize image if necessary
img = cv2.resize(img, (0,0), fx=0.15, fy=0.15)

# convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# blur
# gray = cv2.blur(gray, (7,7)) # blur image
# gray = cv2.blur(gray, (10, 10), 0)
# gray = cv2.GaussianBlur(gray, (9,9), 0)
# gray = cv2.medianBlur(gray, 5) # not so good
gray = cv2.bilateralFilter(gray,9,75,75)

# find edges
edges = cv2.Canny(gray, 85, 100)

# # identify pixels within (127, 255) intensity range (shapes)
# # _, thresh = cv2.threshold(gray, 127, 255, 1)
# _, thresh = cv2.threshold(edges, 127, 255, 1)
# cv2.imshow("thresh", thresh)
# thresh = cv2.GaussianBlur(thresh, (9,9), 0)
# cv2.imshow("thresh2", thresh)

# find shapes
# contours, _ = cv2.findContours(thresh, 1, 2)
contours, _ = cv2.findContours(edges, 1, 2)
print(len(contours), ' Shapes identified')

triangles, squares, circles, lines = (0, 0, 0, 0)  # used to count shapes

# find the number of sides in each shape
for c in contours:
    # Approximates a polygonal curve(s) with the specified precision
    # epsilon – approximation accuracy, max distance between the original curve and its approximation.
    approx = cv2.approxPolyDP(c, 0.02*cv2.arcLength(c, closed=True), closed=True)
    # print('Sides found:', len(approx), end=" -> ")
    # draw shapes in colors over original image
    if len(approx) < 3:
        lines += 1
        # print('Line')
        cv2.drawContours(img, [c], 0, (255, 0, 0), -1)
    elif len(approx) == 3:
        triangles += 1
        # print('Triangle')
        cv2.drawContours(img, [c], 0, (0, 255, 0), -1)
    elif len(approx) == 4:
        # could be either a square or line -> check side lengths
        # square: if the side lengths are within 20 pixels (might need to tune)
        # line: any other 4 sided shape (rectangle)
        side1 = math.sqrt((approx[0][0][0] - approx[1][0][0]) ** 2 + (approx[0][0][1] - approx[1][0][1]) ** 2)
        side2 = math.sqrt((approx[1][0][0] - approx[2][0][0]) ** 2 + (approx[1][0][1] - approx[2][0][1]) ** 2)
        if abs(side1 - side2) < 20:
            squares += 1
            # print('Square')
            cv2.drawContours(img, [c], 0, (0, 0, 255), -1)
        else:
            lines += 1
            # print('Line (rectangle)')
            cv2.drawContours(img, [c], 0, (255, 0, 0), -1)
    elif len(approx) >= 5:  # any more sides than square must be a circle
        circles += 1
        # print('Circle')
        cv2.drawContours(img, [c], 0, (255, 0, 255), -1)

# print('----------------------')
# print(lines, ' lines')
# print(triangles, ' triangles')
# print(squares, ' squares')
# print(circles, ' circles')

cv2.imshow('original grayscale', gray)
cv2.imshow('colored shapes', img)

# load base output image for display of species found
outpath = os.path.join("images", "output.png")
out = cv2.imread(outpath)
# add text to image to format output like in manual
cv2.putText(out, str(circles), (150,100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
cv2.putText(out, str(triangles), (150,230), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
cv2.putText(out, str(lines), (150,370), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
cv2.putText(out, str(squares), (150,510), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
cv2.imshow('out', out)

cv2.waitKey(0)  # hit esc to exit
cv2.destroyAllWindows()
