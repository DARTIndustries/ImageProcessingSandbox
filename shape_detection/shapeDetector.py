import cv2

# https://docs.opencv.org/2.4/modules/imgproc/doc/structural_analysis_and_shape_descriptors.html

# load image and convert to grayscale
img = cv2.imread('shapes.png')  # benthic species screen shot from manual
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# identify pixels within (127, 255) intensity range (shapes)
_, thresh = cv2.threshold(gray, 127, 255, 1)

# find shapes
contours, _ = cv2.findContours(thresh, 1, 2)
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
        squares += 1
        # print('Rectangle')
        cv2.drawContours(img, [c], 0, (0, 0, 255), -1)
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
out = cv2.imread('output.png')
# add text to image to format output like in manual
cv2.putText(out, str(circles), (150,100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
cv2.putText(out, str(triangles), (150,230), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
cv2.putText(out, str(lines), (150,370), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
cv2.putText(out, str(squares), (150,510), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
cv2.imshow('out', out)

cv2.waitKey(0)  # hit esc to exit
cv2.destroyAllWindows()
