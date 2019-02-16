import cv2

# capture frames from a camera
cap = cv2.VideoCapture(0)

# loop runs if capturing has been initialized
while True:

    # reads frames from a camera
    _, frame = cap.read()

    # Display an original image
    cv2.imshow('Original image', frame)

    # finds edges in the input image image and
    # marks them in the output map edges
    edges = cv2.Canny(frame, 75, 100)  # try changing the threshold values

    # Display edges in a frame
    cv2.imshow('Edges', edges)

    # Wait for Esc key to stop
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

# Close the window
cap.release()

# De-allocate any associated memory usage
cv2.destroyAllWindows()

