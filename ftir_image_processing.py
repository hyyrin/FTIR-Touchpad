
import cv2
import numpy as np

def callback(x):
    pass

cv2.namedWindow('Threshold Sliders')
cv2.createTrackbar('R', 'Threshold Sliders', 225, 255, callback)
cv2.createTrackbar('B', 'Threshold Sliders', 200, 255, callback)

# Read the image
frame = cv2.imread("test.png")

# Split RGB channels
b, g, r = cv2.split(frame)

r_threshold = cv2.getTrackbarPos('R', 'Threshold Sliders')
b_threshold = cv2.getTrackbarPos('B', 'Threshold Sliders')

# Perform thresholding to each channel
_, r = cv2.threshold(r, r_threshold, 255, cv2.THRESH_BINARY)
_, b_inv = cv2.threshold(b, b_threshold, 255, cv2.THRESH_BINARY_INV)

zeros = np.zeros(frame.shape[:2], dtype="uint8")
cv2.imshow("Red", cv2.merge([zeros, zeros, r]))

# Get the final result using bitwise operation
result = cv2.bitwise_and(r, b_inv, mask=None)

# Find and draw contours
contours, hierarchy = cv2.findContours(result, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Draw the contours
display = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
cv2.drawContours(display, contours, -1, (0, 0, 255))
cv2.imshow('Contours', display)

# Iterate through each contour, check the area and find the center
for cnt in contours:
    area = cv2.contourArea(cnt)
    (x,y), radius = cv2.minEnclosingCircle(cnt)

# Show the frame
cv2.imshow('frame', frame)

# Press any key to quit
cv2.waitKey(0)

# Close all OpenCV windows
cv2.destroyAllWindows()

