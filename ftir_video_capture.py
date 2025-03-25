
from collections import deque
from scipy import ndimage
import math
import time
import numpy as np
import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.image as img
from sklearn import datasets
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.datasets import fetch_openml

def callback(x):
    pass

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def getBestShift(img):
    cy,cx = ndimage.center_of_mass(img)
    rows,cols = img.shape
    shiftx = np.round(cols/2.0-cx).astype(int)
    shifty = np.round(rows/2.0-cy).astype(int)
    return shiftx,shifty

def shift(img,sx,sy):
    rows,cols = img.shape
    M = np.float32([[1,0,sx],[0,1,sy]])
    shifted = cv2.warpAffine(img,M,(cols,rows))
    return shifted

# train classifier
size = 1000
mnist = fetch_openml('mnist_784',parser='auto')

x, y = mnist['data'], mnist['target']
x_train, y_train = x[:size], y[:size]

svm_clf = SVC(decision_function_shape = 'ovo')
print('training start')
svm_clf.fit(x_train, y_train)
SVC(decision_function_shape='ovo')
print("training end")

# Create threshold trackbar
cv2.namedWindow('Threshold Sliders')
cv2.createTrackbar('R', 'Threshold Sliders', 156, 255, callback)
cv2.createTrackbar('B', 'Threshold Sliders', 127, 255, callback)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    cap.open()

pts = deque(maxlen=256)
start_time = time.time()
tmp = ""
while(True):
    end_time = time.time()
    idle_time = end_time - start_time
    # if touch pad is idle for 1 second, predict the digit
    if (pts and idle_time > 1):
        # create empty white image
        blank_image = np.zeros((display.shape[0],display.shape[1],3), np.uint8)

        # draw line for digit
        for i in range(1, len(pts)):
            if pts[i - 1] is None or pts[i] is None:
                continue
            cv2.line(blank_image, pts[i - 1], pts[i], (255, 255, 255), 30)
        # save image
        cv2.imwrite('a.png', blank_image)

        gray = cv2.cvtColor(blank_image, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (28, 28))

        if gray.size != 0:
            while gray[0].size != 0 and np.sum(gray[0]) == 0:
                gray = gray[1:]
            while gray[:,0].size != 0 and np.sum(gray[:,0]) == 0:
                gray = np.delete(gray,0,1)
            while gray[-1].size != 0 and np.sum(gray[-1]) == 0:
                gray = gray[:-1]
            while gray[:,-1].size != 0 and np.sum(gray[:,-1]) == 0:
                gray = np.delete(gray,-1,1)
            rows,cols = gray.shape

            if rows > cols:
                factor = 20.0/rows
                rows = 20
                cols = int(round(cols*factor))
                gray = cv2.resize(gray, (cols,rows))
            else:
                factor = 20.0/cols
                cols = 20
                rows = int(round(rows*factor))
                gray = cv2.resize(gray, (cols, rows))
                        
            shiftx,shifty = getBestShift(gray)
            shifted = shift(gray,shiftx,shifty)
            gray = shifted

            colsPadding = (int(math.ceil((28-cols)/2.0)),int(math.floor((28-cols)/2.0)))
            rowsPadding = (int(math.ceil((28-rows)/2.0)),int(math.floor((28-rows)/2.0)))
            gray = np.lib.pad(gray,(rowsPadding,colsPadding),'constant')

            cv2.imwrite('b.png', gray)

            flatten = gray.flatten()
            # predict the digit
            predicted = svm_clf.predict([flatten])
            tmp = predicted
            print(predicted)
        
        pts.clear()

    # Get one frame from the camera
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)

    # Crop the frame to 480x480
    w = 640
    h = 480
    x = h//2 - 240
    y = w//2 - 240
    frame = frame[x:x+h, y:y+h]

    # Split RGB channels
    b, g, r = cv2.split(frame)
    r_threshold = cv2.getTrackbarPos('R', 'Threshold Sliders')
    b_threshold = cv2.getTrackbarPos('B', 'Threshold Sliders')

    # Perform thresholding to each channel
    _, r = cv2.threshold(r, r_threshold, 255, cv2.THRESH_BINARY)
    _, b_inv = cv2.threshold(b, b_threshold, 255, cv2.THRESH_BINARY_INV)

    # Get the final result using bitwise operation
    result = cv2.bitwise_and(r, b_inv, mask=None)

    # Find and draw contours
    contours, hierarchy = cv2.findContours(result, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    display = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(display, contours, -1, (0, 0, 255))

    # Iterate through each contour, check the area and find the center
    if (len(contours) > 0):
        c = max(contours, key=cv2.contourArea)
        (x,y), radius = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)

        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            cX, cY = 0, 0

        center = (cX, cY)

        if radius > 10:
            start_time = time.time()
            org = (int(x), int(y))
            text = f'({org})'
            scale = 1
            color = (0, 255, 0)
            cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color)
            cv2.putText(display, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color)
            cv2.circle(frame, org, int(radius), color, 2)
            cv2.circle(display, org, int(radius), color, 2)
            pts.appendleft(center)

    # Draw line for frame and display
    for i in range(1, len(pts)):
        if pts[i - 1] is None or pts[i] is None:
            continue
        cv2.line(frame, pts[i - 1], pts[i], (0, 255, 0),5)
        cv2.line(display, pts[i - 1], pts[i], (0, 255, 0), 5)
    text_to_display = f'prediction: {tmp}'
    cv2.putText(display, text_to_display, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    # Show the frame and contours
    cv2.imshow('frame', frame)
    cv2.imshow('Contours', display)

    # Press q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    

# Release the camera
cap.release()

# Close all OpenCV windows
cv2.destroyAllWindows()

