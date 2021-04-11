import cv2
import numpy as np
import imutils
import argparse
import time
from collections import deque
from directkeys import space_pressed
from directkeys import PressKey, ReleaseKey

accelerator_key_pressed = space_pressed

ap = argparse.ArgumentParser()
ap.add_argument("-b", "--buffer", type=int, default=64)
args = vars(ap.parse_args())
pts = deque(maxlen=args["buffer"])

time.sleep(0)
cap = cv2.VideoCapture(0)

hsv_value = np.load('hsv_value.npy')

lower = hsv_value[0]
upper = hsv_value[1]

current_key_pressed = set()
accelerator_pressed = False
height = 500
width = 500
while True:
    keyPressed = False
    accelerator_pressed = False
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1);
    # cv2.imshow("Cam", frame)
    frame = imutils.resize(frame, height = height)
    frame = imutils.resize(frame, width = width)
    img = cv2.GaussianBlur(frame, (15, 15), 2)
    img_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # cv2.imshow("HSV Image", img_hsv)
    img_mask = cv2.inRange(img_hsv, lower, upper)
    # cv2.imshow("Mask Image", img_mask)
    kernel = np.ones((5, 5), np.int8)
    img_mask = cv2.erode(img_mask, kernel, 1)
    img_mask = cv2.dilate(img_mask, kernel, 2)
    # cv2.imshow("Erosion & Dilation Image", img_mask)
    contours = cv2.findContours(img_mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    centroid = None

    key_count = 0
    key_pressed = 0
    if len(contours) > 0:
        max_cnt = max(contours, key=cv2.contourArea)
        # x, y, w, h = cv2.boundingRect(max_cnt)
        # frame = cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
        # cv2.imshow("Object Image", img)
        ((x, y), radius) = cv2.minEnclosingCircle(max_cnt)
        moment = cv2.moments(max_cnt)
        centroid = (int(moment['m10']/moment['m00']), int(moment['m01']/moment['m00']))
        if radius > 10:
            frame = cv2.circle(frame, (int(x), int(y)), int(radius)+4, (0, 255, 0), 2)
            frame = cv2.circle(frame, centroid, 3, (0, 0, 255), -1)
            if centroid[1] > 250:
                cv2.putText(frame, 'Jump Applied', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 4)
                PressKey(accelerator_key_pressed)
                accelerator_pressed = True
                current_key_pressed.add(accelerator_key_pressed)
                # Break key- 75 #Acc key-77
                key_pressed = accelerator_key_pressed
                keyPressed = True
                key_count = key_count + 1
    frame_copy = frame.copy()
    frame_copy = cv2.rectangle(frame_copy, (0, width//2), (height-1, width), (0, 255, 0), 2)
    cv2.putText(frame_copy, 'Jump', (10, 280), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2, cv2.LINE_8)
    cv2.imshow("Frame", frame_copy)
    if not keyPressed and len(current_key_pressed) != 0:
        for key in current_key_pressed:
            ReleaseKey(key)
        current_key_pressed = set()
    elif key_count == 1 and len(current_key_pressed) == 2:
        for key in current_key_pressed:
            if key_pressed != key:
                ReleaseKey(key)
        current_key_pressed = set()
        for key in current_key_pressed:
            ReleaseKey(key)
        current_key_pressed = set()

    key = cv2.waitKey(100) & 0xFF
    if key == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
