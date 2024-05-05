from src.vision.maincam import MainCam
import config 
import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO(config.BIDDING_BOX_NET_PATH)

with MainCam(config.MAINCAM_INDEX, config.MAINCAM_WIDTH, config.MAINCAM_HEIGHT) as camera:
    while True:
        frame = cv2.resize(camera.raw_read(), None, fx=0.3, fy=0.3)
        
        #hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        ## Threshold the HSV image to get only red colors
        #mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        #mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        #mask = cv2.bitwise_or(mask1, mask2)

        ## Find contours in the mask
        #contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        #cv2.drawContours(frame, contours, -1, (0, 255, 0), 3)

        ## Find the contour with the largest area
        #largest_contour = max(contours, key=cv2.contourArea)
        #cv2.drawContours(frame, [largest_contour], -1, (255, 0, 0), 5)
        ## Display the resulting frame
        #cv2.imshow('frame', frame)
        #if cv2.waitKey(1) & 0xFF == ord('q'):
        #    break
        #continue
        ## Get the bounding rectangle for the largest contour
        #x, y, w, h = cv2.boundingRect(largest_contour)

        ## Warp transform the largest contour into a rectangle
        #src_pts = np.float32([largest_contour[0], largest_contour[1], largest_contour[2], largest_contour[3]])
        #dst_pts = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        #matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
        #result = cv2.warpPerspective(frame, matrix, (w, h))

        ## Display the resulting frame
        #cv2.imshow('frame', result)
        #if cv2.waitKey(1) & 0xFF == ord('q'):
        #    break