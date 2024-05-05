import cv2
import numpy as np
from ultralytics import YOLO
import config

CARD_AREA = 3000
BKG_THRESH = 110
COALESCE_DISTANCE_SQUARED = 900
UPSIDE_DOWN_THRESH = 180


model = YOLO(config.CARDNET_PATH)  # load a pretrained model (recommended for training)


def run_detection(frame, model):
    top_corner = frame[10:84, 10:40]
    cv2.imshow("hi", top_corner)
    cv2.waitKey(1000)
    #result = list(model([frame], stream=True, conf=0.65, verbose=False))[0]
    #if len(result) == 0:
    #    return "?"
    #return result[0].names[int(result[0].boxes[0].cls.item())] or None



def threshold(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = gray#cv2.GaussianBlur(gray, (1, 1), 0)
    img_w, img_h = np.shape(frame)[:2]
    bkg_level = np.median(blur[0:img_w, 0:img_h])
    thresh_level = bkg_level + BKG_THRESH

    _, thresh = cv2.threshold(blur, thresh_level, 255, cv2.THRESH_BINARY)
    return thresh


def main():
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
    print(cap.get(cv2.CAP_PROP_FRAME_HEIGHT), cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    photos = []

    for i in range(10):
        ret, frame = cap.read()
        thresh = threshold(frame)
        photos.append(thresh)

    mask = photos[0]
    for i in photos[1:]:
        mask = cv2.bitwise_or(mask, i)
    mask = cv2.bitwise_not(mask)
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Fail")
            continue
        disp = cv2.resize(frame, (800, 450))
        cv2.imshow("ORIG", disp)
        gray_thresh = cv2.bitwise_and(threshold(frame), mask)
        thresh = cv2.cvtColor(gray_thresh, cv2.COLOR_GRAY2BGR)
        cnts, hier = cv2.findContours(gray_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)      

        filtered_cnts = []
        centroids = []
        for i in range(len(cnts)):
            M = cv2.moments(cnts[i])
            if M["m00"] == 0:
                continue
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            centroid = (cX, cY)     

            if not any(abs(cX - x) ** 2 + abs(cY - y) ** 2 < COALESCE_DISTANCE_SQUARED for x, y in centroids):
                filtered_cnts.append(cnts[i])
                centroids.append(centroid)
        cnts = filtered_cnts

        num_contours = 0
        for i in range(len(cnts)):
            if cv2.contourArea(cnts[i]) < CARD_AREA:
                cnts = cnts[:i]
                break
            if hier[0][i][3] != -1:  # this is a child
                continue
            num_contours += 1
            M = cv2.moments(cnts[i])
            try:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
            except ZeroDivisionError:
                cX, cY = 0, 0
            rect = cv2.minAreaRect(cnts[i])
            box = cv2.boxPoints(rect)
            box = np.int64(box)

            box = box[box[:,1].argsort()]
            top_points = box[:2][box[:2,0].argsort()]
            bottom_points = box[2:][box[2:,0].argsort()[::-1]]
            box = np.vstack([top_points, bottom_points])

            dist_top = np.sqrt((top_points[0,0] - top_points[1,0])**2 + (top_points[0,1] - top_points[1,1])**2)
            dist_left = np.sqrt((box[0,0] - box[3,0])**2 + (box[0,1] - box[3,1])**2)
            
            rotate = dist_top > dist_left
            
            width = 200
            height = 300

            dst_pts = np.array([
                [0, 0],
                [width - 1, 0],
                [width - 1, height - 1],
                [0, height - 1]], dtype='float32')

            M = cv2.getPerspectiveTransform(box.astype('float32'), dst_pts)
            warped = cv2.warpPerspective(thresh, M, (width, height))

            if rotate:
                warped = cv2.resize(warped, (300, 300))
                (h, w) = warped.shape[:2]
                center = (w // 2, h // 2)

                angle = 90
                M = cv2.getRotationMatrix2D(center, 90, 1.0)
                rotated = cv2.warpAffine(warped, M, (h, w))
                rotated = cv2.resize(rotated, (width, height))
            else:
                rotated = warped

            warped = rotated
            mean_value = np.mean(warped)
            
            # Check if the majority of the pixels are white
            #if mean_value > 100:
                # If yes, print "Right side up" in green text
            #    cv2.putText(warped, "Right side up", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            #else:
                # If no, print "Upside down" in red text
            #    cv2.putText(warped, "Upside down", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            # Display the warped image
            cv2.circle(thresh, (cX, cY), 7, (255, 0, 0), -1)
            cv2.polylines(thresh, [box], True, (0, 255, 0) if mean_value > UPSIDE_DOWN_THRESH else (0, 0, 255), 2)
            if mean_value > UPSIDE_DOWN_THRESH:
                class_name = run_detection(warped, model)
                cv2.putText(thresh, class_name, (cX, cY - 10), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 1)
            
            #cv2.imshow(f"Warped {num_contours}", warped)

        cv2.putText(thresh, f"Number of objects: {num_contours}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0))
        thresh = cv2.resize(thresh, (800, 450))
        cv2.imshow("Frame", thresh)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()