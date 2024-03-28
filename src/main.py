import cv2
import numpy as np


CARD_AREA = 6000
BKG_THRESH = 130


def threshold(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (1, 1), 0)
    img_w, img_h = np.shape(frame)[:2]
    bkg_level = np.median(blur[0:img_w, 0:img_h])
    thresh_level = bkg_level + BKG_THRESH

    _, thresh = cv2.threshold(blur, thresh_level, 255, cv2.THRESH_BINARY)
    return thresh


if __name__ == "__main__":
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 576)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1024)

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
        gray_thresh = cv2.bitwise_and(threshold(frame), mask)
        thresh = cv2.cvtColor(gray_thresh, cv2.COLOR_GRAY2BGR)
        cnts, hier = cv2.findContours(gray_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        try:
            print(hier.shape)
        except: pass
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
        num_contours = 0
        for i in range(len(cnts)):
            if cv2.contourArea(cnts[i]) < CARD_AREA:
                cnts = cnts[:i]
                break
            if hier[0][i][3] != -1:  # this is a child
                continue
            num_contours += 1
            M = cv2.moments(cnts[i])
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            if hier[0][i][2] != -1 and cv2.contourArea(cnts[hier[0][i][2]]) > 1500:  # this contour has a child
                cv2.circle(thresh, (cX, cY), 7, (0, 0, 255), -1)
                thresh = cv2.drawContours(thresh, [cnts[hier[0][i][2]]], -1, color=(0, 255, 0), thickness=1)
                thresh = cv2.drawContours(thresh, [cnts[i]], -1, color=(0, 0, 255), thickness=1)

                cv2.putText(thresh, "upside down", (cX - 20, cY - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            else:
                cv2.circle(thresh, (cX, cY), 7, (255, 0, 0), -1)
                rect = cv2.minAreaRect(cnts[i])
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                cv2.drawContours(thresh, [box], 0, (0, 255, 0), 2)
                cv2.putText(thresh, "right way up", (cX - 20, cY - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)


        cv2.putText(thresh, f"Number of objects: {num_contours}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0))
        cv2.imshow("Frame", thresh)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()