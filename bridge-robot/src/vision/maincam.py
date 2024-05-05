from src.vision.camera import Camera
import cv2
import config
import numpy as np

CARD_AREA = 3000
BKG_THRESH = 110
COALESCE_DISTANCE_SQUARED = 900
UPSIDE_DOWN_THRESH = 180

class MainCam(Camera):
    def __init__(self, camera_index, width, height):
        super().__init__(camera_index, width, height, True)
        self.mask = self.create_mask()

    def create_mask(self):
        photos = []

        for i in range(10):
            frame = self.raw_read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            img_w, img_h = np.shape(frame)[:2]
            bkg_level = np.median(gray[0:img_w, 0:img_h])
            thresh_level = bkg_level + BKG_THRESH
            _, thresh = cv2.threshold(gray, thresh_level, 255, cv2.THRESH_BINARY)
            photos.append(thresh)

        mask = photos[0]
        for i in photos[1:]:
            mask = cv2.bitwise_or(mask, i)
        mask = cv2.bitwise_not(mask)
        return mask

    def old_algo(self):
        frame = self.raw_read()
        disp = cv2.resize(frame, (800, 450))
        cv2.imshow("ORIG", disp)
        cv2.waitKey(1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        img_w, img_h = np.shape(frame)[:2]
        bkg_level = np.median(gray[0:img_w, 0:img_h])
        thresh_level = bkg_level + BKG_THRESH
        _, thresh = cv2.threshold(gray, thresh_level, 255, cv2.THRESH_BINARY)
        gray_thresh = cv2.bitwise_and(thresh, self.mask)
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
         
            cv2.circle(thresh, (cX, cY), 7, (255, 0, 0), -1)
            cv2.polylines(thresh, [box], True, (0, 255, 0) if mean_value > UPSIDE_DOWN_THRESH else (0, 0, 255), 2)
            if mean_value > UPSIDE_DOWN_THRESH:
                data = self.detect(warped) 
                if len(data) == 0:
                    class_name = "Unknown"
                else:
                    class_name = data[0][0]
                cv2.putText(thresh, class_name, (cX, cY - 10), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 1)
                
if __name__ == "__main__":
    with MainCam(config.MAINCAM_INDEX, 1920, 1080) as cam:
        while True:
            frame = cam.draw_boxes(bidding=True)
            cv2.imshow("frame", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    cv2.destroyAllWindows()