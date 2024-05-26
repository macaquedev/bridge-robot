from src.vision.camera import Camera
import cv2
import config
import numpy as np
import time

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
            thresh_level = bkg_level + config.BKG_THRESH
            _, thresh = cv2.threshold(gray, thresh_level, 255, cv2.THRESH_BINARY)
            photos.append(thresh)

        mask = photos[0]
        for i in photos[1:]:
            mask = cv2.bitwise_or(mask, i)
        mask = cv2.bitwise_not(mask)
        return mask

    def detect_cards(self, frame=None, draw=False):
        if frame is None:
            frame = self.raw_read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        img_w, img_h = np.shape(frame)[:2]
        bkg_level = np.median(gray[0:img_w, 0:img_h])
        thresh_level = bkg_level + config.BKG_THRESH
        _, thresh = cv2.threshold(gray, thresh_level, 255, cv2.THRESH_BINARY)
        gray_thresh = cv2.bitwise_and(thresh, thresh)#, self.mask)
        thresh = cv2.cvtColor(gray_thresh, cv2.COLOR_GRAY2BGR)
        cnts, _ = cv2.findContours(gray_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        filtered_cnts = []
        centroids = []
        for i in range(len(cnts)):
            M = cv2.moments(cnts[i])
            if M["m00"] == 0:
                continue
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            centroid = (cX, cY)     
            if not any(abs(cX - x) ** 2 + abs(cY - y) ** 2 < config.COALESCE_DISTANCE_SQUARED for x, y in centroids):
                filtered_cnts.append(cnts[i])
                centroids.append(centroid)
        cnts = filtered_cnts
        num_contours = 0
        to_detect = []
        
        centroids = [cv2.moments(cnt) for cnt in cnts if cv2.moments(cnt)["m00"] != 0]
        y_coords = [int(M["m01"] / M["m00"]) for M in centroids]
        pairs = list(zip(cnts, y_coords))
        sorted_pairs = sorted(pairs, key=lambda x: x[1])
        cnts = [pair[0] for pair in sorted_pairs]
        for i in range(len(cnts)):
            if cv2.contourArea(cnts[i]) < config.CARD_AREA:
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
            warped = cv2.warpPerspective(frame, M, (width, height))
            if rotate:
                warped = cv2.resize(warped, (300, 300))
                (h, w) = warped.shape[:2]
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, 90, 1.0)
                rotated = cv2.warpAffine(warped, M, (h, w))
                rotated = cv2.resize(rotated, (width, height))
            else:
                rotated = warped
            warped = cv2.resize(rotated, (config.DETECT_CARD_WIDTH, config.DETECT_CARD_HEIGHT))#[:200, :150]
            to_detect.append((warped, box))
            cv2.circle(thresh, (cX, cY), 15, (255, 0, 0), -1)
            cv2.polylines(thresh, [box], True, (0, 255, 0), 10)

        detector_image = np.zeros((config.DETECT_FRAME_SIZE, config.DETECT_FRAME_SIZE, 3), np.uint8)
        
        for i, (warped, box) in enumerate(to_detect):
            r, c = divmod(i, 6)
            if r >= 4:
                continue
            #sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            detector_image[r*config.DETECT_CARD_HEIGHT:(r+1)*config.DETECT_CARD_HEIGHT, c*config.DETECT_CARD_WIDTH:config.DETECT_CARD_WIDTH*(c+1)] = warped#cv2.filter2D(warped, -1, sharpen_kernel)

        cards = [["No Detection" for _ in range(6)] for _ in range(4)]
        for label, corner1, corner2 in self.detect(frame=detector_image, bidding=False):
            if draw:
                cv2.rectangle(detector_image, corner1, corner2, (255, 0, 0), 2)
                cv2.putText(detector_image, label, (corner1[0], corner2[1]+30),
                             cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 1)

            x, y = (corner1[0] + corner2[0]) // 2, (corner1[1] + corner2[1]) // 2
            r, c = int(y // (300 / 1200 * config.DETECT_FRAME_SIZE)), int(x // (200 / 1200 * config.DETECT_FRAME_SIZE))
            cards[r][c] = label
        output = []
        for i, (warped, box) in enumerate(to_detect):
            r, c = divmod(i, 6)
            cv2.putText(thresh, cards[r][c], box[0], cv2.FONT_HERSHEY_SIMPLEX, 10, (43, 75, 255), 16)
            output.append((cards[r][c], box))
        return thresh, detector_image, output

if __name__ == "__main__":
    with MainCam(config.MAINCAM_INDEX, 1920, 1080) as cam:
        #cam.create_mask()
        #while True:
        #    thresh, detector_image, output = cam.detect_cards()
        #    cv2.imshow("thresh", thresh)
        #    cv2.imshow("detector_image", detector_image)
        #    if cv2.waitKey(1) & 0xFF == ord('q'):
        #        break
        while True:
            frame = cam.draw_boxes(bidding=True)
            cv2.imshow("frame", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cv2.destroyAllWindows()