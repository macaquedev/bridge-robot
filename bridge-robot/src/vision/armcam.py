from collections import defaultdict
import json
import time
from src.vision.camera import Camera, robust_mean
import config
import cv2
from ultralytics import YOLO
import numpy as np


class ArmCam(Camera):
    def __init__(self, camera_index, width, height):
        super().__init__(camera_index, width, height, True)
    

    def detect(self, frame=None):
        data = super().detect(frame, bidding=False)
        return [card for card in data if card[1][1]+card[2][1] < self.height]
    

    def calibrate(self, arduino):
        arduino.send_data_and_wait_for_acknowledgement("ARMUP")
        detections = []
        for curr in range(0, 400, 30):
            arduino.send_data_and_wait_for_acknowledgement(f"STEPPER {curr}")
            time.sleep(0.5)
            detections.extend([(curr, self.detect()) for _ in range(5)])
            cv2.imshow("HI", self.draw_boxes())
            cv2.waitKey(1)
        arduino.send_data_and_wait_for_acknowledgement("STEPPER 200")
        arduino.send_data_and_wait_for_acknowledgement("ARMDOWN")
        arduino.send_data_and_wait_for_acknowledgement("STEPPER 0")
        print("Finished!")
        cards = defaultdict(list)
        for pos, det in detections:
            for card, top_left, bottom_right in det:
                centroid = (top_left[0] + bottom_right[0])//2
                cards[card].append((pos, centroid))
        for key in list(cards.keys()):
            if key in ["AS"]:
                continue
            if len(cards[key]) < config.ARMCAM_MISDETECTION_THRESHOLD:
                cards.pop(key)
        result = [(card, len(data)) for card, data in cards.items()]
        print(sorted(result))
        distances = []
        for card, data in cards.items():
            for i in range(len(data)):
                for j in range(i+1, len(data)):
                    (stepper_pos_1, px_1) = data[i]
                    (stepper_pos_2, px_2) = data[j]
                    if px_1 == px_2:
                        continue
                    distances.append(abs(stepper_pos_1 - stepper_pos_2) / abs(px_1 - px_2))
        mm_per_px = robust_mean(np.array(distances))
        result.sort(key=lambda card: cards[card[0]])
        print([i[0] for i in result])
        for card, _ in result:
            local_distances = []
            for i in range(len(cards[card])):
                for j in range(i+1, len(cards[card])):
                    (stepper_pos_1, px_1) = cards[card][i]
                    (stepper_pos_2, px_2) = cards[card][j]
                    if px_1 == px_2:
                        continue
                    local_distances.append(abs(stepper_pos_1 - stepper_pos_2) / abs(px_1 - px_2))
            local_mm_per_px = robust_mean(np.array(local_distances))
            if abs(local_mm_per_px - mm_per_px) > 0.1:
                local_mm_per_px = mm_per_px
            m = robust_mean(np.array([data[0] + (data[1]-config.ARMCAM_OFFSET) * local_mm_per_px for data in cards[card]]))
            arduino.send_data_and_wait_for_acknowledgement(f"STEPPER {m}")
            input()
            arduino.send_data_and_wait_for_acknowledgement("GRAB")
            arduino.send_data_and_wait_for_acknowledgement("RELEASE")
        #for card, data in cards.items():
        #    print(data)
        

if __name__ == "__main__":
    with ArmCam(config.ARMCAM_INDEX, config.ARMCAM_WIDTH, config.ARMCAM_HEIGHT) as cam:
        while True:
            frame = cam.draw_boxes()
            cv2.imshow("frame", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    cv2.destroyAllWindows()
