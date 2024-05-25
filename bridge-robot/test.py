#import cv2
#import mediapipe as mp
#import numpy as np
#from collections import deque, Counter
#
#mp_drawing = mp.solutions.drawing_utils
#mp_hands = mp.solutions.hands
#
## Start capturing video from the webcam
#cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
#
#prev_centroids = {}
#directions = {}
#hand_detected = {}
#
#with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
#    while cap.isOpened():
#        ret, frame = cap.read()
#        if not ret:
#            continue
#
#        # Flip the image horizontally for a later selfie-view display
#        image = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
#
#        # Process the image and find hand landmarks
#        results = hands.process(image)
#
#        # Convert the image color back so it can be displayed
#        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
#
#        # Draw hand landmarks of each hand
#        if results.multi_hand_landmarks:
#            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
#                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
#
#                # Calculate the centroid of the hand
#                coords = np.zeros((21, 2))
#                for j, landmark in enumerate(hand_landmarks.landmark):
#                    coords[j] = [landmark.x * image.shape[1], landmark.y * image.shape[0]]
#                centroid_x, centroid_y = np.mean(coords, axis=0)
#
#                # Use the handedness index as the hand ID
#                hand_id = handedness.classification[0].index
#
#                # Determine the direction of the hand
#                if hand_id not in directions:
#                    if hand_id in prev_centroids:
#                        dx, dy = centroid_x - prev_centroids[hand_id][0], centroid_y - prev_centroids[hand_id][1]
#                        if abs(dx) > abs(dy):
#                            direction = "left" if dx > 0 else "right"
#                        else:
#                            direction = "top" if dy > 0 else "bottom"
#                        directions[hand_id] = direction
#                        print(f"Hand {hand_id} initially moved from {direction}")
#
#                if hand_id not in prev_centroids:
#                    prev_centroids[hand_id] = (centroid_x, centroid_y)
#
#                hand_detected[hand_id] = True
#
#        # Remove the stored centroid and direction for hands that are no longer detected
#        hands_to_remove = set(prev_centroids.keys()) - set(hand_detected.keys())
#        for hand_id in hands_to_remove:
#            if hand_id in prev_centroids:
#                del prev_centroids[hand_id]
#            if hand_id in directions:
#                del directions[hand_id]
#
#        hand_detected.clear()
#
#        # Display the image
#        cv2.imshow('MediaPipe Hands', image)
#
#        # Exit loop if 'q' is pressed
#        if cv2.waitKey(5) & 0xFF == ord('q'):
#            break
#
#cap.release()
#cv2.destroyAllWindows()


#a = int(input())
#
#
#for i in range(2, 1000000000000000000000):
#    curr = a ** (1/i)
#    
#    if curr < 2:
#        print("False")
#        break
#
#    if curr == int(curr):
#        print("True")
#        break
    
#else:
#    print("False")


import ray
from src.vision.maincam import MainCam
import config
import cv2
import time


ray.init()

@ray.remote
class CameraProcess:
    def __init__(self):
        self.curr_mode = None
        self.cap = MainCam(config.MAINCAM_INDEX, config.MAINCAM_WIDTH, config.MAINCAM_HEIGHT)

    def set_mode(self, mode):
        self.curr_mode = mode

    def process(self):
        frame = self.cap.raw_read()
        match self.curr_mode:
            case "auction":
                image = self.cap.preprocess_image(frame)
                det = self.cap.detect(image, bidding=True, preprocess=False)
                return (image, det)
            case "cardplay":
                d = self.cap.detect_cards(frame)
                return d
            case _:
                d = self.cap.preprocess_image(frame)
                return d, None
            

if __name__ == "__main__":

    # Create an instance of the CameraProcess actor.
    camera_process = CameraProcess.remote()

    # Set the mode of the camera process.
    camera_process.set_mode.remote("auction")  # Or whatever mode you want to set.

    # Initialize the frame rate and the time of the previous frame.
    frame_rate = 0.0
    prev_time = time.time()

    # Continuously get the last frame from the camera process and output it.
    while True:
        # Get the last frame from the camera process.
        frame_future = camera_process.process.remote()

        # Wait for the frame to be ready and get the result.
        frame, det = ray.get(frame_future)
        frame_copy = frame.copy()
        # Calculate the frame rate using an exponential moving average.
        curr_time = time.time()
        alpha = 0.1  # Adjust as needed.
        frame_rate = (1 - alpha) * frame_rate + alpha * (1.0 / (curr_time - prev_time))
        prev_time = curr_time

        # Draw the frame rate on the frame.
        cv2.putText(frame_copy, f"FPS: {frame_rate:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display the frame.
        cv2.imshow("Frame", frame_copy)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()
    ray.shutdown()
    camera_process.cap.release()
# a, b = [int(i) for i in input().strip().split()]
# n = int(input())
# curr = 0
# for x in range(200000000000000):
#     curr = a * x
#     if curr % b == n % b:
#         break
#     elif (-curr) % b == n % b:
#         curr = -curr
#         x = -x
#         break
# 
# print(x)
# print((n-curr) // b)