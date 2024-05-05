import cv2
import mediapipe as mp
import numpy as np
from collections import deque, Counter

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Start capturing video from the webcam
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

prev_centroids = {}
directions = {}
hand_detected = {}

with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            continue

        # Flip the image horizontally for a later selfie-view display
        image = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)

        # Process the image and find hand landmarks
        results = hands.process(image)

        # Convert the image color back so it can be displayed
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Draw hand landmarks of each hand
        if results.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Calculate the centroid of the hand
                coords = np.zeros((21, 2))
                for j, landmark in enumerate(hand_landmarks.landmark):
                    coords[j] = [landmark.x * image.shape[1], landmark.y * image.shape[0]]
                centroid_x, centroid_y = np.mean(coords, axis=0)

                # Use the handedness index as the hand ID
                hand_id = handedness.classification[0].index

                # Determine the direction of the hand
                if hand_id not in directions:
                    if hand_id in prev_centroids:
                        dx, dy = centroid_x - prev_centroids[hand_id][0], centroid_y - prev_centroids[hand_id][1]
                        if abs(dx) > abs(dy):
                            direction = "left" if dx > 0 else "right"
                        else:
                            direction = "top" if dy > 0 else "bottom"
                        directions[hand_id] = direction
                        print(f"Hand {hand_id} initially moved from {direction}")

                if hand_id not in prev_centroids:
                    prev_centroids[hand_id] = (centroid_x, centroid_y)

                hand_detected[hand_id] = True

        # Remove the stored centroid and direction for hands that are no longer detected
        hands_to_remove = set(prev_centroids.keys()) - set(hand_detected.keys())
        for hand_id in hands_to_remove:
            if hand_id in prev_centroids:
                del prev_centroids[hand_id]
            if hand_id in directions:
                del directions[hand_id]

        hand_detected.clear()

        # Display the image
        cv2.imshow('MediaPipe Hands', image)

        # Exit loop if 'q' is pressed
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()