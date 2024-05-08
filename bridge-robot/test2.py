import cv2
import numpy as np
from src.vision.maincam import MainCam
import config
from collections import defaultdict

add_threshold = 2
remove_threshold = 10

with MainCam(config.MAINCAM_INDEX, config.MAINCAM_WIDTH, config.MAINCAM_HEIGHT) as cap:
    frame = cv2.resize(cap.raw_read(), (1280, 720))
    in_frame = defaultdict(list)
    to_be_added = defaultdict(list)
    to_be_removed = defaultdict(list)
    while True:
        frame = cv2.resize(cap.raw_read(), (1280, 720))
        frame_center = (frame.shape[1] // 2, frame.shape[0] // 2)  # Center of the frame
        det = cap.detect(frame, bidding=True)
        
        for card, top_left, bottom_right in det:
            for i in range(len(in_frame[card])):
                box = in_frame[card][i][0]
                if (top_left[0] - box[0]) ** 2 + (top_left[1] - box[1]) ** 2 < config.COALESCE_DISTANCE_SQUARED:
                    in_frame[card][i] = [top_left, bottom_right, in_frame[card][i][2]]
                    break
            else:
                for i in range(len(to_be_added[card])):
                    box = to_be_added[card][i][1]
                    if (top_left[0] - box[0]) ** 2 + (top_left[1] - box[1]) ** 2 < config.COALESCE_DISTANCE_SQUARED:
                        break
                else:
                    to_be_added[card] = [[0, top_left, bottom_right]]
            
            if card not in in_frame:
                if card in to_be_removed:
                    for i in range(len(to_be_removed[card])):
                        box = to_be_removed[card][i][1]
                        if (top_left[0] - box[0]) ** 2 + (top_left[1] - box[1]) ** 2 < config.COALESCE_DISTANCE_SQUARED:
                            to_be_removed[card].pop(i)
                            break


        for card in in_frame:
            to_remove = set()
            for i in range(len(in_frame[card])):
                if i in to_remove:
                    continue
                to_remove.add(i)
                for det_card, top_left, top_right in det:
                    if det_card == card and (top_left[0] - in_frame[card][i][0][0]) ** 2 + (top_left[1] - in_frame[card][i][0][1]) ** 2 < config.COALESCE_DISTANCE_SQUARED:
                        to_remove.remove(i)
                        break

            for i in to_remove:
                if card not in to_be_removed:
                    to_be_removed[card] = []
                for j in range(len(to_be_removed[card])):
                    if (in_frame[card][i][0][0] - to_be_removed[card][j][1][0]) ** 2 + (in_frame[card][i][0][1] - to_be_removed[card][j][1][1]) < config.COALESCE_DISTANCE_SQUARED:
                        break    
                
                else:
                    to_be_removed[card].append([0, in_frame[card][i][0], in_frame[card][i][1]])


        for card in to_be_added:
            for i in range(len(to_be_added[card])):
                to_be_added[card][i][0] += 1
                if to_be_added[card][i][0] == add_threshold:
                    if card not in in_frame:
                        in_frame[card] = []
                    in_frame[card].append([to_be_added[card][i][1], to_be_added[card][i][2], None])
                    box = in_frame[card][-1]
                    x_pos = box[0][0] + (box[1][0] - box[0][0]) // 2
                    y_pos = box[0][1] + (box[1][1] - box[0][1]) // 2
                    dx = min(x_pos, frame.shape[1] - x_pos)
                    dy = min(y_pos, frame.shape[0] - y_pos)
                    if dx < frame.shape[1]/3:
                        if x_pos < frame_center[0]:
                            player = "left"
                        else:
                            player = "right"
                    elif dy < frame.shape[0]/3:
                        if y_pos < frame_center[1]:
                            player = "top"
                        else:
                            player = "bottom"
                    else:
                        player = "center"
                    to_be_added[card].pop(i)
                    in_frame[card][-1][2] = player   
                    break
        

        for card in to_be_removed:
            to_remove = set()
            for i in range(len(to_be_removed[card])):
                to_be_removed[card][i][0] += 1
                if to_be_removed[card][i][0] >= remove_threshold:
                    to_remove.add(i)
                    break

            for i in to_remove:
                top_left = to_be_removed[card][i][1]
                for j in range(len(in_frame[card])):
                    if (top_left[0] - in_frame[card][j][0][0]) ** 2 + (top_left[1] - in_frame[card][j][0][1]) ** 2 < config.COALESCE_DISTANCE_SQUARED:
                        in_frame[card].pop(j)
                        break

                to_be_removed[card].pop(i)

        
        for card in in_frame:
            for top_left, bottom_right, player in in_frame[card]:
                cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)
                cv2.putText(frame, f"{player} played {card}", (top_left[0], top_left[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                #cv2.circle(frame, center, 5, (0, 255, 0), -1)

        cv2.imshow("Frame", frame)
        
        # Update previous frame
        print(to_be_added, in_frame, to_be_removed)
        img = np.zeros((500, 500, 3), dtype=np.uint8)
        y = 20
        for card_label in in_frame:
            for top_left, bottom_right, player in in_frame[card_label]:
                cv2.putText(img, f"{player} played {card_label}", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                y += 20
        cv2.imshow("In Frame", img)


        # Break loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    cv2.destroyAllWindows()

"""
import cv2
import numpy as np
from src.vision.maincam import MainCam
import config
from collections import defaultdict
import threading

add_threshold = 3
remove_threshold = 4

# Shared variables
frame = None
det = None
in_frame = {}
to_be_added = {}
to_be_removed = {}
barycenter_history = []
lock = threading.Lock()


def detection_thread(cap):
    global frame
    global det
    global in_frame
    global to_be_added
    global to_be_removed
    global barycenter_history
    while True:
        with lock:
            det = cap.detect(frame, bidding=True)
            detections = defaultdict(list)
            for card, top_left, bottom_right in det:
                detections[card].append((top_left, bottom_right))
            for card, boxes in detections.items():
                if card in in_frame:
                    in_frame[card] = (boxes, in_frame[card][1])
                elif card not in to_be_added:
                    to_be_added[card] = 0
            for card in in_frame:
                if card in detections:
                    continue
                if card in to_be_removed:
                    to_be_removed[card] += 1
                else:
                    to_be_removed[card] = 0
            cards_to_remove = []
            for card in to_be_removed:
                if card in detections:
                    cards_to_remove.append(card)
                    continue
                if to_be_removed[card] == remove_threshold:
                    cards_to_remove.append(card)
            for card in cards_to_remove:
                to_be_removed.pop(card)
                in_frame.pop(card)
            cards_to_remove = []
            for card in to_be_added:
                if card not in detections:
                    cards_to_remove.append(card)
                    continue
                if card in to_be_added:
                    to_be_added[card] += 1
                else:
                    to_be_added[card] = 0
                if len(barycenter_history) > 1 and to_be_added[card] == add_threshold:
                    cards_to_remove.append(card)
                    print(barycenter_history)
                    initial = barycenter_history[0]
                    final = barycenter_history[-1]
                    dx = final[0] - initial[0]
                    dy = final[1] - initial[1]
                    if abs(dx) > abs(dy):
                        if dx > 0:
                            direction = 'right'
                        else:
                            direction = 'left'
                    else:
                        if dy > 0:
                            direction = 'bottom'
                        else:
                            direction = 'top'
                    direction = f"{initial}->{final}" + direction
                    in_frame[card] = (detections[card], direction) 
            for card in cards_to_remove:
                to_be_added.pop(card)

def grab_and_compute_barycenter_thread(cap):
    global frame
    global in_frame
    global barycenter_history
    frame = cv2.resize(cap.raw_read(), (1280, 720))
    prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    prev_gray = cv2.GaussianBlur(prev_gray, (21, 21), 0)
    barycenter_history = []

    while True:
        with lock:
            frame = cv2.resize(cap.raw_read(), (1280, 720))
            frame_copy = frame.copy()
            in_frame_copy = in_frame.copy()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        # Compute delta frame
        delta_frame = cv2.absdiff(prev_gray, gray)

        # Threshold delta frame
        thresh_frame = cv2.threshold(delta_frame, 25, 255, cv2.THRESH_BINARY)[1]

        # Compute barycenter
        M = cv2.moments(thresh_frame)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            barycenter = (cX, cY)
            barycenter_history.append(barycenter)
            if len(barycenter_history) > 20:
                barycenter_history.pop(0)

            # Draw barycenter on frame
            cv2.circle(frame, barycenter, 5, (0, 255, 0), -1)

        # Display frame
        cv2.imshow("Frame", frame)
        
        # Update previous frame
        prev_gray = gray

        img = np.zeros((500, 500, 3), dtype=np.uint8)
        y = 20
        for key in in_frame.keys():
            cv2.putText(img, f"{key} came from {in_frame[key][1]}", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y += 20
        cv2.imshow("In Frame", img)
        # Break loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

with MainCam(config.MAINCAM_INDEX, config.MAINCAM_WIDTH, config.MAINCAM_HEIGHT) as cap:
    frame = cv2.resize(cap.raw_read(), (1280, 720))
    det = cap.detect(frame, bidding=True)

    detection_t = threading.Thread(target=detection_thread, args=(cap,))
    grab_and_compute_t = threading.Thread(target=grab_and_compute_barycenter_thread, args=(cap,))

    detection_t.start()
    grab_and_compute_t.start()

    detection_t.join()
    grab_and_compute_t.join()"""

"""
import cv2
import numpy as np
from src.vision.maincam import MainCam
import config
from collections import defaultdict
import threading


add_threshold = 3
remove_threshold = 4

frame = None
det = None
in_frame = {}
to_be_added = {}
to_be_removed = {}
barycenter_history = []
lock = threading.Lock()
stop_flag = False

def detection_thread():
    global frame
    global det
    global in_frame
    global to_be_added
    global to_be_removed
    global barycenter_history
    global stop_flag
    while not stop_flag:
        det = cap.detect(frame, bidding=True)
        detections = defaultdict(list)
        for card, top_left, bottom_right in det:
            detections[card].append((top_left, bottom_right))
        for card, boxes in detections.items():
            if card in in_frame:
                in_frame[card] = (boxes, in_frame[card][1])
            elif card not in to_be_added:
                to_be_added[card] = 0
        for card in in_frame:
            if card in detections:
                continue
            if card in to_be_removed:
                to_be_removed[card] += 1
            else:
                to_be_removed[card] = 0
        cards_to_remove = []
        for card in to_be_removed:
            if card in detections:
                cards_to_remove.append(card)
                continue
            if to_be_removed[card] == remove_threshold:
                cards_to_remove.append(card)
        for card in cards_to_remove:
            to_be_removed.pop(card)
            in_frame.pop(card)
        cards_to_remove = []
        for card in to_be_added:
            if card not in detections:
                cards_to_remove.append(card)
                continue
            if card in to_be_added:
                to_be_added[card] += 1
            else:
                to_be_added[card] = 0
            if len(barycenter_history) > 1 and to_be_added[card] == add_threshold:
                cards_to_remove.append(card)
                initial = barycenter_history[0]
                final = barycenter_history[-1]
                dx = final[0] - initial[0]
                dy = final[1] - initial[1]
                if abs(dx) > abs(dy):
                    if dx > 0:
                        direction = 'right'
                    else:
                        direction = 'left'
                else:
                    if dy > 0:
                        direction = 'bottom'
                    else:
                        direction = 'top'
                direction = f"{initial}->{final}" + direction
                in_frame[card] = (detections[card], direction) 
        for card in cards_to_remove:
            to_be_added.pop(card)

with MainCam(config.MAINCAM_INDEX, config.MAINCAM_WIDTH, config.MAINCAM_HEIGHT) as cap:
    frame = cv2.resize(cap.raw_read(), (1280, 720))
    prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    prev_gray = cv2.GaussianBlur(prev_gray, (21, 21), 0)
    detection_t = threading.Thread(target=detection_thread)
    detection_t.start()

    while True:
        frame = cv2.resize(cap.raw_read(), (1280, 720))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        # Compute delta frame
        delta_frame = cv2.absdiff(prev_gray, gray)
        # Threshold delta frame
        thresh_frame = cv2.threshold(delta_frame, 25, 255, cv2.THRESH_BINARY)[1]
        # Compute barycenter
        M = cv2.moments(thresh_frame)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            barycenter = (cX, cY)
            barycenter_history.append(barycenter)
            if len(barycenter_history) > 5:
                barycenter_history.pop(0)
            # Draw barycenter on frame
            for n, bc in enumerate(barycenter_history):
                cv2.circle(frame, bc, 5, (0, 255-n*5, n*5), -1)
        cv2.imshow("Frame", frame)
        img = np.zeros((500, 500, 3), dtype=np.uint8)
        y = 20
        for key in in_frame.keys():
            cv2.putText(img, f"{key} came from {in_frame[key][1]}", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y += 20
        cv2.imshow("In Frame", img)
        prev_gray = gray
        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_flag
            break
    
    cv2.destroyAllWindows()"""