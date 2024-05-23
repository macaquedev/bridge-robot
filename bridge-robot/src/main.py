from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit, Namespace
import cv2
from shapely.geometry import Point
from src.vision.maincam import MainCam
from src.communication.arduino import Arduino
import config
import base64
from multiprocessing import Process, Queue, Manager, Pool
from queue import Empty, Full
from collections import defaultdict
import random

import numpy as np


def remove_white_borders(img):
    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the coordinates of non-white (i.e., colored) pixels.
    colored_pixels = np.where(gray != 255)

    # Get the minimum and maximum coordinates.
    y_min, y_max = np.min(colored_pixels[0]), np.max(colored_pixels[0])
    x_min, x_max = np.min(colored_pixels[1]), np.max(colored_pixels[1])

    # Crop the image using the coordinates.
    cropped_img = img[y_min:y_max+1, x_min:x_max+1]

    return cropped_img


def bid_value(card):
    return [
        None, '1C', '1D', '1H', '1S', '1NT',
        '2C', '2D', '2H', '2S', '2NT',
        '3C', '3D', '3H', '3S', '3NT',
        '4C', '4D', '4H', '4S', '4NT',
        '5C', '5D', '5H', '5S', '5NT',
        '6C', '6D', '6H', '6S', '6NT',
        '7C', '7D', '7H', '7S', '7NT'
    ].index(card)


def arduino_process(queue):
    print("Connecting to Arduino....")
    arduino = Arduino(config.ARDUINO_BAUDRATE)
    print("Arduino connected!")
    while True:
        command = queue.get()
        if command and arduino.currently_acknowledged:
            arduino.send_data(command)


def camera_process(input_queue, output_queue):
    curr_mode = None
    with MainCam(config.MAINCAM_INDEX, config.MAINCAM_WIDTH, config.MAINCAM_HEIGHT) as cap:
        while True:
            try:
                event = input_queue.get(timeout=0)
                if event["type"] == "set_mode":
                    curr_mode = event["value"]
                else:
                    print(event)
            except Empty:
                pass
            frame = cap.raw_read()
            match curr_mode:
                case "auction":
                    image = cap.preprocess_image(frame)
                    det = cap.detect(image, bidding=True, preprocess=False)
                    try:
                        output_queue.put((image, det))
                    except Full:
                        output_queue.get()
                        output_queue.put(image, det)
                case "cardplay":
                    d = cap.detect_cards(frame)
                    try:
                        output_queue.put(d)
                    except Full:
                        output_queue.get()
                        output_queue.put(d)
                case _:
                    d = cap.preprocess_image(frame)
                    try:
                        output_queue.put(d, None)
                    except Full:
                        output_queue.get()
                        output_queue.put(d)

def auction_mode(incoming_frame_queue, outgoing_frame_queue, event_queue, frontend_queue, dealer, northDirection, vulnerability, result_queue):  
    try:
        highest_bids = [(None, None)]
        doubled = False
        redoubled = False
        num_passes = 0
        in_frame = defaultdict(list)
        to_be_added = defaultdict(list)
        to_be_removed = defaultdict(list)
        directions = ["North", "East", "South", "West"]
        directions = directions[["top", "left", "bottom", "right"].index(northDirection):] + directions[:["top", "left", "bottom", "right"].index(northDirection)]
        dealer = directions.index(dealer)
        auction = [[], [], [], []]
        current_turn = dealer
        current_error = None
        while True:
            try:
                incoming_frame_queue.get(timeout=0)
            except Empty:
                break
        while True:
            try:
                frame, det = incoming_frame_queue.get(timeout=0)
                if det is None:
                    continue
            except (Empty, ValueError):
                continue
            #print(directions[current_turn])
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
                    if to_be_added[card][i][0] == config.AUCTION_ADD_THRESHOLD:
                        if card not in in_frame:
                            in_frame[card] = []
                        in_frame[card].append([to_be_added[card][i][1], to_be_added[card][i][2], None])
                        box = in_frame[card][-1]
                        x_pos = box[0][0] + (box[1][0] - box[0][0]) // 2
                        y_pos = box[0][1] + (box[1][1] - box[0][1]) // 2
                        p = Point(x_pos, y_pos)
                        if config.AUCTION_TOP_POLYGON.contains(p):
                            player = 0
                        elif config.AUCTION_RIGHT_POLYGON.contains(p):
                            player = 1
                        elif config.AUCTION_BOTTOM_POLYGON.contains(p):
                            player = 2
                        elif config.AUCTION_LEFT_POLYGON.contains(p):
                            player = 3
                        else:
                            player = None
                        to_be_added[card].pop(i)
                        in_frame[card][-1][2] = player 
                        #event_queue.put({'type': 'add_bid', 'card': card, 'player': directions[player]})
                        break
                    
            for card in to_be_removed:
                to_remove = set()
                for i in range(len(to_be_removed[card])):
                    to_be_removed[card][i][0] += 1
                    if to_be_removed[card][i][0] >= config.AUCTION_REMOVE_THRESHOLD:
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
                    cv2.putText(frame, f"{directions[player] if player else 'Centre'} played {card}", (top_left[0], top_left[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    #cv2.circle(frame, center, 5, (0, 255, 0), -1)
                    if player is None:
                        current_error = (card, top_left, bottom_right)

            # two cases, either a bid was made or not. 
            # if a bid was made, we need to update the current turn and the auction list
            # if no bid was made, we need to check if the current player has a bid in the auction list
            # auction["player"] is a list of cards sorted by coordinate
            played = False
            
            if current_error:
                card, top_left, bottom_right = current_error
                for i in range(len(in_frame[card])):
                    if (top_left[0] - in_frame[card][i][0][0]) ** 2 + (top_left[1] - in_frame[card][i][0][1]) ** 2 < config.COALESCE_DISTANCE_SQUARED:
                        current_error = (card, in_frame[card][i][0], in_frame[card][i][1])
                        break
                else:
                    current_error = None
                    event_queue.put({'type': 'delete_error'})
            else:
                by_player = [[], [], [], []]
                for card in in_frame:
                    for top_left, bottom_right, player in in_frame[card]:
                        by_player[player].append((card, top_left, bottom_right))
                for i in range(4):
                    by_player[i].sort(key=config.SORTFUNCS[i])
                for i in range(4):
                    if not by_player[i]:
                        continue
                    
                    card, top_left, bottom_right = by_player[i][-1]

                    for j in range(len(auction[i])):
                        other_card, other_top_left, _ = auction[i][j]
                        if ((top_left[0] - other_top_left[0]) ** 2 + (top_left[1] - other_top_left[1]) ** 2 < config.COALESCE_DISTANCE_SQUARED
                            ) or (other_card == card and card not in ["Pass", "Double", "Redouble"]):
                            if card == other_card:
                                auction[i][j] = (other_card, top_left, bottom_right)
                            break
                    else:
                        if i == current_turn:
                            highest_bid = highest_bids[-1]
                            if (card == "Double" and ((not highest_bid[0]) or (highest_bid[1] % 2 == i % 2) or doubled or redoubled)
                                ) or (card == "Redouble" and ((not highest_bid[0]) or (highest_bid[1] % 2 != i % 2) or (not doubled) or redoubled)
                                ) or (card not in ["Pass", "Double", "Redouble"] and bid_value(highest_bid[0]) >= bid_value(card)):
                                current_error = (card, top_left, bottom_right)
                                event_queue.put({'type': 'error', 'card': card, 'player': directions[i]})

                            else: 
                                auction[i].append((card, top_left, bottom_right))
                                if card not in ["Pass", "Double", "Redouble"]:
                                    highest_bids.append((card, i))
                                event_queue.put({'type': 'new_bid', 'card': card, 'player': directions[i]})
                                if card == "Pass":
                                    num_passes += 1
                                    if not highest_bid[0]:
                                        if num_passes == 4:
                                            event_queue.put({'type': 'end_auction', 'passout': True})
                                            result_queue.put({'type': 'end_auction', 'passout': True})
                                    else:
                                        if num_passes == 3:
                                            contract = highest_bid[0]
                                            suit = contract[1]
                                            for (bid, player) in highest_bids:
                                                if bid and bid[1] == suit and (player % 2) == (highest_bid[1] % 2):
                                                    declarer = player
                                                    break
                                                
                                            event_queue.put({'type': 'end_auction', 'passout': False, 'contract': highest_bid[0], 'declarer': directions[declarer], 'doubled': doubled, 'redoubled': redoubled})
                                            result_queue.put({'type': 'end_auction', 'passout': False, 'contract': highest_bid[0], 'declarer': directions[declarer], 'doubled': doubled, 'redoubled': redoubled})
                                else:
                                    num_passes = 0
                                    if card == "Double":
                                        doubled = True
                                    elif card == "Redouble":
                                        doubled = False
                                        redoubled = True
                                    else:
                                        doubled = False
                                        redoubled = False

                                played = True
                        else:
                            current_error = (card, top_left, bottom_right)
                            event_queue.put({'type': 'error', 'card': card, 'player': directions[i]})
            cv2.polylines(frame, [np.array(config.TOP_POLYGON.exterior.coords).astype(int)], isClosed=True, color=(0, 0, 255), thickness=2)
            cv2.polylines(frame, [np.array(config.LEFT_POLYGON.exterior.coords).astype(int)], isClosed=True, color=(0, 255, 255), thickness=2)
            cv2.polylines(frame, [np.array(config.BOTTOM_POLYGON.exterior.coords).astype(int)], isClosed=True, color=(0, 255, 0), thickness=2)
            cv2.polylines(frame, [np.array(config.RIGHT_POLYGON.exterior.coords).astype(int)], isClosed=True, color=(255, 255, 0), thickness=2)

            if played:
                current_turn = (current_turn + 1) % 4
            
            ret, buffer = cv2.imencode('.jpg', cv2.resize(remove_white_borders(frame), (320, 180)))
            if not ret:
                continue
            outgoing_frame_queue.put(base64.b64encode(buffer).decode('utf-8'))
            try:
                event = frontend_queue.get(timeout=0)
                if event == 'undo':
                    current_turn = (current_turn + 3) % 4
                    if auction[current_turn]:
                        card, top_left, bottom_right = auction[current_turn].pop()
                        for i in range(len(in_frame[card])):
                            if (top_left[0] - in_frame[card][i][0][0]) ** 2 + (top_left[1] - in_frame[card][i][0][1]) ** 2 < config.COALESCE_DISTANCE_SQUARED:
                                in_frame[card].pop(i)
                                break
                        for i in range(len(to_be_removed[card])):
                            if (top_left[0] - to_be_removed[card][i][1][0]) ** 2 + (top_left[1] - to_be_removed[card][i][1][1]) ** 2 < config.COALESCE_DISTANCE_SQUARED:
                                to_be_removed[card].pop(i)
                                break
                        if highest_bids[-1] == (card, current_turn):
                            highest_bids.pop()
                        if card == "Pass":
                            num_passes -= 1
                        elif card == "Double":
                            doubled = False
                        elif card == "Redouble":
                            redoubled = False
                            doubled = True
                    current_error = None
                    event_queue.put({"type": "delete_error"})
            except Empty:
                pass
    except Exception as e: 
        print("Error in auction_mode: ", e)
        raise


def cardplay_mode(incoming_frame_queue, outgoing_frame_queue, event_queue, frontend_queue, northDirection, event, result_queue):  
    print("entered cardplay")
    in_frame = defaultdict(list)
    to_be_added = defaultdict(list)
    to_be_removed = defaultdict(list)
    directions = ["North", "East", "South", "West"]
    directions = directions[["top", "left", "bottom", "right"].index(northDirection):] + directions[:["top", "left", "bottom", "right"].index(northDirection)]
    tricks = [[]]
    prev_on_table = [None, None, None, None]
    current_trick_index = 1
    num_people_played = 0
    on_lead = (directions.index(event["declarer"]) + 1) % 4
    suit_led = None
    current_turn = on_lead
    current_error = None
    just_played_trick = True
    def is_valid_card(card):
        return (not suit_led) or (suit_led == card[-1]) or (card[-1] == event["contract"][1]) # nothing's been led, or the card is in the suit led, or the card is a trump card
    

    while True:
        try:
            incoming_frame_queue.get(timeout=0)
        except Empty:
            break
    while True:
        try:
            thresh, detector_frame, det = incoming_frame_queue.get(timeout=0)
            if det is None:
                continue
        except Empty:
            continue
        #print(current_turn, on_lead, suit_led, tricks, just_played_trick, num_people_played)
        print([[j[0] for j in i] for i in tricks])
        new_det = []
        for card, polygon in det:
            if card == "No Detection":
                continue
            M = cv2.moments(polygon)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                continue
            new_det.append((card, (cx, cy)))
        det = new_det
        if num_people_played == 4:
            just_played_trick = True
            current_trick_index += 1
            tricks.append([])
            num_people_played = 0
            suit_led = None
            prev_on_table = [None, None, None, None]
            if current_trick_index == 14:
                break
        if just_played_trick and len(new_det) != 0:
            continue
        just_played_trick = False
        for card, top_left in det:
            for i in range(len(in_frame[card])):
                box = in_frame[card][i][0]
                if (top_left[0] - box[0]) ** 2 + (top_left[1] - box[1]) ** 2 < config.COALESCE_DISTANCE_SQUARED:
                    in_frame[card][i] = [top_left, in_frame[card][i][1]]
                    break
            else:
                for i in range(len(to_be_added[card])):
                    box = to_be_added[card][i][1]
                    if (top_left[0] - box[0]) ** 2 + (top_left[1] - box[1]) ** 2 < config.COALESCE_DISTANCE_SQUARED:
                        break
                else:
                    to_be_added[card] = [[0, top_left]]
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
                for det_card, position in det:
                    if det_card == card and (position[0] - in_frame[card][i][0][0]) ** 2 + (position[1] - in_frame[card][i][0][1]) ** 2 < config.COALESCE_DISTANCE_SQUARED:
                        to_remove.remove(i)
                        break
            for i in to_remove:
                if card not in to_be_removed:
                    to_be_removed[card] = []
                for j in range(len(to_be_removed[card])):
                    if (in_frame[card][i][0][0] - to_be_removed[card][j][1][0]) ** 2 + (in_frame[card][i][0][1] - to_be_removed[card][j][1][1]) < config.COALESCE_DISTANCE_SQUARED:
                        break    
                    
                else:
                    to_be_removed[card].append([0, in_frame[card][i][0]])
        for card in to_be_added:
            for i in range(len(to_be_added[card])):
                to_be_added[card][i][0] += 1
                if to_be_added[card][i][0] == config.CARDPLAY_ADD_THRESHOLD:
                    if card not in in_frame:
                        in_frame[card] = []
                    in_frame[card].append([to_be_added[card][i][1], None])
                    box = in_frame[card][-1][0]
                    x_pos = box[0]
                    y_pos = box[1]
                    p = Point(x_pos, y_pos)
                    if config.CARDPLAY_TOP_POLYGON.contains(p):
                        player = 0
                    elif config.CARDPLAY_RIGHT_POLYGON.contains(p):
                        player = 1
                    elif config.CARDPLAY_BOTTOM_POLYGON.contains(p):
                        player = 2
                    elif config.CARDPLAY_LEFT_POLYGON.contains(p):
                        player = 3
                    else:
                        player = None
                    to_be_added[card].pop(i)
                    in_frame[card][-1][1] = player 
                    #event_queue.put({'type': 'add_bid', 'card': card, 'player': directions[player]})
                    break
                
        for card in to_be_removed:
            to_remove = set()
            for i in range(len(to_be_removed[card])):
                to_be_removed[card][i][0] += 1
                if to_be_removed[card][i][0] >= config.CARDPLAY_REMOVE_THRESHOLD:
                    to_remove.add(i)
                    break
            for i in to_remove:
                top_left = to_be_removed[card][i][1]
                for j in range(len(in_frame[card])):
                    if (top_left[0] - in_frame[card][j][0][0]) ** 2 + (top_left[1] - in_frame[card][j][0][1]) ** 2 < config.COALESCE_DISTANCE_SQUARED:
                        in_frame[card].pop(j)
                        break
                to_be_removed[card].pop(i)
        person_played = [False, False, False, False]
        for card in in_frame:
            for position, player in in_frame[card]:
                if player is None or person_played[player]:
                    current_error = (card, position, position)
                else:
                    person_played[player] = (card, position)
        played = False
        if current_error:
            card, top_left, bottom_right = current_error
            for i in range(len(in_frame[card])):
                if (top_left[0] - in_frame[card][i][0][0]) ** 2 + (top_left[1] - in_frame[card][i][0][1]) ** 2 < config.COALESCE_DISTANCE_SQUARED:
                    current_error = (card, in_frame[card][i][0], in_frame[card][i][0])
                    break
            else:
                current_error = None
                event_queue.put({'type': 'delete_error'})
        else:
            diff = [None, None, None, None]
            for i in range(4):
                if person_played[i] and not prev_on_table[i]:
                    diff[i] = person_played[i]
            for i in range(4):
                if not diff[i]:
                    continue
                card, position = diff[i]
                if i == current_turn:
                    if not is_valid_card(card):
                        current_error = (card, position, position)
                        event_queue.put({'type': 'error', 'card': card, 'player': directions[i]})
                    else:
                        tricks[-1].append((card, position))
                        if not suit_led:
                            suit_led = card[-1]
                        
                        played = True
                        prev_on_table[i] = person_played[i]

                else:
                    current_error = (card, position, position)
                    event_queue.put({'type': 'error', 'card': card, 'player': directions[i]})
        if played:
            current_turn = (current_turn + 1) % 4
            num_people_played += 1 

        cv2.polylines(thresh, [np.array(config.CARDPLAY_TOP_POLYGON.exterior.coords).astype(int)], isClosed=True, color=(255, 255, 255), thickness=10)
        cv2.polylines(thresh, [np.array(config.CARDPLAY_LEFT_POLYGON.exterior.coords).astype(int)], isClosed=True, color=(255, 255, 255), thickness=10)
        cv2.polylines(thresh, [np.array(config.CARDPLAY_BOTTOM_POLYGON.exterior.coords).astype(int)], isClosed=True, color=(255, 255, 255), thickness=10)
        cv2.polylines(thresh, [np.array(config.CARDPLAY_RIGHT_POLYGON.exterior.coords).astype(int)], isClosed=True, color=(255, 255, 255), thickness=10)

        
        ret, buffer = cv2.imencode('.jpg', cv2.resize(remove_white_borders(thresh), (320, 180)))
        if not ret:
            continue
        outgoing_frame_queue.put(base64.b64encode(buffer).decode('utf-8'))
        try:
            event = frontend_queue.get(timeout=0)
            if event == 'undo':
                current_turn = (current_turn + 3) % 4
                if auction[current_turn]:
                    card, top_left, bottom_right = auction[current_turn].pop()
                    for i in range(len(in_frame[card])):
                        if (top_left[0] - in_frame[card][i][0][0]) ** 2 + (top_left[1] - in_frame[card][i][0][1]) ** 2 < config.COALESCE_DISTANCE_SQUARED:
                            in_frame[card].pop(i)
                            break
                    for i in range(len(to_be_removed[card])):
                        if (top_left[0] - to_be_removed[card][i][1][0]) ** 2 + (top_left[1] - to_be_removed[card][i][1][1]) ** 2 < config.COALESCE_DISTANCE_SQUARED:
                            to_be_removed[card].pop(i)
                            break
                    if highest_bids[-1] == (card, current_turn):
                        highest_bids.pop()
                    if card == "Pass":
                        num_passes -= 1
                    elif card == "Double":
                        doubled = False
                    elif card == "Redouble":
                        redoubled = False
                        doubled = True
                current_error = None
                event_queue.put({"type": "delete_error"})
        except Empty:
            pass
        # Update previous frame
        #print(to_be_added, in_frame, to_be_removed)
        #img = np.zeros((500, 500, 3), dtype=np.uint8)
        #y = 20
        #for card_label in in_frame:
        #    for top_left, bottom_right, player in in_frame[card_label]:
        #        cv2.putText(img, f"{player} played {card_label}", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        #        y += 20
        #cv2.imshow("In Frame", img)
        ## Break loop if 'q' is pressed
        #if cv2.waitKey(1) & 0xFF == ord('q'):
        #    break


def create_app():
    manager = Manager()
    arduino_queue = manager.Queue()
    camera_output_queue = manager.Queue(maxsize=10)
    camera_input_queue = manager.Queue()
    p_arduino = Process(target=arduino_process, args=(arduino_queue,))
    p_camera = Process(target=camera_process, args=(camera_input_queue, camera_output_queue))
    p_arduino.start()
    p_camera.start()
    app = Flask(__name__)
    socketio = SocketIO(app, async_mode="threading", cors_allowed_origins='*')
    

    @app.route('/send_command', methods=['POST'])
    def send_command():
        command = request.form.get('command')
        if command:
            arduino_queue.put(command)
            return 'Command sent!'
        else:
            return 'No command received', 400
        
    class MyNamespace(Namespace):
        frontend_queue = Queue()
        def on_connect(self):
            pass

        def on_undo(self):
            self.frontend_queue.put("undo")

        def on_setup(self, data):
            northDirection = data.get('northDirection')
            dealer = data.get('dealer')
            vulnerability = data.get('vulnerability')

            def send_frames(output_queue):
                with Manager() as manager:
                    frame_queue = manager.Queue()
                    event_queue = manager.Queue()
                    result_queue = manager.Queue()
                    camera_input_queue.put({'type': 'set_mode', 'value': 'auction'})
                    auction_process = Process(target=auction_mode, args=(camera_output_queue, frame_queue, event_queue, self.frontend_queue, dealer, northDirection, vulnerability, result_queue))
                    auction_process.start()  # Start the process
                    cardplay_process = None
                    auction_result = None
                    cardplay_result = None
                    while True:
                        frame = frame_queue.get()
                        self.emit('image', frame)
                        try:
                            event = event_queue.get(timeout=0)
                            self.emit('event', event)
                        except Empty:
                            pass

                        try:
                            result = result_queue.get(timeout=0)
                            if auction_process is not None:
                                auction_result = result
                                auction_process.terminate()
                                auction_process.join()
                                auction_process = None
                                if result["passout"]:
                                    print("PASSOUT")
                                    camera_input_queue.put({'type': 'set_mode', 'value': 'none'})
                                else:
                                    camera_input_queue.put({'type': 'set_mode', 'value': 'cardplay'})
                                    cardplay_process = Process(target=cardplay_mode, args=(camera_output_queue, frame_queue, event_queue, self.frontend_queue, northDirection, result, result_queue))
                                    cardplay_process.start()
                            else:
                                cardplay_result = result
                                cardplay_process.join()
                                cardplay_process = None
                                break
                        except Empty:
                            #print(f"still doing stuff {random.randint(1, 10000)}")
                            pass
                        except Exception as e:
                            print(e)
                            raise
                    
                    output_queue.put(auction_result)
                    output_queue.put(cardplay_result)

            output_queue = manager.Queue()
            socketio.start_background_task(send_frames, output_queue)

    socketio.on_namespace(MyNamespace('/'))
    @app.route('/watch')
    def watch():
        return render_template('watch.html')
    @app.route('/config')
    def config_():
        return render_template('config.html')
    @app.route('/')
    def index():
        return render_template('index.html')
    
    @app.route('/currently_playing/<int:boardnumber>')
    def currently_playing(boardnumber):
        dealer = request.args.get('dealer')
        vulnerability = request.args.get('vulnerability')
        return render_template('currently_playing.html', boardnumber=boardnumber, dealer=dealer, vulnerability=vulnerability) 
    return app, socketio

def main():
    app, socketio = create_app()
    socketio.run(app, host='0.0.0.0', port=5000)

if __name__ == '__main__':
    main()