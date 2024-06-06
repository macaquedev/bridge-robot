from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit, Namespace
import cv2
from shapely.geometry import Point
from src.vision.maincam import MainCam
from src.vision.armcam import ArmCam
from src.communication.arduino import Arduino
import config
import base64
from multiprocessing import Process, Queue, Manager, Pool
from queue import Empty, Full
from collections import defaultdict
import random
import flask_profiler
import numpy as np
import json
from src import utils


def arduino_process(queue):
    print("Connecting to Arduino....")
    arduino = Arduino(config.ARDUINO_BAUDRATE)
    with ArmCam(config.ARMCAM_INDEX, config.ARMCAM_WIDTH, config.ARMCAM_HEIGHT) as armcam:
        while True:
            command = queue.get()
            if command and arduino.currently_acknowledged:
                if command == "CALIBRATE":
                    armcam.calibrate(arduino)
                elif command.startswith("PLAYCARD"):
                    armcam.play_card(command.split(" ")[1], arduino)
                else:
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
    highest_bids = [(None, None)]
    doubled = False
    redoubled = False
    num_passes = 0
    in_frame = []
    to_be_added = []
    to_be_removed = []
    directions = ["North", "East", "South", "West"]
    directions = directions[["top", "left", "bottom", "right"].index(northDirection):] + directions[:["top", "left", "bottom", "right"].index(northDirection)]
    dealer = directions.index(dealer)
    prev_auction = [[], [], [], []]
    current_turn = dealer
    current_error = None
    
    while True:
        try:
            incoming_frame_queue.get(timeout=0)
        except Empty:
            break
    
    while True:
        auction = [[], [], [], []]

        try:
            frame, det = incoming_frame_queue.get(timeout=0)
            if det is None:
                continue
        except (Empty, ValueError):
            continue

        not_updated_in_to_add = set(list(range(len(to_be_added))))
        for i in range(len(det)):
            card, top_left, bottom_right = det[i]
            for j in range(len(to_be_removed)):
                if to_be_removed[j][1] == card and utils.near(to_be_removed[j][2], top_left):
                    to_be_removed.pop(j)
                    break
            else:
                for j in range(len(to_be_added)):
                    if to_be_added[j][1] == card and utils.near(to_be_added[j][2], top_left):
                        not_updated_in_to_add.remove(j)
                        to_be_added[j][0] += 1
                        if to_be_added[j][0] == config.AUCTION_ADD_THRESHOLD:
                            not_updated_in_to_add.add(j)
                            in_frame.append(to_be_added[j][1:])
                        break
                else:
                    for j in range(len(in_frame)):
                        if in_frame[j][0] == card and utils.near(in_frame[j][1], top_left):
                            in_frame[j][1] = top_left
                            in_frame[j][2] = bottom_right
                            break
                    else:
                        to_be_added.append([0, card, top_left, bottom_right])
        
        for i in range(len(in_frame)-1, -1, -1):
            for j in range(len(det)):
                if utils.near(det[j][1], in_frame[i][1]):
                    break
            else:
                for j in range(len(to_be_removed)-1, -1, -1):
                    if utils.near(to_be_removed[j][2], in_frame[i][1]):
                        break
                else:
                    to_be_removed.append([0] + in_frame[i])

        not_updated_in_to_add = sorted(list(not_updated_in_to_add), reverse=True)
        for i in not_updated_in_to_add:
            to_be_added.pop(i)
        for i in range(len(to_be_removed)-1, -1, -1):
            to_be_removed[i][0] += 1
            if to_be_removed[i][0] == config.AUCTION_REMOVE_THRESHOLD:
                for j in range(len(in_frame)):
                    if utils.near(in_frame[j][1], to_be_removed[i][2]):
                        in_frame.pop(j)
                        break
                to_be_removed.pop(i)
            
        for card, top_left, bottom_right in in_frame:
            box = [top_left, bottom_right]
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
                continue
            
            auction[player].append((card, top_left, bottom_right))
                
            #for card, top_left, bottom_right, player in in_frame[card]:
            #    cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)
            #    cv2.putText(frame, f"{directions[player] if player else 'Centre'} played {card}", (top_left[0], top_left[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            #    #cv2.circle(frame, center, 5, (0, 255, 0), -1)
            #    if player is None:
            #        current_error = (card, top_left, bottom_right)
        played = False
        
        if current_error:
            card, top_left, bottom_right = current_error
            for i in range(len(in_frame)):
                if utils.near(in_frame[i][1], top_left) and card == in_frame[i][0]:
                    break
            else:
                current_error = None
                event_queue.put({'type': 'delete_error'})
        else:
            for player in range(4):
                for i in range(len(auction[player])-1, -1, -1):
                    for j in range(len(prev_auction[player])):
                        if utils.near(prev_auction[player][j][1], auction[player][i][1]):
                            previous_card = auction[player].pop(i)
                            prev_auction[player][j] = (prev_auction[player][j][0], previous_card[1], previous_card[2])
                            break
                        if auction[player][i][0] == prev_auction[player][j][0] and auction[player][i][0] not in ["Pass", "Double", "Redouble"]:
                            previous_card = auction[player].pop(i)
                            prev_auction[player][j] = (prev_auction[player][j][0], previous_card[1], previous_card[2])
                            break

            for player in range(4):
                if len(auction[player]) > 1:
                    current_error = auction[player][0]
                    event_queue.put({'type': 'error', 'message': f"{directions[player]} has made multiple bids on one turn."})
                    break
                elif len(auction[player]) == 0:
                    continue
                card, top_left, bottom_right = auction[player][0]

                if player == current_turn:
                    highest_bid = highest_bids[-1]
                    if (card == "Double" and ((not highest_bid[0]) or (highest_bid[1] % 2 == player % 2) or doubled or redoubled)
                        ) or (card == "Redouble" and ((not highest_bid[0]) or (highest_bid[1] % 2 != player % 2) or (not doubled) or redoubled)
                        ) or (card not in ["Pass", "Double", "Redouble"] and utils.bid_value(highest_bid[0]) >= utils.bid_value(card)):
                        current_error = (card, top_left, bottom_right)
                        event_queue.put({'type': 'error', 'message': f"Invalid bid: {card} by {directions[player]}."})

                    else: 
                        prev_auction[player].append((card, top_left, bottom_right))
                        if card not in ["Pass", "Double", "Redouble"]:
                            highest_bids.append((card, player))
                        event_queue.put({'type': 'new_bid', 'card': card, 'player': directions[player]})
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
                    event_queue.put({'type': 'error', 'message': f"Play out of turn: {card} by {directions[player]}."})

        cv2.polylines(frame, [np.array(config.AUCTION_TOP_POLYGON.exterior.coords).astype(int)], isClosed=True, color=(0, 0, 255), thickness=2)
        cv2.polylines(frame, [np.array(config.AUCTION_LEFT_POLYGON.exterior.coords).astype(int)], isClosed=True, color=(0, 255, 255), thickness=2)
        cv2.polylines(frame, [np.array(config.AUCTION_BOTTOM_POLYGON.exterior.coords).astype(int)], isClosed=True, color=(0, 255, 0), thickness=2)
        cv2.polylines(frame, [np.array(config.AUCTION_RIGHT_POLYGON.exterior.coords).astype(int)], isClosed=True, color=(255, 255, 0), thickness=2)
        for player in range(4):
            for card, top_left, bottom_right in prev_auction[player]:
                cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)
                cv2.putText(frame, card, (top_left[0], top_left[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        if played:
            current_turn = (current_turn + 1) % 4
        
        ret, buffer = cv2.imencode('.jpg', cv2.resize(utils.remove_white_borders(frame), (320, 180)))
        if not ret:
            continue
        outgoing_frame_queue.put(base64.b64encode(buffer).decode('utf-8'))
        try:
            event = frontend_queue.get(timeout=0)
            if event == 'undo':
                current_turn = (current_turn + 3) % 4
                if prev_auction[current_turn]:
                    card, top_left, bottom_right = prev_auction[current_turn].pop()
                    for i in range(len(in_frame)):
                        if utils.near(top_left, in_frame[i][1]):
                            in_frame.pop(i)
                            break
                    for i in range(len(to_be_removed)):
                        if utils.near(top_left, to_be_removed[i][2]):
                            in_frame.pop(i)
                            break
                    print(highest_bids[-1], (card, current_turn))
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


def cardplay_mode(incoming_frame_queue, outgoing_frame_queue, event_queue, frontend_queue, northDirection, event, result_queue):  
    print("entered cardplay")
    in_frame = []
    to_be_added = []
    to_be_removed = []
    prev_cards = [[], [], [], []]
    directions = ["North", "East", "South", "West"]
    directions = directions[["top", "left", "bottom", "right"].index(northDirection):] + directions[:["top", "left", "bottom", "right"].index(northDirection)]
    tricks = []
    current_trick_index = 1
    num_people_played = 0
    dummy = (directions.index(event["declarer"]) + 2) % 4
    on_lead = (directions.index(event["declarer"]) + 1) % 4
    suit_led = None
    current_turn = on_lead
    trump_suit = event["contract"][1]
    already_played = {}
    ns_tricks = 0
    ew_tricks = 0
    current_error = None
    just_played_trick = True
    currently_placing_dummy = False
    dummy_cards = []
    def in_dummy(card):
        return card in [i[0] for i in dummy_cards]
    def is_valid_card(card):
        return card not in already_played
    while True:
        try:
            incoming_frame_queue.get(timeout=0)
        except Empty:
            break
    
    while True:
        cards = [[], [], [], []]
        try:
            thresh, detector_frame, det = incoming_frame_queue.get(timeout=0)
            if det is None:
                continue
        except (Empty, ValueError):
            continue
        
        print("in_frame: ", in_frame)
        print("error: ", current_error)
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
            for _, position in dummy_cards:
                if utils.near(position, (cx, cy), cardplay=True):
                    break
            else:
                new_det.append((card, (cx, cy)))
        det = new_det
        not_updated_in_to_add = set(list(range(len(to_be_added))))
        if num_people_played == 4:
            just_played_trick = True
            current_trick_index += 1
            num_people_played = 0
            suit_led = None
            cards = [[], [], [], []]
            if current_trick_index == 14:
                result_queue.put({'type': 'end_cardplay', 'result': ns_tricks if directions[event["declarer"]] in ["North", "South"] else ew_tricks})
                while True:
                    pass
                
        if just_played_trick:
            if len(det) != 0:
                continue
            else:
                if len(tricks) > 0:
                    curr_max_index = 0
                    for index, (card, _) in enumerate(tricks[-1]):
                        curr_max = tricks[-1][curr_max_index][0]
                        if curr_max[1] == trump_suit:
                            if card[1] != trump_suit:
                                continue
                            if utils.card_value(card[0]) < utils.card_value(curr_max[0]):
                                continue
                            curr_max_index = index
                        else:
                            if card[1] == trump_suit or (card[1] == curr_max[1] and utils.card_value(card[0]) > utils.card_value(curr_max[0])):
                                curr_max_index = index
                    on_lead = (on_lead + curr_max_index) % 4
                    current_turn = on_lead
                    if directions[on_lead] in ["North", "South"]:
                        ns_tricks += 1
                        event_queue.put({'type': 'set_tricks', 'side': "NS", 'number': ns_tricks})
                    else:
                        ew_tricks += 1
                        event_queue.put({'type': 'set_tricks', 'side': "EW", 'number': ew_tricks})
                tricks.append([])
                just_played_trick = False
                suit_led = None



        for i in range(len(det)):
            card, position = det[i]
            print(card, position)
            for j in range(len(to_be_removed)):
                if to_be_removed[j][1] == card and utils.near(to_be_removed[j][2], position, cardplay=True):
                    to_be_removed.pop(j)
                    break
            else:
                for j in range(len(to_be_added)):
                    if to_be_added[j][1] == card and utils.near(to_be_added[j][2], position, cardplay=True):
                        not_updated_in_to_add.remove(j)
                        to_be_added[j][0] += 1
                        if to_be_added[j][0] == config.AUCTION_ADD_THRESHOLD:
                            not_updated_in_to_add.add(j)
                            in_frame.append(to_be_added[j][1:])
                        break
                else:
                    for j in range(len(in_frame)):
                        if in_frame[j][0] == card and utils.near(in_frame[j][1], position, cardplay=True):
                            in_frame[j][1] = position
                            break
                    else:
                        to_be_added.append([0, card, position])
        
        for i in range(len(in_frame)-1, -1, -1):
            for j in range(len(det)):
                if utils.near(det[j][1], in_frame[i][1], cardplay=True):
                    break
            else:
                for j in range(len(to_be_removed)-1, -1, -1):
                    if utils.near(to_be_removed[j][2], in_frame[i][1], cardplay=True):
                        break
                else:
                    to_be_removed.append([0] + in_frame[i])

        not_updated_in_to_add = sorted(list(not_updated_in_to_add), reverse=True)
        for i in not_updated_in_to_add:
            to_be_added.pop(i)
        for i in range(len(to_be_removed)-1, -1, -1):
            to_be_removed[i][0] += 1
            if to_be_removed[i][0] == config.AUCTION_REMOVE_THRESHOLD:
                for j in range(len(in_frame)):
                    if utils.near(in_frame[j][1], to_be_removed[i][2], cardplay=True):
                        in_frame.pop(j)
                        break
                to_be_removed.pop(i)

        for card, position in in_frame:
            p = Point(*position)

            if config.CARDPLAY_TOP_POLYGONS[dummy].contains(p):
                player = 0
            elif config.CARDPLAY_RIGHT_POLYGONS[dummy].contains(p):
                player = 1
            elif config.CARDPLAY_BOTTOM_POLYGONS[dummy].contains(p):
                player = 2
            elif config.CARDPLAY_LEFT_POLYGONS[dummy].contains(p):
                player = 3
            else:
                player = None
            
            cards[player].append((card, position))

    
        played = False
        if current_error:
            card, position, position = current_error
            for i in range(len(in_frame)):
                if utils.near(in_frame[i][1], position, cardplay=True) and card == in_frame[i][0]:
                    break
            else:
                current_error = None
                event_queue.put({'type': 'delete_error'})
        else:
            for player in range(4):
                for i in range(len(cards[player])-1, -1, -1):
                    for j in range(len(prev_cards[player])):
                        if utils.near(prev_cards[player][j][1], cards[player][i][1], cardplay=True):
                            cards[player].pop(i)
                            break
                        if cards[player][i][0] == prev_cards[player][j][0]:
                            cards[player].pop(i)
                            break
                        
                    else:
                        for j in range(len(dummy_cards)):
                            if utils.near(dummy_cards[j][1], cards[player][i][1], cardplay=True) and current_turn == dummy:
                                cards[player].pop(i)
                                break
        
            for player in range(4):
                if len(cards[player]) > 1 and not currently_placing_dummy:
                    current_error = (cards[player][-1][0], cards[player][-1][1], cards[player][-1][1])
                    print("THE ERROR IS 1")
                    event_queue.put({'type': 'error', 'message':  f"{directions[player]} has played multiple cards."})
                    break
                elif len(cards[player]) == 0:
                    continue
                card, position = cards[player][-1]

                if player == current_turn:
                    if currently_placing_dummy:
                        for card, position in cards[player]:
                            print("HI", cards[player])   # cards[player] gives the old card with the new position!
                            if not is_valid_card(card):
                                current_error = (card, position, position)
                                event_queue.put({'type': 'error', 'message': f"{card} was led, so cannot be in the dummy."})
                            elif in_dummy(card):
                                current_error = (card, position, position)
                                event_queue.put({'type': 'error', 'message': f"{card} is already in the dummy."})
                            else:
                                dummy_cards.append((card, position))
                                event_queue.put({'type': 'place_dummy_card', 'card': card})
                                if len(dummy_cards) == 13:
                                    currently_placing_dummy = False
                                    event_queue.put({'type': 'end_dummy'})
                    else:
                        print(already_played, card)
                        if not is_valid_card(card):
                            current_error = (card, position, position)
                            event_queue.put({'type': 'error', 'message': f"{card} has already been played on trick {already_played[card]}."})
                        elif current_turn != dummy and in_dummy(card):
                            current_error = (card, position, position)
                            event_queue.put({'type': 'error', 'message': f"{card} is in the dummy."})
                        elif current_turn == dummy and not in_dummy(card):
                            current_error = (card, position, position)
                            event_queue.put({'type': 'error', 'message': f"{card} is not in the dummy."})
                        elif current_turn == dummy and suit_led != card[-1] and any(i[-1] == suit_led for i in dummy_cards):
                            current_error = (card, position, position)
                            event_queue.put({'type': 'error', 'message': f"{card} cannot be played - must follow suit!"})
                        else:
                            prev_cards[player].append((card, position))
                            tricks[-1].append((card, position))
                            if not suit_led:
                                suit_led = card[-1]
                            already_played[card] = current_trick_index
                            played = True

                else:
                    current_error = (card, position, position)
                    event_queue.put({'type': 'error', 'message': f"Play out of turn: {card} by {directions[player]}."})
                    print("THE ERROR IS 3")
        cv2.polylines(thresh, [np.array(config.CARDPLAY_TOP_POLYGONS[dummy].exterior.coords).astype(int)], isClosed=True, color=(0, 0, 255), thickness=10)
        cv2.polylines(thresh, [np.array(config.CARDPLAY_LEFT_POLYGONS[dummy].exterior.coords).astype(int)], isClosed=True, color=(0, 255, 255), thickness=10)
        cv2.polylines(thresh, [np.array(config.CARDPLAY_BOTTOM_POLYGONS[dummy].exterior.coords).astype(int)], isClosed=True, color=(0, 255, 0), thickness=10)
        cv2.polylines(thresh, [np.array(config.CARDPLAY_RIGHT_POLYGONS[dummy].exterior.coords).astype(int)], isClosed=True, color=(255, 255, 0), thickness=10)
        polygons = [i[dummy] for i in [config.CARDPLAY_TOP_POLYGONS, config.CARDPLAY_RIGHT_POLYGONS, config.CARDPLAY_BOTTOM_POLYGONS, config.CARDPLAY_LEFT_POLYGONS]]
        polygons = polygons[on_lead:] + polygons[:on_lead]
        for player, (polygon, (card, _)) in enumerate(zip(polygons, tricks[-1])):
            cv2.putText(thresh, card, utils.top_left(polygon), cv2.FONT_HERSHEY_SIMPLEX, 8, (43, 75, 238), 15)

        if currently_placing_dummy:
            polygon = [config.CARDPLAY_TOP_POLYGONS[dummy], config.CARDPLAY_RIGHT_POLYGONS[dummy], config.CARDPLAY_BOTTOM_POLYGONS[dummy], config.CARDPLAY_LEFT_POLYGONS[dummy]][current_turn]
            cv2.putText(thresh, f" ".join(i[0] for i in dummy_cards), utils.top_left(polygon), cv2.FONT_HERSHEY_SIMPLEX, 3, (43, 75, 238), 10)

        if played:
            current_turn = (current_turn + 1) % 4
            num_people_played += 1 
            if current_turn == dummy and current_trick_index == 1:
                currently_placing_dummy = True
                event_queue.put({'type': 'start_dummy'})
        ret, buffer = cv2.imencode('.jpg', cv2.resize(utils.remove_white_borders(thresh), (320, 180)))
        if not ret:
            continue
        outgoing_frame_queue.put(base64.b64encode(buffer).decode('utf-8'))
        try:
            event = frontend_queue.get(timeout=0)
            if event == 'undo' and current_turn != on_lead and len(tricks[-1]) > 0:
                if currently_placing_dummy:
                    if not dummy_cards:
                        currently_placing_dummy = False
                    else:
                        card, position = dummy_cards.pop()
                        for i in range(len(in_frame)):
                            if utils.near(position, in_frame[i][1], cardplay=True):
                                in_frame.pop(i)
                                break
                        for i in range(len(to_be_removed)):
                            if utils.near(position, to_be_removed[i][2], cardplay=True):
                                to_be_removed.pop(i)
                                break

                if not currently_placing_dummy:
                    current_turn = (current_turn + 3) % 4
                    num_people_played -= 1
                    card, position = prev_cards[current_turn].pop()
                    already_played.pop(card)
                    for i in range(len(in_frame)):
                        if utils.near(position, in_frame[i][1], cardplay=True):
                            in_frame.pop(i)
                            break
                    for i in range(len(to_be_removed)):
                        if utils.near(position, to_be_removed[i][2], cardplay=True):
                            to_be_removed.pop(i)
                            break
                    if tricks[-1][-1] == (card, position):
                        tricks[-1].pop()

                    current_error = None
                    event_queue.put({"type": "delete_error"})
        except Empty:
            pass

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
    app.config["flask_profiler"] = {
        "enabled": True,
        "basicAuth":{
            "enabled": False
        },
        "storage": {
            "engine": "sqlite",
            "FILE": "flask_profiler.sql"
        },
        "ignore": [
            "^/static/.*"
        ]
    }
    socketio = SocketIO(app, async_mode="threading", cors_allowed_origins='*')
    flask_profiler.init_app(app)

    @app.route('/send_command', methods=['POST'])
    def send_command():
        command = request.form.get('command')
        if command:
            arduino_queue.put(command)
            return 'Command sent!'
        else:
            return 'No command received', 400

    @app.route('/record_result', methods=['POST'])
    def record_result():
        new_record = request.get_json()

        # Load existing records
        with open(config.HANDS_DATABASE_PATH, 'r') as f:
            records = json.load(f)

        # Append the new record
        records.append(new_record)

        # Save the records back to the file
        with open(config.HANDS_DATABASE_PATH, 'w') as f:
            json.dump(records, f)

        return jsonify({'message': 'Record added successfully'}), 200

    @app.route('/delete_record', methods=['POST'])
    def delete_record():
        # Get the ID of the record to delete
        record_id = request.get_json().get('id')
        print(record_id)
        # Load the existing records
        with open(config.HANDS_DATABASE_PATH, 'r') as f:
            records = json.load(f)

        # Find the record with the given ID
        record_to_delete = next((record for record in records if record['hand_number'] == str(record_id)), None)

        if record_to_delete is None:
            # If no record with the given ID was found, return an error
            return jsonify({'message': 'Record not found'}), 404

        # Remove the record from the list
        records.remove(record_to_delete)

        # Save the records back to the file
        with open(config.HANDS_DATABASE_PATH, 'w') as f:
            json.dump(records, f)

        return jsonify({'message': 'Record deleted successfully'}), 200

    @app.route('/get_results', methods=['GET'])
    def get_results():
        # Load the records
        with open(config.HANDS_DATABASE_PATH, 'r') as f:
            records = json.load(f)

        # Sort the records by hand number
        records.sort(key=lambda record: record['hand_number'])

        return jsonify(records), 200


    class MyNamespace(Namespace):
        frontend_queue = Queue()
        def on_connect(self):
            self.sid = request.sid

        def on_undo(self):
            print("HELLO")
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
                                self.emit('event', result)
                                if result["passout"]:
                                    print("PASSOUT")
                                    camera_input_queue.put({'type': 'set_mode', 'value': 'none'})
                                    break
                                else:
                                    camera_input_queue.put({'type': 'set_mode', 'value': 'cardplay'})
                                    cardplay_process = Process(target=cardplay_mode, args=(camera_output_queue, frame_queue, event_queue, self.frontend_queue, northDirection, result, result_queue))
                                    cardplay_process.start()
                            else:
                                cardplay_result = result
                                cardplay_process.join()
                                cardplay_process = None
                                self.emit('event', result)
                                break
                        except Empty:
                            #print(f"still doing stuff {random.randint(1, 10000)}")
                            pass
                        except Exception as e:
                            print(e)
                            raise
                    
                    output_queue.put(auction_result)
                    output_queue.put(cardplay_result)
                    #self.disconnect(self.sid)  # Disconnect the client

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