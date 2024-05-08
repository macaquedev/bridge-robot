from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit, Namespace
import cv2
from shapely.geometry import Point
from src.vision.maincam import MainCam
from src.communication.arduino import Arduino
import config
import base64
from multiprocessing import Process, Queue, Manager
from queue import Empty
from collections import defaultdict

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
        '1C', '1D', '1H', '1S', '1NT',
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


def auction_mode(frame_queue, event_queue, frontend_queue, dealer, northDirection, vulnerability):  
    with MainCam(config.MAINCAM_INDEX, config.MAINCAM_WIDTH, config.MAINCAM_HEIGHT) as cap:

        frame = cv2.resize(cap.raw_read(), (1280, 720))
        in_frame = defaultdict(list)
        to_be_added = defaultdict(list)
        to_be_removed = defaultdict(list)
        directions = ["North", "East", "South", "West"]
        directions = directions[["top", "left", "bottom", "right"].index(northDirection):] + directions[:["top", "left", "bottom", "right"].index(northDirection)]
        print(directions)
        dealer = directions.index(dealer)
        print(dealer)
        auction = [[], [], [], []]
        current_turn = dealer
        current_error = None
        while True:
            frame = cap.raw_read()

            #frame = frame[0:1080, 420:1500]
            height, width = frame.shape[:2]
            scale = min(1280.0 / height, 1280.0 / width)
            frame = cv2.resize(frame, None, fx=scale, fy=scale)

            # Pad the image to make it square
            top = 0  # No padding at the top
            bottom = 1280 - frame.shape[0]  # All padding at the bottom
            left = right = (1280 - frame.shape[1]) // 2
            frame = cv2.copyMakeBorder(frame, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[255, 255, 255])
        
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
                    if to_be_added[card][i][0] == config.ADD_THRESHOLD:
                        if card not in in_frame:
                            in_frame[card] = []
                        in_frame[card].append([to_be_added[card][i][1], to_be_added[card][i][2], None])
                        box = in_frame[card][-1]
                        x_pos = box[0][0] + (box[1][0] - box[0][0]) // 2
                        y_pos = box[0][1] + (box[1][1] - box[0][1]) // 2
                        p = Point(x_pos, y_pos)
                        if config.TOP_POLYGON.contains(p):
                            player = 0
                        elif config.RIGHT_POLYGON.contains(p):
                            player = 1
                        elif config.BOTTOM_POLYGON.contains(p):
                            player = 2
                        elif config.LEFT_POLYGON.contains(p):
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
                    if to_be_removed[card][i][0] >= config.REMOVE_THRESHOLD:
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
                played = False
                for i in range(4):
                    if not by_player[i]:
                        continue
                    
                    card, top_left, bottom_right = by_player[i][-1]
                    if not auction[i]:
                        if i != current_turn:
                            current_error = (card, top_left, bottom_right)
                            event_queue.put({'type': 'error', 'card': card, 'player': directions[i]})
                        else:
                            auction[i].append((card, top_left, bottom_right))
                            event_queue.put({'type': 'new_bid', 'card': card, 'player': directions[i]})
                            played = True

                    else:
                        prev_card, prev_top_left, prev_bottom_right = auction[i][-1]
                         
                        if card not in ["Pass", "Double", "Redouble"]:
                            if prev_card == card:
                                continue  
                            flag = True
                            for c, tl, br in auction[i]:
                                if card == c or abs(tl[0]-top_left[0])**2 + abs(tl[1]-top_left[1])**2 < config.COALESCE_DISTANCE_SQUARED:
                                    flag = False
                                    break
                            if not flag:
                                continue
#TODO: when undo happens, it grabs detection from the wrong place - very peculiar bug....
                        #TODO WORK HERE because pass cards just keep working and working and working basically.
                        if card in ["Pass", "Double", "Redouble"] or (top_left[0] - prev_top_left[0]) ** 2 + (top_left[1] - prev_top_left[1]) ** 2 > config.COALESCE_DISTANCE_SQUARED:
                            if i != current_turn:
                                comp_list = auction[i][:]
                                comp_list.append((card, top_left, bottom_right))
                                comp_list.sort(key=config.SORTFUNCS[i])
                                if card != comp_list[-1][0] or (top_left[0] - comp_list[-1][1][0]) ** 2 + (top_left[1] - comp_list[-1][1][1]) ** 2 > config.COALESCE_DISTANCE_SQUARED:
                                    current_error = (card, top_left, bottom_right)
                                    event_queue.put({'type': 'error', 'card': card, 'player': directions[i]})
                            else:
                                if (top_left[0] - prev_top_left[0]) ** 2 + (top_left[1] - prev_top_left[1]) ** 2 > config.COALESCE_DISTANCE_SQUARED:
                                    auction[i].append((card, top_left, bottom_right))
                                    event_queue.put({'type': 'new_bid', 'card': card, 'player': directions[i]})
                                    played = True
                                else:
                                    auction[i][-1] = (auction[i][-1][0], top_left, bottom_right)

                                    
                        else:   # close together
                            auction[i][-1] = (auction[i][-1][0], top_left, bottom_right)

            if played:
                current_turn = (current_turn + 1) % 4
                print(directions[current_turn])
            
            ret, buffer = cv2.imencode('.jpg', cv2.resize(remove_white_borders(frame), (320, 180)))
            if not ret:
                continue
            frame_queue.put(base64.b64encode(buffer).decode('utf-8'))
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
                    
                    error = None
                    event_queue.put({"type": "delete_error"})

                        #if error and error[0] 


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
            print(directions[current_turn], current_error)
            


def create_app():
    manager = Manager()
    arduino_queue = manager.Queue()
    p_arduino = Process(target=arduino_process, args=(arduino_queue,))
    p_arduino.start()
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

            def send_frames():
                frame_queue = Queue()
                event_queue = Queue()
                p = Process(target=auction_mode, args=(frame_queue,event_queue, self.frontend_queue, dealer, northDirection, vulnerability))
                p.start()
                while True:
                    frame = frame_queue.get()
                    self.emit('image', frame)
                    try:
                        event = event_queue.get(timeout=0)
                        self.emit('event', event)
                    except Empty:
                        pass
            socketio.start_background_task(send_frames)

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