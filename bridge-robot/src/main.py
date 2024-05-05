from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit, Namespace
import cv2

from src.vision.maincam import MainCam
from src.communication.arduino import Arduino
import config
import base64
from multiprocessing import Process, Queue, Manager


def arduino_process(queue):
    print("Connecting to Arduino....")
    arduino = Arduino(config.ARDUINO_BAUDRATE)
    print("Arduino connected!")
    while True:
        command = queue.get()
        if command and arduino.currently_acknowledged:
            arduino.send_data(command)


def gen_frames(queue):  
    with MainCam(config.MAINCAM_INDEX, config.MAINCAM_WIDTH, config.MAINCAM_HEIGHT) as camera:
        while True:
            frame = cv2.resize(camera.draw_boxes(), (640, 480))
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                continue
            frame = base64.b64encode(buffer).decode('utf-8')
            queue.put(frame)


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
        def on_connect(self):
            def send_frames():
                queue = Queue()
                p = Process(target=gen_frames, args=(queue,))
                p.start()
                while True:
                    frame = queue.get()
                    self.emit('image', frame)
            socketio.start_background_task(send_frames)

        def new_bid(self, bid):
            self.emit('new_bid', bid)
            

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