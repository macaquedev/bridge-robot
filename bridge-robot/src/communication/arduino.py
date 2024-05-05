import serial
import serial.tools.list_ports
import warnings
import threading


class Arduino:
    def __init__(self, port="/dev/ttyUSB0", baud_rate=115200):
        self.lock = threading.Lock()
        arduino_ports = [
            p.device
            for p in serial.tools.list_ports.comports()
            if 'wch.cn' in p.manufacturer  # may need tweaking to match new arduinos
        ]
        if not arduino_ports:
            raise IOError("No Arduino found")
        if len(arduino_ports) > 1:
            warnings.warn('Multiple Arduinos found - using the first')

        self.arduino = serial.Serial(arduino_ports[0], baud_rate)
        self.currently_acknowledged = False
        self.acknowledgement_thread = threading.Thread(target=self.wait_for_acknowledgement)
        self.acknowledgement_thread.start()
        while not self.currently_acknowledged:
            pass
            
        self.send_acknowledgement()
        self.currently_acknowledged = True


    def data_waiting(self):
        return self.arduino.inWaiting()

    def get_data(self):
        return self.arduino.readline().decode("utf-8").strip()

    def send_data(self, d):
        with self.lock:
            self.currently_acknowledged = False
            self.arduino.write((d + "\r").encode())

    def send_data_and_wait_for_acknowledgement(self, d):
        """
        Warning: BLOCKING!!!
        """
        self.send_data(d)
        while not self.currently_acknowledged:
            pass

    def wait_for_acknowledgement(self):
        while True:
            with self.lock:
                if self.data_waiting():
                    data = self.get_data()
                    if data == 'OK':
                        self.currently_acknowledged = True
                    elif data.startswith('ERROR'):
                        raise IOError(data[6:])

    def received_acknowledgement(self):
        if self.data_waiting():
            a = self.get_data()
            if a == "OK":
                return True
            elif a.startswith("ERROR"):
                raise IOError(a[6:])
        return False

    def send_acknowledgement(self):
        self.send_data("OK")
        