import cv2
from imutils.video import WebcamVideoStream
import config
import numpy as np
from ultralytics import YOLO


def robust_mean(data: np.ndarray):
    """Calculate mean values from a set of data points, using the median absolute deviation to discard outliers."""
    
    medians = np.median(data, axis=0)
    # calculates the median absolute deviation
    mad = np.median(np.absolute(data - medians), axis=0)

    cleaned_data = []
    for values in data:
        # if any values in a set of values deviate from the median by more than MAD_THRESHOLD times the median absolute
        # deviation, the entire set is discarded
        if np.all(np.absolute(values - medians) <= config.MAD_THRESHOLD * mad):
            cleaned_data.append(values)

    return np.mean(np.array(cleaned_data), axis=0)


class DShowCamera(WebcamVideoStream):  # i hate this!
    def __init__(self, src, height, width, name="WebcamVideoStream"):
        self.stream = cv2.VideoCapture(src, cv2.CAP_DSHOW)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.grabbed, self.frame = self.stream.read()
        self.name = name  
        self.stopped = False
        

class Camera:
    def __init__(self, camera_index, height, width, rotate=False):
        self.cap = DShowCamera(camera_index, height, width).start()
        self.width = self.cap.stream.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.cap.stream.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.rotate = rotate
        self.cardnet = YOLO(config.CARDNET_PATH)
        self.bidding_box_net = YOLO(config.BIDDING_BOX_NET_PATH)
        self.bidding = True

    def detect(self, frame=None, bidding=None, width=None, height=None):
        if not width:
            width = self.width
        if not height:
            height = self.height
        if frame is None:
            frame = self.raw_read()

        if bidding is None:
            bidding = self.bidding
        
        if bidding:
            result = next(self.bidding_box_net([frame], stream=True, conf=config.BIDDING_BOX_NET_CONFIDENCE, verbose=False))  
        else:
            result = next(self.cardnet([frame], stream=True, conf=config.CARDNET_CONFIDENCE, verbose=False)) 

        data = []
        
        for box in result.boxes:
            x, y, x2, y2 = [int(i.numpy()) for i in box.xyxy[0]]
            if not (20 <= (x2+x)//2 <= width-20 and 20 <= (y2+y)//2 <= height-20):
                continue
            if bidding and abs(x2-x) * abs(y2-y) < config.MIN_BID_AREA:
                continue
            data.append((result.names[int(box.cls.item())], (x, y), (x2, y2)))

        return data
    

    def draw_boxes(self, bidding=None):
        if bidding is None:
            bidding = self.bidding
        frame = self.raw_read()
        #frame = frame[0:1080, 420:1500]
        height, width = frame.shape[:2]
        scale = min(1080.0 / height, 1080.0 / width)
        frame = cv2.resize(frame, None, fx=scale, fy=scale)

        # Pad the image to make it square
        top = 0  # No padding at the top
        bottom = 1080 - frame.shape[0]  # All padding at the bottom
        left = right = (1080 - frame.shape[1]) // 2
        frame = cv2.copyMakeBorder(frame, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[255, 255, 255])
        for label, corner1, corner2 in self.detect(frame, bidding=bidding, width=1080, height=1080):
            cv2.rectangle(frame, corner1, corner2, (255, 0, 0), 2)
            cv2.putText(frame, label, (corner1[0], corner1[1]-20),
                         cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))
        return frame   
    

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.cap.stop()
        self.cap.stream.release()

    def raw_read(self):
        frame = self.cap.read()
        return cv2.rotate(frame, cv2.ROTATE_180) if self.rotate else frame #cv2.resize(frame, (400, 400))
    