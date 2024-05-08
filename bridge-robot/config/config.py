from pathlib import Path 
import shapely


MAINCAM_INDEX = 1
MAINCAM_WIDTH = 3840
MAINCAM_HEIGHT = 2160
BLUR_COEFFICIENT = 15
ARMCAM_INDEX = 2
ARMCAM_WIDTH = 640
ARMCAM_HEIGHT = 480
ARMCAM_OFFSET = 265
ARMCAM_MISDETECTION_THRESHOLD = 7
CARDNET_PATH = Path(__file__).parent.parent / "model" / "cardnet-20ep.pt"
CARDNET_CONFIDENCE = 0.38
BIDDING_BOX_NET_PATH = Path(__file__).parent.parent / "model" / "bidding-box-net-50ep.pt"
BIDDING_BOX_NET_CONFIDENCE = 0.3
ARDUINO_BAUDRATE = 115200

CARD_AREA = 3000
BKG_THRESH = 110
COALESCE_DISTANCE_SQUARED = 1200
UPSIDE_DOWN_THRESH = 150

MAD_THRESHOLD = 2
ADD_THRESHOLD = 3
REMOVE_THRESHOLD = 5
MIN_BID_AREA = 1300

TOP_POLYGON = shapely.geometry.Polygon([(300, 0), (1280, 0), (1280, 300), (300, 300)])
LEFT_POLYGON = shapely.geometry.Polygon([(0, 0), (299, 0), (299, 420), (0, 420)])
RIGHT_POLYGON = shapely.geometry.Polygon([(1280, 720), (1280, 301), (980, 301), (980, 720)])
BOTTOM_POLYGON = shapely.geometry.Polygon([(0, 421), (980, 421), (980, 720), (0, 720)])



SORTFUNCS = [
    lambda x: -x[1][0],
    lambda x: -x[1][1],
    lambda x: x[1][0],
    lambda x: x[1][1]
]