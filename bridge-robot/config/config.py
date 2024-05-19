from pathlib import Path 
import shapely


MAINCAM_INDEX = 1
MAINCAM_WIDTH = 3840
MAINCAM_HEIGHT = 2160
DETECT_FRAME_SIZE = 1080
DETECT_CARD_WIDTH = DETECT_FRAME_SIZE // 6
DETECT_CARD_HEIGHT = DETECT_FRAME_SIZE // 4
BLUR_COEFFICIENT = 15
ARMCAM_INDEX = 2
ARMCAM_WIDTH = 640
ARMCAM_HEIGHT = 480
ARMCAM_OFFSET = 265
ARMCAM_MISDETECTION_THRESHOLD = 7
CARDNET_PATH = Path(__file__).parent.parent / "model" / "smallcardnet-40ep.pt"
CARDNET_CONFIDENCE = 0.28
BIDDING_BOX_NET_PATH = Path(__file__).parent.parent / "model" / "bidding-box-net-50ep.pt"
BIDDING_BOX_NET_CONFIDENCE = 0.28
ARDUINO_BAUDRATE = 115200

CARD_AREA = 2000
BKG_THRESH = 100
COALESCE_DISTANCE_SQUARED = 400
UPSIDE_DOWN_THRESH = 210

MAD_THRESHOLD = 2
ADD_THRESHOLD = 2
REMOVE_THRESHOLD = 3
MIN_BID_AREA = 2000

TOP_POLYGON = shapely.geometry.Polygon([(300, 0), (1080, 0), (1080, 200), (300, 200)])
LEFT_POLYGON = shapely.geometry.Polygon([(0, 0), (299, 0), (299, 400), (0, 400)])
RIGHT_POLYGON = shapely.geometry.Polygon([(1080, 607), (1080, 200), (780, 200), (780, 607)])
BOTTOM_POLYGON = shapely.geometry.Polygon([(0, 400), (780, 400), (780, 607), (0, 607)])



SORTFUNCS = [
    lambda x: -x[1][0],
    lambda x: -x[1][1],
    lambda x: x[1][0],
    lambda x: x[1][1]
]