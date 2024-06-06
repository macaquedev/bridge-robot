from pathlib import Path 
from shapely.geometry import Polygon


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
ARMCAM_CARDNET_PATH = Path(__file__).parent.parent / "model" / "cardnet-70ep.pt"
ARMCAM_CARDNET_CONFIDENCE = 0.2
CARDNET_PATH = Path(__file__).parent.parent / "model" / "cardnet-70ep.pt"
CARDNET_CONFIDENCE = 0.35
BIDDING_BOX_NET_PATH = Path(__file__).parent.parent / "model" / "bidding-box-net-50ep.pt"
BIDDING_BOX_NET_CONFIDENCE = 0.28
ARDUINO_BAUDRATE = 115200
HANDS_DATABASE_PATH = Path(__file__).parent / "hands_database.json"

MIN_CARD_AREA = 25000
MAX_CARD_AREA = 60000
BKG_THRESH = 100
CARDPLAY_COALESCE_DISTANCE_SQUARED = 400
AUCTION_COALESCE_DISTANCE_SQUARED = 60
UPSIDE_DOWN_THRESH = 210

MAD_THRESHOLD = 2
AUCTION_ADD_THRESHOLD = 2
AUCTION_REMOVE_THRESHOLD = 3
CARDPLAY_ADD_THRESHOLD = 3
CARDPLAY_REMOVE_THRESHOLD = 1
MIN_BID_AREA = 3000

AUCTION_TOP_POLYGON = Polygon([(300, 0), (1080, 0), (1080, 200), (300, 200)])
AUCTION_LEFT_POLYGON = Polygon([(0, 0), (299, 0), (299, 400), (0, 400)])
AUCTION_RIGHT_POLYGON = Polygon([(1080, 607), (1080, 200), (780, 200), (780, 607)])
AUCTION_BOTTOM_POLYGON = Polygon([(0, 400), (780, 400), (780, 607), (0, 607)])


CARDPLAY_TOP_POLYGONS = [
    Polygon([(0, 0), (0, 720), (2560, 720), (2560, 0)]),
    Polygon([(0, 0), (0, 474), (1280, 474), (1280, 0)]),
    Polygon([(853, 0), (853, 720), (1706, 720), (1706, 0)]),
    Polygon([(1280, 0), (1280, 474), (2560, 474), (2560, 0)])
]
   
CARDPLAY_LEFT_POLYGONS = [
    Polygon([(0, 720), (0, 1440), (853, 1440), (853, 720)]),
    Polygon([(0, 966), (0, 475), (1280, 475), (1280, 966)]),
    Polygon([(0, 0), (0, 720), (853, 720), (853, 0)]),
    Polygon([(0, 0), (0, 1440), (1280, 1440), (1280, 0)])
]

CARDPLAY_RIGHT_POLYGONS = [
    Polygon([(1706, 720), (1706, 1440), (2560, 1440), (2560, 720)]),
    Polygon([(1280, 0), (1280, 1440), (2560, 1440), (2560, 0)]),
    Polygon([(1706, 0), (1706, 720), (2560, 720), (2560, 0)]),
    Polygon([(1280, 966), (1280, 475), (2560, 475), (2560, 966)])
]

CARDPLAY_BOTTOM_POLYGONS = [
    Polygon([(853, 720), (853, 1440), (1706, 1440), (1706, 720)]),
    Polygon([(0, 1440), (0, 966), (1280, 966), (1280, 1440)]),
    Polygon([(0, 720), (0, 1440), (2560, 1440), (2560, 720)]),
    Polygon([(1280, 1440), (1280, 966), (2560, 966), (2560, 1440)])
]