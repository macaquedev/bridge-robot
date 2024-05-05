import cv2

# List of common resolutions to check
resolutions = [(640, 480), (1600, 1200), (800, 600), (1024, 768), (1280, 720), (1920, 1080), (352, 288), (320, 240), (160, 120), (3840, 2160), (5120, 2880), (4096, 2160)]
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

for width, height in resolutions:
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    
    if (actual_width, actual_height) == (width, height):
        print(f"Resolution {width}x{height} is supported.")
    else:
        pass
        #print(f"Resolution {width}x{height} is not supported. Current: {actual_width}x{actual_height}.")

cap.release()
exit(-1)