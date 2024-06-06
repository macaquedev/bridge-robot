import cv2
import numpy as np
import config


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


def near(point1, point2, cardplay=False):
    return (point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2 < (config.CARDPLAY_COALESCE_DISTANCE_SQUARED if cardplay else config.AUCTION_COALESCE_DISTANCE_SQUARED) 


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


def top_left(polygon):
    tl = min(polygon.exterior.coords, key=lambda x: x[0] + x[1])
    return (int(tl[0] + 30), int(tl[1] + 200))


def card_value(card):
    return "23456789TJQKA".index(card)