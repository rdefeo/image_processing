import cv2
import numpy as np
import logging
LOGGER = logging.getLogger(__name__)

def Flatten(img):
    """
    Keeps the color dimension by merges the X, Y dimensions
    """
    return img.reshape((-1, 3)).take((0, 1, 2), 1)


def Ratio(img):
    return float(img.shape[1]) / img.shape[0]


def ScaleMax(img, width, height):
    r = float(height) / img.shape[0]
    proposedWidth = int(img.shape[1] * r)
    if proposedWidth > width:
        return ScaleWidth(img, width)
    else:
        return ScaleHeight(img, height)


def ScaleHeight(img, height):
    r = float(height) / img.shape[0]
    dim = (int(img.shape[1] * r), height)

    # perform the actual resizing of the image and show it
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    LOGGER.info(
        "action=ScaleHeight,width=%s,height=%s",
        resized.shape[1], resized.shape[0]
    )
    return resized


def ScaleWidth(img, width):
    r = float(width) / img.shape[1]
    dim = (width, int(img.shape[0] * r))

    # perform the actual resizing of the image and show it
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    LOGGER.info(
        "action=ScaleWidth,width=%s,height=%s",
        resized.shape[1], resized.shape[0]
    )
    return resized


def MakeSquare(img):
    '''Make an image square by adding white pixels to the smaller dimension,
    It keeps the original image centered'''
    img_size = max(img.shape[1], img.shape[0])
    whitebar_size = (img_size - min(img.shape[1], img.shape[0]))/2

    if img.shape[0] > img.shape[1]:
        whitebar = np.ones((img_size, whitebar_size, 3), np.uint8)*255
        img = np.concatenate((whitebar, img, whitebar), axis=1)
        if img.shape[0] - img.shape[1] == 1:
            img = np.concatenate((img, np.ones((img_size, 1, 3), np.uint8)*255), axis=1)

    if img.shape[0] < img.shape[1]:
        whitebar = np.ones((whitebar_size, img_size, 3), np.uint8)*255
        img = np.concatenate((whitebar, img, whitebar), axis=0)
        if img.shape[1] - img.shape[0] == 1:
            img = np.concatenate((img, np.ones((1, img_size, 3), np.uint8)*255), axis=0)

    return img
