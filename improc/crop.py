import logging
import time

import cv2
import numpy as np
from color import IsWhite

LOGGER = logging.getLogger(__name__)


def ms(start):
    return (time.time() - start) * 1000


def Crop(img, x, y, width, height):
    return img[y:y + height, x:x + width]
    # Crop from x, y, w, h -> 100, 200, 300, 400
    # NOTE: its img[y: y + h, x: x + w] and *not* img[x: x + w, y: y + h]


def AutoCrop_calculate_row(row, new_img, found, counter):
    min_num_non_white_pixels = 10
    non_white_pixels = [x for x in row if x[0] < 253 and x[1] < 253 and x[2] < 253]
    if len(non_white_pixels) > min_num_non_white_pixels:
        found = True
        new_img.append(row)
    elif not found:
        counter += 1

    return new_img, found, counter


def AutoCrop_calculate_rowv2(row, new_img, found, counter):
    min_num_non_white_pixels = 10
    non_white_pixels = 0
    for x in row:
        if non_white_pixels > min_num_non_white_pixels:
            found = True
            new_img.append(row)
            break

        if x[0] < 253 and x[1] < 253 and x[2] < 253:
            non_white_pixels += 1

    # if len(non_white_pixels) > 10:
        # xfound = True
        # crop_img_rotated.append(row)

    if not found and non_white_pixels <= min_num_non_white_pixels:
        counter += 1

    return new_img, found, counter


def AutoCrop(img):
    """
    returns x, y, width, height
    """
    start = time.time()
    img = img.swapaxes(1, 0)
    crop_img_rotated = []
    xfound = False
    # want to have a sligher before
    counterx = -1
    min_num_non_white_pixels = 10
    return_value = None, None, None, None, None
    if IsWhite(img[0][0]) and IsWhite(img[len(img)-1][0]) and IsWhite(img[len(img)-1][len(img[0])-1]) and IsWhite(img[0][len(img[0])-1]):
        # any (e for e in [1, 2, 'joe'] if isinstance(e, int) and e > 0)
        for row in img:
            crop_img_rotated, xfound, counterx = AutoCrop_calculate_rowv2(
                row, crop_img_rotated, xfound, counterx)

        yfound = False
        # want to have a sligther before
        countery = -1

        img = np.array(crop_img_rotated).swapaxes(1, 0)
        crop_img = []
        for row in img:
            crop_img, yfound, countery = AutoCrop_calculate_rowv2(
                row, crop_img, yfound, countery)

        cropped = np.array(crop_img)

        if countery == -1 or counterx == -1:
            return_value = None, None, None, None, None
        elif not IsWhite(cropped[0][0]) and not IsWhite(cropped[len(cropped)-1][0]) and not IsWhite(cropped[len(cropped)-1][len(cropped[0])-1]) and not IsWhite(cropped[0][len(cropped[0])-1]):
            return_value = None, None, None, None, None
        else:
            return_value = cropped, counterx, countery, cropped.shape[1] + 2, cropped.shape[0] + 2

    else:
        return_value = None, None, None, None, None

    LOGGER.info('ms=%s' % (ms(start)))
    return return_value
