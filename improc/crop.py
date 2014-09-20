import logging
from time import time

import numpy as np
from color import IsWhiteish

LOGGER = logging.getLogger(__name__)


def ms(start):
    return (time() - start) * 1000


def Crop(img, x, y, width, height):
    # original_image[ystart:ystop, xstart:xstop]
    return img[y:y + height, x:x + width]
    # Crop from x, y, w, h -> 100, 200, 300, 400
    # NOTE: its img[y: y + h, x: x + w] and *not* img[x: x + w, y: y + h]


def AutoCrop(original_image, labels, background_label):
    # uses the outputs from Matrix_scikit_kmeans to get work with
    top_left_pixel = original_image[0][0]
    if not IsWhiteish(top_left_pixel):
        LOGGER.warn("top left pixel is not whiteish,function=autocrop")
        return None, None, None, None, None

    reshaped_labels = labels.reshape(
        original_image.shape[0], original_image.shape[1])
    start = time()

    bounding_box = np.argwhere(reshaped_labels != background_label)
    if len(bounding_box) == 0:
        LOGGER.warn("no bounding box found after reshaping,function=autocrop")
        return None, None, None, None, None
        
    (ystart, xstart) = bounding_box.min(0) - 1
    (ystop, xstop) = bounding_box.max(0) + 1
    ystart = max(ystart, 0)
    xstart = max(xstart, 0)

    bounding_box_image = original_image[ystart:ystop, xstart:xstop]
    if len(bounding_box_image) == 0 or len(bounding_box_image[0]) == 0:
        LOGGER.warn("bounding box image empty after crop,function=autocrop")
        return None, None, None, None, None

    top_left_pixel_bounding_box = bounding_box_image[0][0]
    bottom_right_pixel_bounding_box = bounding_box_image[::-1][0][::-1][0]

    if (
        not IsWhiteish(top_left_pixel_bounding_box) or
        not IsWhiteish(bottom_right_pixel_bounding_box)
    ):
        LOGGER.warn("top left or bottom right pixel is not whiteish after crop,function=autocrop")
        return None, None, None, None, None

    LOGGER.info('function=autocrop,ms=%s' % (ms(start)))

    return bounding_box_image, xstart, ystart, xstop - xstart, ystop - ystart
    # return cropped_image, x, y, width, height
