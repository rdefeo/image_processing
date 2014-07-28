import logging
from itertools import groupby
from time import time

import cv2
import numpy as np

LOGGER = logging.getLogger(__name__)


def ms(start):
    return (time() - start) * 1000


def IsWhite(pixel):
    return pixel[0] == 255 and pixel[1] == 255 and pixel[2] == 255


def Background(img):
    if img[0][0][0] == img[len(img)-1][0][0] == img[len(img)-1][len(img[0])-1][0] == img[0][len(img[0])-1][0] \
        and img[0][0][1] == img[len(img)-1][0][1] == img[len(img)-1][len(img[0])-1][1] == img[0][len(img[0])-1][1] \
            and img[0][0][2] == img[len(img)-1][0][2] == img[len(img)-1][len(img[0])-1][2] == img[0][len(img[0])-1][2]:
        return img[0][0]
    else:
        return None


def Reduce(img, number_of_colors, method="cv2"):
    if method == "cv2":
        return Reduce_cv2(img, number_of_colors)
    elif method == "sklearn":
        return Reduce_sklearn(img, number_of_colors)
    else:
        raise Exception("jkhgjfh")


def Reduce_sklearn(img, number_of_colors):
    from sklearn.cluster import KMeans
    img_64 = np.array(img, dtype=np.float64) / 255
    w, h, d = original_shape = tuple(china.shape)
    assert d == 3
    image_array = np.reshape(img_64, (w * h, d))

    t0 = time()
    image_array_sample = shuffle(image_array, random_state=1)[:1000]
    kmeans = KMeans(
        n_clusters=n_colors,
        random_state=1,
        precompute_distances=True).fit(image_array_sample)

    LOGGER.info(
        "Fitting model on a small sub-sample of the data,ms=%0.3fs.",
        (time() - t0))

    t0 = time()
    labels = kmeans.predict(image_array)
    LOGGER.info(
        "Predicting color indices on the full image (k-means),ms=%0.3fs.",
        (time() - t0))

    print labels


def Reduce_cv2(img, number_of_colors):
    start = time()
    """
    Reduce image to, number of colors supplied that are most relevant
    """

    Z = img.reshape((-1, 3))
    # convert to np.float32
    Z = np.float32(Z)

    K = number_of_colors
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

    ret, label, center = cv2.kmeans(
        Z,
        K,
        criteria,
        10,
        cv2.KMEANS_RANDOM_CENTERS)

    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((img.shape))

    LOGGER.info('ms=%s' % (ms(start)))
    return res2


def Hex(r, g, b):
    return '0x%02x%02x%02x' % (r, g, b)


def Matrix(img):
    """
    Ignores the background image color, and returns a simplied break down of
    all the color information in the image, array of percentages
    """
    start = time()

    background_image_color = Background(img)
    if background_image_color is None:
        return None

    background_color_hex = Hex(
        background_image_color[2],
        background_image_color[1],
        background_image_color[0]
    )

    from shape import Flatten
    flattened_image = Flatten(img)

    color_ids = np.array(
        filter(
            lambda x: x != background_color_hex,
            map(
                lambda x: Hex(x[2], x[1], x[0]),
                flattened_image
            )
        )
    )
    color_ids.sort()

    matrix = []
    for key, group in groupby(color_ids, lambda x: x):
        percent = float(float(len(list(group))) / len(color_ids))
        matrix.append({
            "hex": key,
            "percent": percent
        })

    LOGGER.info('ms=%s' % (ms(start)))
    return matrix
