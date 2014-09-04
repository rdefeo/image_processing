__author__ = 'robdefeo'
import cv2
from improc.color import Matrix_scikit_kmeans
from improc.crop import AutoCrop
from improc.shape import ScaleMax, ScaleHeight, ScaleWidth
import numpy as np

def scale_max(img, width=250, height=250):
    return ScaleMax(img, width, height)

def grey(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def autocrop(img):
    (
        matrix,
        cluster_centers_,
        labels,
        background_label
    ) = Matrix_scikit_kmeans(img, 5)

    (
        img_autocrop,
        x,
        y,
        width,
        height
    ) = AutoCrop(img, labels, background_label)

    return img_autocrop

def resize(img, size):
    return cv2.resize(img, size)


def add_border(img, color_value=255, width=250, height=250, border_size=15):
    bottom = border_size
    left = border_size
    right = (width - img.shape[1]) + border_size
    top = (height - img.shape[0]) + border_size

    return cv2.copyMakeBorder(
        img,
        top, bottom, left, right,
        cv2.BORDER_CONSTANT,
        value=color_value
    )

# def add_border(img, border_size=15, color_value=255):
#     return cv2.copyMakeBorder(
#         img,
#         border_size, border_size, border_size, border_size,
#         cv2.BORDER_CONSTANT,
#         value=color_value
#     )

def outline_contour(img):
    outline = np.zeros(img.shape, dtype="uint8")
    (cnts, _) = cv2.findContours(
        img.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[0]
    cv2.drawContours(outline, [cnts], -1, 255, -1)

    return outline

def bitwise(img):
    return cv2.bitwise_not(img)

def thresh(img):
    img[img > 0] = 255

    return img