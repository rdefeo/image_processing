__author__ = 'robdefeo'
import cv2
from improc.color import Matrix_scikit_kmeans
from improc.crop import AutoCrop
from improc.shape import ScaleMax, ScaleHeight, ScaleWidth
from skimage import filters
from skimage import measure
from skimage import feature
from skimage import morphology
from skimage.color import rgb2gray
import numpy as np
import improc.smooth
from scipy.misc import imshow


def blur(img, gaussian_blur=None, median_blur=None):
    x = np.copy(img)
    if gaussian_blur is not None and gaussian_blur["enabled"]:
        x = improc.smooth.gaussian_blur(
            x,
            (
                gaussian_blur["ksize_width"],
                gaussian_blur["ksize_height"]
            ),
            gaussian_blur["sigmaX"]
        )

    if median_blur is not None and median_blur["enabled"]:
        x = improc.smooth.median_blur(
            x,
            median_blur["ksize"]
        )

    return x


def scale_max(img, width=250, height=250):
    return ScaleMax(img, width, height)

def make_square(img):
    return MakeSquare(img)


def grey(img):
    return rgb2gray(img)


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


def add_border(img, color_value=255, width=250, height=250, border_size=15, fill_dimensions=True):
    bottom = border_size
    left = border_size
    right = (width - img.shape[1]) + border_size if fill_dimensions else border_size
    top = (height - img.shape[0]) + border_size if fill_dimensions else border_size

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
    contours = measure.find_contours(img.copy(), 0.8)

    if len(outline) > 0:
        for n, contour in enumerate(contours):
            print outline.shape
            print contours[n].shape
            outline = outline + contours[n]

        return outline

    else:
        return None


def canny(img, threshold1=100, threshold2=200):
    # return cv2.Canny(img, threshold1, threshold2)
    return feature.canny(img, sigma=1, low_threshold=threshold1, high_threshold=threshold2) # sigma = width of the noise


def bitwise(img):
    return np.invert(img)


def laplacian(img):
    # edge detection
    return cv2.Laplacian(img,cv2.CV_64F)


def thresh(img):
    img[img > 0] = 255
    return img


def dilate(img, width=5, height=5):
    # kernel = np.ones((width, height), np.uint8)
    return morphology.binary_dilation(img)


def closing(img, width=5, height=5):
    # kernel = np.ones((width, height), np.uint8)
    return morphology.binary_closing(img)

