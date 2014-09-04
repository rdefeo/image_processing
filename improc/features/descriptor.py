import logging
import time

import cv2
import mahotas.features
import numpy as np
from preprocess import grey, autocrop, resize, add_border, outline_contour, thresh_bitwise

LOGGER = logging.getLogger(__name__)


def ms(start):
    return (time.time() - start) * 1000


class FeatureDescriptor(object):
    name = None
    properties = {}

    def create_spec(self, img):
        return {
            "shape": list(img.shape),
            "name": self.name,
            "properties": self.properties
        }

    def create_result(self, img, value):
        return {
            "spec": self.create_spec(img),
            "value": value
        }


class HarlickDescriptor(FeatureDescriptor):
    def __init__(self, preprocess=False, mean=0, size=(250, 250)):
        self.name = "harlick"
        self.properties["mean"] = mean
        self.properties["size"] = size
        self.preprocess = preprocess

    def do_preprocess(self, img):
        x = np.copy(img)
        return cv2.resize(x, self.properties["size"])

    def describe(self, img):
        start = time.time()
        x = img
        if self.preprocess:
            x = self.do_preprocess(img)

        result = self.create_result(
            x,
            mahotas.features.haralick(x).mean(
                self.properties["mean"]
            )
        )

        LOGGER.info('ms=%s' % (ms(start)))
        return result


class RgbHistogramDescriptor(FeatureDescriptor):
    def __init__(self, preprocess=False, bins=[8, 8, 8], size=(250, 250)):
        self.name = "rgb_histogram"
        # store the number of bins the histogram will use
        self.properties["bins"] = bins
        self.properties["size"] = size
        self.preprocess = preprocess

    def do_preprocess(self, img):
        x = np.copy(img)
        return cv2.resize(x, self.properties["size"])

    def describe(self, img):
        start = time.time()
        x = img
        if self.preprocess:
            x = self.do_preprocess(img)
        # compute a 3D histogram in the RGB colorspace,
        # then normalize the histogram so that images
        # with the same content, but either scaled larger
        # or smaller will have (roughly) the same histogram
        hist = cv2.calcHist(
            [x],
            [0, 1, 2],
            None,
            self.properties["bins"],
            [0, 256, 0, 256, 0, 256]
        )
        hist = cv2.normalize(hist)

        # return out 3D histogram as a flattened array
        result = self.create_result(img, hist.flatten())
        LOGGER.info('ms=%s' % (ms(start)))
        return result


class ZernikeDescriptor(FeatureDescriptor):
    """

    """
    def __init__(self,
                 preprocess=True,
                 radius=21,
                 resize_info={"enabled": True, "width": 250, "height": 250},
                 grey_info={"enabled": True},
                 autocrop_info={"enabled": True},
                 outline_contour_info={"enabled": True},
                 add_border_info={"enabled": True, "color_value": 0, "border_size": 15}
    ):
        self.name = "zernike"
        self.properties["radius"] = radius
        self.properties["resize"] = resize_info
        self.properties["grey"] = grey_info
        self.properties["autocrop"] = autocrop_info
        self.properties["add_border"] = add_border_info
        self.properties["outline_contour"] = outline_contour_info,
        self.properties["thresh_bitwise"] = thresh_bitwise,
        self.preprocess = preprocess

    def do_preprocess(self, img):
        x = np.copy(img)
        if self.properties["autocrop"]["enabled"]:
            x = autocrop(x)

        if self.properties["grey"]["enabled"]:
            x = grey(x)

        if self.properties["thresh_bitwise"]["enabled"]:
            x = thresh_bitwise(x)

        if self.properties["outline_contour"]["enabled"]:
            x = outline_contour(x)

        if self.properties["resize"]["enabled"]:
            x = resize(
                x,
                (
                    self.properties["resize"]["width"],
                    self.properties["resize"]["height"]
                )
            )

        if self.properties["add_border"]["enabled"]:
            x = add_border(
                x,
                border_size=self.properties["add_border"]["border_size"],
                color_value=self.properties["add_border"]["color_value"]
            )


    def describe(self, img):
        start = time.time()
        x = img
        if self.preprocess:
            x = self.do_preprocess(img)

        value = mahotas.features.zernike_moments(x, self.properties["radius"])
        result = self.create_result(img, value)
        LOGGER.info('ms=%s' % (ms(start)))
        return result
