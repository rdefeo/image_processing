import logging
import time

import uuid

import cv2
import mahotas.features
import numpy as np
import preprocess
import improc.color


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
    def __init__(
            self,
            preprocess=True,
            bins=[8, 8, 8],
            resize={"enabled": True, "width": 250, "height": 250},
            grey={"enabled": False},
            autocrop={"enabled": True},
            gaussian_blur={"enabled": False, "ksize_width": 5,
                           "ksize_height": 5, "sigmaX": 0},
            median_blur={"enabled": False, "ksize": 5},
            scale_max={"enabled": True, "width": 250, "height": 250},
            convert_to_matrix_colors={"enabled": False, "height": 1, "width": 1000, "number_of_colors": 5}

    ):
        self.name = "rgb_histogram"
        # store the number of bins the histogram will use
        self.properties["bins"] = bins
        self.properties["resize"] = resize
        self.properties["grey"] = grey
        self.properties["autocrop"] = autocrop
        self.properties["scale_max"] = scale_max
        self.properties["gaussian_blur"] = gaussian_blur
        self.properties["median_blur"] = median_blur
        self.properties["convert_to_matrix_colors"] = convert_to_matrix_colors

        self.preprocess = preprocess

    def do_preprocess(self, img):
        x = np.copy(img)

        if self.properties["autocrop"]["enabled"]:
            x = preprocess.autocrop(x)

        if x is None:
            return None

        x = preprocess.blur(
            x,
            gaussian_blur=self.properties["gaussian_blur"],
            median_blur=self.properties["median_blur"]

        )

        if self.properties["grey"]["enabled"]:
            x = preprocess.grey(x)

        if self.properties["resize"]["enabled"]:
            x = preprocess.resize(
                x,
                (
                    self.properties["resize"]["width"],
                    self.properties["resize"]["height"]
                )
            )
        elif self.properties["scale_max"]["enabled"]:
            x = preprocess.scale_max(
                x,
                self.properties["scale_max"]["width"],
                self.properties["scale_max"]["height"]
            )

        if self.properties["convert_to_matrix_colors"]["enabled"]:
            (
                matrix, cluster_centers_, labels, background_label
            ) = improc.color.Matrix_scikit_kmeans(
                x,
                self.properties["convert_to_matrix_colors"]["number_of_colors"]
            )
            x = improc.color.Image_from_matrix(
                matrix,
                self.properties["convert_to_matrix_colors"]["height"],
                self.properties["convert_to_matrix_colors"]["width"]
            )

        return x

    def describe(self, img):
        start = time.time()
        x = img
        if self.preprocess:
            x = self.do_preprocess(img)
            if x is None:
                return None


        # compute a 3D histogram in the RGB colorspace,
        # then normalize the histogram so that images
        # with the same content, but either scaled larger
        # or smaller will have (roughly) the same histogram
        if self.properties["grey"]["enabled"]:
            hist = cv2.calcHist(
                [x],
                [0],
                None,
                [self.properties["bins"][0]],
                [0,256]
            )
        else:
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
                 dilate={"enabled": False, "width": 5, "height": 5, "iterations": 1},
                 closing={"enabled": False, "width": 5, "height": 5},
                 resize={"enabled": False, "width": 250, "height": 250},
                 grey={"enabled": True},
                 autocrop={"enabled": True},
                 canny={"enabled": True, "threshold1": 100, "threshold2": 200},
                 outline_contour={"enabled": True},
                 add_border={"enabled": True, "color_value": 0, "border_size": 15, "fill_dimensions": True},
                 bitwise_info={"enabled": True},
                 thresh={"enabled": True},
                 gaussian_blur={"enabled": False, "ksize_width": 5, "ksize_height": 5, "sigmaX": 0},
                 median_blur={"enabled": False, "ksize": 5},
                 scale_max={"enabled": True, "width": 250, "height": 250},
                 laplacian={"enabled": False}
    ):
        self.name = "zernike"
        self.properties["radius"] = radius
        self.properties["resize"] = resize
        self.properties["grey"] = grey
        self.properties["autocrop"] = autocrop
        self.properties["add_border"] = add_border
        self.properties["outline_contour"] = outline_contour
        self.properties["bitwise"] = bitwise_info
        self.properties["thresh"] = thresh
        self.properties["scale_max"] = scale_max
        self.properties["canny"] = canny
        self.properties["gaussian_blur"] = gaussian_blur
        self.properties["median_blur"] = median_blur
        self.properties["laplacian"] = laplacian
        self.properties["dilate"] = dilate
        self.properties["closing"] = closing

        self.preprocess = preprocess

    def do_preprocess(self, img):
        x = np.copy(img)
        if self.properties["autocrop"]["enabled"]:
            x = preprocess.autocrop(x)

        if x is None:
            return None

        x = preprocess.blur(
            x,
            gaussian_blur=self.properties["gaussian_blur"],
            median_blur=self.properties["median_blur"]

        )

        if self.properties["grey"]["enabled"]:
            x = preprocess.grey(x)

        if self.properties["bitwise"]["enabled"]:
            x = preprocess.bitwise(x)


        if self.properties["canny"]["enabled"]:
            x = preprocess.canny(
                x,
                self.properties["canny"]["threshold1"],
                self.properties["canny"]["threshold2"]
            )

        if self.properties["laplacian"]["enabled"]:
            x = preprocess.laplacian(x)

        if self.properties["thresh"]["enabled"]:
            x = preprocess.thresh(x)

        if self.properties["closing"]["enabled"]:
            x = preprocess.closing(
                x,
                self.properties["closing"]["width"],
                self.properties["closing"]["height"]
            )

        if self.properties["dilate"]["enabled"]:
            x = preprocess.dilate(
                x,
                self.properties["dilate"]["width"],
                self.properties["dilate"]["height"],
                self.properties["dilate"]["iterations"]
            )

        if self.properties["outline_contour"]["enabled"]:
            x = preprocess.outline_contour(x)


        if self.properties["resize"]["enabled"]:
            x = preprocess.resize(
                x,
                (
                    self.properties["resize"]["width"],
                    self.properties["resize"]["height"]
                )
            )
        elif self.properties["scale_max"]["enabled"]:
            x = preprocess.scale_max(
                x,
                self.properties["scale_max"]["width"],
                self.properties["scale_max"]["height"]
            )

        if self.properties["add_border"]["enabled"]:
            x = preprocess.add_border(
                x,
                border_size=self.properties["add_border"]["border_size"],
                color_value=self.properties["add_border"]["color_value"],
                fill_dimensions=self.properties["add_border"]["fill_dimensions"]
            )

        return x

    def describe(self, img):
        start = time.time()
        x = img
        if self.preprocess:
            x = self.do_preprocess(img)
            if x is None:
                return None

            # cv2.destroyWindow("preprocessed")
            # cv2.imshow("preprocessed", x)
            # cv2.imshow(str(uuid.uuid4()), x)


        value = mahotas.features.zernike_moments(x, self.properties["radius"])
        result = self.create_result(img, value)
        LOGGER.info('ms=%s' % (ms(start)))
        return result

class LinearBinaryPatternsDescriptor(FeatureDescriptor):
    """

    """
    def __init__(self,
                 preprocess=True,
                 radius=21,
                 number_of_points=100,
                 ignore_zeros=False,
                 dilate={"enabled": False, "width": 5, "height": 5, "iterations": 1},
                 closing={"enabled": False, "width": 5, "height": 5},
                 resize={"enabled": False, "width": 250, "height": 250},
                 grey={"enabled": True},
                 autocrop={"enabled": True},
                 canny={"enabled": True, "threshold1": 100, "threshold2": 200},
                 outline_contour={"enabled": True},
                 add_border={"enabled": True, "color_value": 0, "border_size": 15, "fill_dimensions": True},
                 bitwise_info={"enabled": True},
                 thresh={"enabled": True},
                 gaussian_blur={"enabled": False, "ksize_width": 5, "ksize_height": 5, "sigmaX": 0},
                 median_blur={"enabled": False, "ksize": 5},
                 scale_max={"enabled": True, "width": 250, "height": 250},
                 laplacian={"enabled": False}
    ):
        self.name = "linear_binary_patterns"
        self.properties["radius"] = radius
        self.properties["number_of_points"] = number_of_points
        self.properties["ignore_zeros"] = ignore_zeros

        self.properties["resize"] = resize
        self.properties["grey"] = grey
        self.properties["autocrop"] = autocrop
        self.properties["add_border"] = add_border
        self.properties["outline_contour"] = outline_contour
        self.properties["bitwise"] = bitwise_info
        self.properties["thresh"] = thresh
        self.properties["scale_max"] = scale_max
        self.properties["canny"] = canny
        self.properties["gaussian_blur"] = gaussian_blur
        self.properties["median_blur"] = median_blur
        self.properties["laplacian"] = laplacian
        self.properties["dilate"] = dilate
        self.properties["closing"] = closing

        self.preprocess = preprocess

    def do_preprocess(self, img):
        x = np.copy(img)
        if self.properties["autocrop"]["enabled"]:
            x = preprocess.autocrop(x)

        if x is None:
            return None

        x = preprocess.blur(
            x,
            gaussian_blur=self.properties["gaussian_blur"],
            median_blur=self.properties["median_blur"]
        )

        if self.properties["grey"]["enabled"]:
            x = preprocess.grey(x)

        if self.properties["bitwise"]["enabled"]:
            x = preprocess.bitwise(x)


        if self.properties["canny"]["enabled"]:
            x = preprocess.canny(
                x,
                self.properties["canny"]["threshold1"],
                self.properties["canny"]["threshold2"]
            )

        if self.properties["laplacian"]["enabled"]:
            x = preprocess.laplacian(x)

        if self.properties["thresh"]["enabled"]:
            x = preprocess.thresh(x)

        if self.properties["closing"]["enabled"]:
            x = preprocess.closing(
                x,
                self.properties["closing"]["width"],
                self.properties["closing"]["height"]
            )

        if self.properties["dilate"]["enabled"]:
            x = preprocess.dilate(
                x,
                self.properties["dilate"]["width"],
                self.properties["dilate"]["height"],
                self.properties["dilate"]["iterations"]
            )

        if self.properties["outline_contour"]["enabled"]:
            x = preprocess.outline_contour(x)


        if self.properties["resize"]["enabled"]:
            x = preprocess.resize(
                x,
                (
                    self.properties["resize"]["width"],
                    self.properties["resize"]["height"]
                )
            )
        elif self.properties["scale_max"]["enabled"]:
            x = preprocess.scale_max(
                x,
                self.properties["scale_max"]["width"],
                self.properties["scale_max"]["height"]
            )

        if self.properties["add_border"]["enabled"]:
            x = preprocess.add_border(
                x,
                border_size=self.properties["add_border"]["border_size"],
                color_value=self.properties["add_border"]["color_value"],
                fill_dimensions=self.properties["add_border"]["fill_dimensions"]
            )

        return x

    def describe(self, img):
        start = time.time()
        x = img
        if self.preprocess:
            x = self.do_preprocess(img)
            if x is None:
                return None

            # cv2.destroyWindow("preprocessed")
            # cv2.imshow("preprocessed", x)
            # cv2.imshow(str(uuid.uuid4()), x)


        value = mahotas.features.lbp(
            x,
            self.properties["radius"],
            self.properties["number_of_points"],
            self.properties["ignore_zeros"]
        )
        result = self.create_result(img, value)
        LOGGER.info('ms=%s' % (ms(start)))
        return result
