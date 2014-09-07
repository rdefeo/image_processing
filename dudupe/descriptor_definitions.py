__author__ = 'robdefeo'

from improc.features.descriptor import ZernikeDescriptor, LinearBinaryPatternsDescriptor

def zernike_001():
    return ZernikeDescriptor(
        preprocess=True,
        radius=1,
        resize={"enabled": False, "width": 250, "height": 250},
        grey={"enabled": True},
        autocrop={"enabled": True},
        outline_contour={"enabled": True},
        add_border={"enabled": True, "color_value": 0, "border_size": 15,
                    "fill_dimensions": True},
        bitwise_info={"enabled": True},
        thresh={"enabled": False},
        scale_max={"enabled": True, "width": 250, "height": 250},
        dilate={"enabled": True, "width": 7, "height": 7, "iterations": 1},
        closing={"enabled": True, "width": 5, "height": 5},
        canny={"enabled": True, "threshold1": 100, "threshold2": 200},
        gaussian_blur={"enabled": True, "ksize_width": 5, "ksize_height": 5,
                       "sigmaX": 0},
        laplacian={"enabled": False}
    )

def zernike_002():
    return ZernikeDescriptor(
        preprocess=True,
        radius=3,
        resize={"enabled": False, "width": 250, "height": 250},
        grey={"enabled": True},
        autocrop={"enabled": True},
        outline_contour={"enabled": True},
        add_border={"enabled": True, "color_value": 0, "border_size": 15,
                    "fill_dimensions": True},
        bitwise_info={"enabled": True},
        thresh={"enabled": False},
        scale_max={"enabled": True, "width": 250, "height": 250},
        dilate={"enabled": True, "width": 7, "height": 7, "iterations": 1},
        closing={"enabled": True, "width": 5, "height": 5},
        canny={"enabled": True, "threshold1": 100, "threshold2": 200},
        gaussian_blur={"enabled": True, "ksize_width": 5, "ksize_height": 5,
                       "sigmaX": 0},
        laplacian={"enabled": False}
    )

def zernike_003():
    return ZernikeDescriptor(
        preprocess=True,
        radius=11,
        resize={"enabled": False, "width": 250, "height": 250},
        grey={"enabled": True},
        autocrop={"enabled": True},
        outline_contour={"enabled": True},
        add_border={"enabled": True, "color_value": 0, "border_size": 15,
                    "fill_dimensions": True},
        bitwise_info={"enabled": True},
        thresh={"enabled": False},
        scale_max={"enabled": True, "width": 250, "height": 250},
        dilate={"enabled": True, "width": 7, "height": 7, "iterations": 1},
        closing={"enabled": True, "width": 5, "height": 5},
        canny={"enabled": True, "threshold1": 100, "threshold2": 200},
        gaussian_blur={"enabled": True, "ksize_width": 5, "ksize_height": 5,
                       "sigmaX": 0},
        laplacian={"enabled": False}
    )
def zernike_004():
    return ZernikeDescriptor(
        preprocess=True,
        radius=21,
        resize={"enabled": False, "width": 250, "height": 250},
        grey={"enabled": True},
        autocrop={"enabled": True},
        outline_contour={"enabled": True},
        add_border={"enabled": True, "color_value": 0, "border_size": 15,
                    "fill_dimensions": True},
        bitwise_info={"enabled": True},
        thresh={"enabled": False},
        scale_max={"enabled": True, "width": 250, "height": 250},
        dilate={"enabled": True, "width": 7, "height": 7, "iterations": 1},
        closing={"enabled": True, "width": 5, "height": 5},
        canny={"enabled": True, "threshold1": 100, "threshold2": 200},
        gaussian_blur={"enabled": True, "ksize_width": 5, "ksize_height": 5,
                       "sigmaX": 0},
        laplacian={"enabled": False}
    )

def zernike_005():
    return ZernikeDescriptor(
        preprocess=True,
        radius=42,
        resize={"enabled": False, "width": 250, "height": 250},
        grey={"enabled": True},
        autocrop={"enabled": True},
        outline_contour={"enabled": True},
        add_border={"enabled": True, "color_value": 0, "border_size": 15,
                    "fill_dimensions": True},
        bitwise_info={"enabled": True},
        thresh={"enabled": False},
        scale_max={"enabled": True, "width": 250, "height": 250},
        dilate={"enabled": True, "width": 7, "height": 7, "iterations": 1},
        closing={"enabled": True, "width": 5, "height": 5},
        canny={"enabled": True, "threshold1": 100, "threshold2": 200},
        gaussian_blur={"enabled": True, "ksize_width": 5, "ksize_height": 5,
                       "sigmaX": 0},
        laplacian={"enabled": False}
    )

def zernike_006():
    return ZernikeDescriptor(
        preprocess=True,
        radius=84,
        resize={"enabled": False, "width": 250, "height": 250},
        grey={"enabled": True},
        autocrop={"enabled": True},
        outline_contour={"enabled": True},
        add_border={"enabled": True, "color_value": 0, "border_size": 15,
                    "fill_dimensions": True},
        bitwise_info={"enabled": True},
        thresh={"enabled": False},
        scale_max={"enabled": True, "width": 250, "height": 250},
        dilate={"enabled": True, "width": 7, "height": 7, "iterations": 1},
        closing={"enabled": True, "width": 5, "height": 5},
        canny={"enabled": True, "threshold1": 100, "threshold2": 200},
        gaussian_blur={"enabled": True, "ksize_width": 5, "ksize_height": 5,
                       "sigmaX": 0},
        laplacian={"enabled": False}

    )

def zernike_007():
    return ZernikeDescriptor(
        preprocess=True,
        radius=168,
        resize={"enabled": False, "width": 250, "height": 250},
        grey={"enabled": True},
        autocrop={"enabled": True},
        outline_contour={"enabled": True},
        add_border={"enabled": True, "color_value": 0, "border_size": 15,
                    "fill_dimensions": True},
        bitwise_info={"enabled": True},
        thresh={"enabled": False},
        scale_max={"enabled": True, "width": 250, "height": 250},
        dilate={"enabled": True, "width": 7, "height": 7, "iterations": 1},
        closing={"enabled": True, "width": 5, "height": 5},
        canny={"enabled": True, "threshold1": 100, "threshold2": 200},
        gaussian_blur={"enabled": True, "ksize_width": 5, "ksize_height": 5,
                       "sigmaX": 0},
        laplacian={"enabled": False}

    )

def zernike_008():
    return ZernikeDescriptor(
        preprocess=True,
        radius=84,
        resize={"enabled": False, "width": 250, "height": 250},
        grey={"enabled": True},
        autocrop={"enabled": True},
        outline_contour={"enabled": False},
        add_border={"enabled": True, "color_value": 0, "border_size": 15,
                    "fill_dimensions": True},
        bitwise_info={"enabled": False},
        thresh={"enabled": True},
        scale_max={"enabled": True, "width": 250, "height": 250},
        dilate={"enabled": True, "width": 5, "height": 5, "iterations": 1},
        closing={"enabled": True, "width": 5, "height": 5},
        canny={"enabled": True, "threshold1": 100, "threshold2": 200},
        gaussian_blur={"enabled": True, "ksize_width": 7, "ksize_height": 7,
                       "sigmaX": 0},
        laplacian={"enabled": True}
    )

def lbp_001():
    return LinearBinaryPatternsDescriptor(
        preprocess=True,
        radius=84,
        number_of_points=10,
        ignore_zeros=False,
        resize={"enabled": False, "width": 250, "height": 250},
        grey={"enabled": True},
        autocrop={"enabled": True},
        outline_contour={"enabled": True},
        add_border={"enabled": True, "color_value": 0, "border_size": 15,
                    "fill_dimensions": True},
        bitwise_info={"enabled": True},
        thresh={"enabled": False},
        scale_max={"enabled": True, "width": 250, "height": 250},
        dilate={"enabled": True, "width": 7, "height": 7, "iterations": 1},
        closing={"enabled": True, "width": 5, "height": 5},
        canny={"enabled": True, "threshold1": 100, "threshold2": 200},
        gaussian_blur={"enabled": True, "ksize_width": 5, "ksize_height": 5,
                       "sigmaX": 0},
        laplacian={"enabled": False}

    )
def lbp_002():
    return LinearBinaryPatternsDescriptor(
        preprocess=True,
        radius=42,
        number_of_points=10,
        ignore_zeros=False,
        resize={"enabled": False, "width": 250, "height": 250},
        grey={"enabled": True},
        autocrop={"enabled": True},
        outline_contour={"enabled": True},
        add_border={"enabled": True, "color_value": 0, "border_size": 15,
                    "fill_dimensions": True},
        bitwise_info={"enabled": True},
        thresh={"enabled": False},
        scale_max={"enabled": True, "width": 250, "height": 250},
        dilate={"enabled": True, "width": 7, "height": 7, "iterations": 1},
        closing={"enabled": True, "width": 5, "height": 5},
        canny={"enabled": True, "threshold1": 100, "threshold2": 200},
        gaussian_blur={"enabled": True, "ksize_width": 5, "ksize_height": 5,
                       "sigmaX": 0},
        laplacian={"enabled": False}

    )
def lbp_003():
    return LinearBinaryPatternsDescriptor(
        preprocess=True,
        radius=168,
        number_of_points=10,
        ignore_zeros=False,
        resize={"enabled": False, "width": 250, "height": 250},
        grey={"enabled": False},
        autocrop={"enabled": False},
        outline_contour={"enabled": False},
        add_border={"enabled": False, "color_value": 0, "border_size": 15,
                    "fill_dimensions": True},
        bitwise_info={"enabled": False},
        thresh={"enabled": False},
        scale_max={"enabled": False, "width": 250, "height": 250},
        dilate={"enabled": True, "width": 9, "height": 9, "iterations": 1},
        closing={"enabled": False, "width": 5, "height": 5},
        canny={"enabled": True, "threshold1": 100, "threshold2": 200},
        gaussian_blur={"enabled": False, "ksize_width": 5, "ksize_height": 5,
                       "sigmaX": 0},
        median_blur={"enabled": True, "ksize": 23},
        laplacian={"enabled": False}

    )

dd = {
    # "zernike_001": "zernike_001",
    # "zernike_002": "zernike_002",
    # "zernike_003": "zernike_003",
    # "zernike_004": "zernike_004",
    # "zernike_005": "zernike_005",
    # "zernike_006": "zernike_006",
    "zernike_007": "zernike_007",
    "zernike_008": "zernike_008",
    "lbp_001": "lbp_001",
    "lbp_002": "lbp_002",
    "lbp_003": "lbp_003"

}
