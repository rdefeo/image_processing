import logging
from time import time
import operator
import math
from improc import __version__

import numpy as np

LOGGER = logging.getLogger(__name__)


def ms(start):
    return (time() - start) * 1000


def IsWhite(pixel):
    return pixel[0] == 255 and pixel[1] == 255 and pixel[2] == 255


def IsWhiteish(pixel):
    if (
        pixel[0] < 252 or
        pixel[1] < 252 or
        pixel[2] < 252
    ):
        return False
    else:
        return True


def Background_from_kmeans_reduced_result(cluster_centers_, labels):
    top_left = np.array(cluster_centers_[labels[0]] * 255, dtype=np.uint8)

    bottom_right = np.array(
        cluster_centers_[labels[len(labels) - 1]] * 255, dtype=np.uint8)

    top_left_hex = Hex(
        top_left[0],
        top_left[1],
        top_left[2]
    )

    bottom_right_hex = Hex(
        bottom_right[0],
        bottom_right[1],
        bottom_right[2]
    )

    if bottom_right_hex == top_left_hex:
        return (
            np.array(cluster_centers_[labels[0]] * 255, dtype=np.uint8),
            labels[0]
        )
    else:
        LOGGER.info(
            "non matching background corners,top_left_hex=%s"
            ",bottom_right_hex=%s",
            top_left_hex,
            bottom_right_hex
        )
        return None, None


def Basic_color_info(pixel):
    hex_value = Hex_from_array(pixel)
    return {
        "hex": hex_value,
        "decimal": int(hex_value, 0),
        "rgb": pixel.tolist()
    }


def Background(img):
    top_left_pixel = img[0][0]
    bottom_right_pixel = img[::-1][0][::-1][0]
    # confirm this reverse order!
    top_left_hex = Hex(
        top_left_pixel[2],
        top_left_pixel[1],
        top_left_pixel[0]
    )
    bottom_right_hex = Hex(
        bottom_right_pixel[2],
        bottom_right_pixel[1],
        bottom_right_pixel[0]
    )
    if bottom_right_hex == top_left_hex:
        return top_left_pixel
    else:
        return None


def Matrix_scikit_kmeans(img, number_of_colors):
    t0 = time()
    cluster_centers_, labels = Reduce_scikit_kmeans(img, number_of_colors)
    labels_counts = np.bincount(labels)

    labels_list = np.nonzero(labels_counts)[0]
    rgb_colors = np.array(cluster_centers_[labels_list] * 255, dtype=np.uint8)

    background, background_label = Background_from_kmeans_reduced_result(
        cluster_centers_, labels)

    if background is None and background_label is None:
        return None, None, None, None


    counts = zip(rgb_colors, labels_counts[labels_list])

    #
    background_hex = Hex_from_array(background)
    LOGGER.info("background_color_dectected=%s", background_hex)

    counts_without_background_color = [
        x for x in counts if Hex_from_array(x[0]) != background_hex
        ]
    without_background_count = np.sum(counts_without_background_color, axis=0)

    matrix = {
        "source": "scikit_kmeans_reduce",
        "version": __version__,
        "values": []
    }
    for rgb, count in counts_without_background_color:
        percent = float(count) / without_background_count[1]
        basic_info = Basic_color_info(rgb)
        basic_info["percent"] = percent
        matrix["values"].append(basic_info)

    LOGGER.info("ms=%s", ms(t0))
    return matrix, cluster_centers_, labels, background_label


def Reduce_scikit_kmeans(img, number_of_colors):
    t0 = time()
    from sklearn.cluster import KMeans
    img_64 = np.array(img, dtype=np.float64) / 255
    w, h, d = tuple(img_64.shape)
    assert d == 3
    image_array = np.reshape(img_64, (w * h, d))

    LOGGER.info("shape=%s", image_array.shape)
    from sklearn.utils import resample
    image_array_sample = resample(
        image_array,
        replace=True,
        n_samples=min([image_array.shape[0], 1000]),
        random_state=1
    )

    kmeans = KMeans(
        n_clusters=number_of_colors,
        random_state=1,
        precompute_distances=True).fit(image_array_sample)

    labels = kmeans.predict(image_array)
    LOGGER.info("ms=%s", ms(t0))

    return kmeans.cluster_centers_, labels


def Hex_from_array(rgb):
    return Hex(rgb[0], rgb[1], rgb[2])


def Hex(r, g, b):
    return '0x%02x%02x%02x' % (r, g, b)


def Image_from_matrix(matrix, height=150, width=1000):
    import cv2
    # initialize the bar chart representing the relative frequency
    # of each of the colors

    bar = np.zeros((height, width, 3), dtype="uint8")
    startX = 0

    sorted_values = sorted(matrix["values"], key=operator.itemgetter("percent"),
                           reverse=True)

    for value in sorted_values:
        percent = value["percent"]
        color = np.array(value["rgb"])
        endX = startX + (percent * width)
        cv2.rectangle(
            bar,
            (int(startX), 0),
            (int(endX), height),
            color.astype("uint8").tolist(),
            -1
        )
        startX = endX

    return bar


def hex_to_rgb(hex_code):
    hex_code = hex_code.lower()
    return int(hex_code[2:4], 16), int(hex_code[4:6], 16), int(hex_code[6:8], 16)


def distance_between_colors_3d(red, green, blue, compare_red, compare_green, compare_blue):
    return math.sqrt(
        math.pow((red - compare_red), 2) +
        math.pow((green - compare_green), 2) +
        math.pow((blue - compare_blue), 2)
    )


def near_by_colors(source_hex_code, hex_codes_to_check_against, max_distance=32):
    # search near color
    red, green, blue = hex_to_rgb(source_hex_code)

    distances = []

    for hex_code in hex_codes_to_check_against:
        compare_red, compare_green, compare_blue = hex_to_rgb(hex_code)
        # 3D distense
        distance = distance_between_colors_3d(
            red, green, blue,
            compare_red, compare_green, compare_blue
        )

        if distance < max_distance:
            distances.append(
                {
                    "distance": distance,
                    "hex_code": hex_code,
                    "red_diff": red - compare_red,
                    "green_diff": green - compare_green,
                    "blue_diff": blue - compare_blue
                }
            )

    return sorted(distances, key=operator.itemgetter("distance"), reverse=False)
