import logging
from time import time

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
        cluster_centers_[labels[len(labels)-1]] * 255, dtype=np.uint8)

    top_left_hex = Hex(
        top_left[2],
        top_left[1],
        top_left[0]
    )

    bottom_right_hex = Hex(
        bottom_right[2],
        bottom_right[1],
        bottom_right[0]
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
        "rgb": pixel[::-1].tolist()
    }


def Background(img):
    top_left_pixel = img[0][0]
    bottom_right_pixel = img[::-1][0][::-1][0]
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

    background_hex = Hex_from_array(background)
    LOGGER.info("background_color_dectected=%s", background_hex)

    counts = zip(rgb_colors, labels_counts[labels_list])

    counts_without_background_color = [
        x for x in counts if Hex_from_array(x[0]) != background_hex
    ]
    without_background_count = np.sum(counts_without_background_color, axis=0)

    matrix = {
        "source": "scikit_kmeans_reduce",
        "values": []
    }
    for rgb, count in counts_without_background_color:
        percent = float(count) / without_background_count[1]
        basic_info = Basic_color_info(rgb)
        basic_info["percent"] = percent
        matrix["values"].append(basic_info)

    LOGGER.info("ms=%0.3fs.", (time() - t0))
    return matrix, cluster_centers_, labels, background_label


def Reduce_scikit_kmeans(img, number_of_colors):
    t0 = time()
    from sklearn.cluster import KMeans
    img_64 = np.array(img, dtype=np.float64) / 255
    w, h, d = tuple(img_64.shape)
    assert d == 3
    image_array = np.reshape(img_64, (w * h, d))

    from sklearn.utils import shuffle
    image_array_sample = shuffle(image_array, random_state=1)[:1000]
    kmeans = KMeans(
        n_clusters=number_of_colors,
        random_state=1,
        precompute_distances=True).fit(image_array_sample)

    labels = kmeans.predict(image_array)
    LOGGER.info("ms=%0.3fs.", (time() - t0))

    return kmeans.cluster_centers_, labels


def Hex_from_array(rgb):
    return Hex(rgb[2], rgb[1], rgb[0])


def Hex(r, g, b):
    return '0x%02x%02x%02x' % (r, g, b)
