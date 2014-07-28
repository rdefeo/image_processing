# Authors: Robert Layton <robertlayton@gmail.com>
#          Olivier Grisel <olivier.grisel@ensta.org>
#          Mathieu Blondel <mathieu@mblondel.org>
#
# License: BSD 3 clause

print(__doc__)
from time import time

from sklearn.cluster import KMeans
from sklearn.datasets import load_sample_image
from sklearn.metrics import pairwise_distances_argmin
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

import cv2
import matplotlib.pyplot as plt
import numpy as np
import improc.shape as shape
import improc.color as color
import logging
LOG_FORMAT = (
    'level=%(levelname)s,ts=%(asctime)s,name=%(name)s,'
    'funcName=%(funcName)s,lineno=%(lineno)s'
    ',%(message)s')
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)

n_colors = 5

src = "improc/data/6.jpg"
test_data = cv2.imread(src)

matrix, cluster_centers_, labels = color.Matrix_scikit_kmeans(cv2.imread(src), n_colors)
print "new", matrix

reduced = color.Reduce_cv2(cv2.imread(src), n_colors)
print "old", color.Matrix(reduced)

# cluster_centers_, labels = color.Reduce_sklearn(cv2.imread("improc/data/6.jpg"), n_colors)
# china = np.array(test_data, dtype=np.float64) / 255
# # Load Image and transform to a 2D numpy array.
w, h, d = original_shape = tuple(test_data.shape)
# assert d == 3
# image_array = np.reshape(china, (w * h, d))

# print("Fitting model on a small sub-sample of the data")
# t0 = time()
# image_array_sample = shuffle(image_array, random_state=1)[:1000]
# kmeans = KMeans(
#     n_clusters=n_colors,
#     random_state=1,
#     precompute_distances=True).fit(image_array_sample)
# print("done in %0.3fs." % (time() - t0))

# Get labels for all points
# print("Predicting color indices on the full image (k-means)")
# /t0 = time()
# labels = kmeans.predict(image_array)
# print("done in %0.3fs." % (time() - t0))
# cluster = kmeans.cluster_centers_
# print "cluster", cluster
# cluster = np.array(cluster, dtype=np.uint8) * 255
# print "length", len(labels)
# print "length", type(kmeans.cluster_centers_)

# def colormatrix_image(codebook, labels, w, h):
#     print "starting"
#     t0 = time()
#     # print codebook
#     # print labels
#     flatten_image = np.array(codebook[labels] * 255, dtype=np.uint8)
#     background = color.Background_from_flattened_image(flatten_image)
#     print  background
#     y = np.bincount(labels)
#     # print "y", y[0], y
#     ii = np.nonzero(y)[0]
#     # print "ii", ii
#     rgb_colors = np.array(codebook[ii] * 255, dtype=np.uint8)
#     d = zip(rgb_colors, y[ii])
#     matrix = []
#     for rgb, count in [x for x in d if color.Hex_from_array(x[0]) != color.Hex_from_array(background)]:
#         percent = float(count) / len(labels)
#         matrix.append({
#             "hex": color.Hex_from_array(rgb),
#             "percent": percent
#         })
#
#     print matrix
#
#     print len([x for x in d if color.Hex_from_array(x[0]) != color.Hex_from_array(background)])
#     print len(d)
#
#
#     print("done in %0.3fs." % (time() - t0))


def recreate_image(codebook, labels, w, h):
    print "Recreate the (compressed) image from the code book & labels"
    t0 = time()
    d = codebook.shape[1]
    image = np.zeros((w, h, d))
    label_idx = 0
    for i in range(w):
        for j in range(h):
            # print labels[label_idx]
            image[i][j] = codebook[labels[label_idx]]
            # image[i][j] = d[labels[label_idx]]
            label_idx += 1
    print("done in %0.3fs." % (time() - t0))
    return image

cv2.imshow('Original image (96,615 colors)', test_data)
# print len(cluster), w, d, h

# cv2.imshow(
#     'Quantized image (64 colors, K-Means)',
#     reshaped)
# colormatrix_image(cluster_centers_, labels, w, h)
cv2.imshow(
    'Quantized image (64 colors, K-Means)',
    recreate_image(cluster_centers_, labels, w, h))
# cv2.imshow(
#     'Quantized image (64 colors, Random)',
#     recreate_image(codebook_random, labels_random, w, h))

cv2.waitKey(0)

# # Display all results, alongside original image
# plt.figure(1)
# plt.clf()
# ax = plt.axes([0, 0, 1, 1])
# plt.axis('off')
# plt.title('Original image (96,615 colors)')
# plt.imshow(test_data)
#
# plt.figure(2)
# plt.clf()
# ax = plt.axes([0, 0, 1, 1])
# plt.axis('off')
# plt.title('Quantized image (64 colors, K-Means)')
# plt.imshow(recreate_image(kmeans.cluster_centers_, labels, w, h))
#
# plt.figure(3)
# plt.clf()
# ax = plt.axes([0, 0, 1, 1])
# plt.axis('off')
# plt.title('Quantized image (64 colors, Random)')
# plt.imshow(recreate_image(codebook_random, labels_random, w, h))
# plt.show()
