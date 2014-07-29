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
import improc.crop as crop
import logging
LOG_FORMAT = (
    'level=%(levelname)s,ts=%(asctime)s,name=%(name)s,'
    'funcName=%(funcName)s,lineno=%(lineno)s'
    ',%(message)s')
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)

n_colors = 5

src = "improc/data/9.jpg"
# src = "improc/data/white_background_data.jpg"
test_data = cv2.imread(src)

#
#
# A = np.array([[0, 0, 0, 1, 1, 0, 0],
#            [0, 0, 0, 0, 0, 0, 0],
#            [0, 0, 1, 0, 0, 0, 0],
#            [0, 0, 1, 1, 0, 0, 0],
#            [0, 0, 0, 0, 1, 0, 0],
#            [0, 0, 0, 0, 0, 0, 0],
#            [0, 0, 0, 0, 0, 0, 100]])
#
# print "last", A[::-1][0][::-1][0]
# print "somthing"
#
# from scipy import ndimage
#
# sx = ndimage.sobel(test_data, axis=0, mode='constant')
# sy = ndimage.sobel(test_data, axis=1, mode='constant')
# sob = np.hypot(sx, sy)
start = time()
matrix, cluster_centers_, labels, background_label = color.Matrix_scikit_kmeans(test_data, n_colors)
print matrix
fast_image, fast_x, fast_y, fast_width, fast_height = crop.AutoCrop(test_data, labels, background_label)
print fast_x, fast_y, fast_width, fast_height
# autocrop_image, autocrop_x, autocrop_y, autocrop_width, autocrop_height = crop.AutoCrop(test_data)
# print autocrop_x, autocrop_y, autocrop_width, autocrop_height

print  (time() - start) * 1000
print "background_label", background_label
# t0 = time()

# print "cluster_centers_.shape", cluster_centers_.shape
# print "labels.shape", labels.shape
# print "test_data.shape", test_data.shape

# print "not sure what it is", np.memmap(test_data, dtype=np.int64, shape=test_data.shape)
# reshaped_labels = labels.reshape(test_data.shape[0], test_data.shape[1])
# reshaped_labels = np.arange(6).reshape(2,3)
# print  reshaped_labels
# print  reshaped_labels.shape
# print np.trim_zeros(reshaped_labels)
# print "cluster_centers_", cluster_centers_
#
# bounding_box = np.argwhere(reshaped_labels != background_label)
# (ystart, xstart) = bounding_box.min(0) - 1
# (ystop, xstop) = bounding_box.max(0) + 1
# print "args-start", (ystart, xstart)
# print "args-stop", (ystop, xstop)
#
# bounding_box_test_data = test_data[ystart:ystop, xstart:xstop]
# print "Atrim-cords", ystart, ystop, xstart, xstop
#
# print("things done in %0.3fs." % (time() - t0))
# from PIL import Image
# im = Image.open(src)
# print im.getbbox()
# im2=im.crop(im.getbbox())
# im2.show()


# reduced = color.Reduce_cv2(cv2.imread(src), n_colors)
# print "old", color.Matrix(reduced)

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
# cv2.imshow('bounding_box_test_data', bounding_box_test_data)
# if autocrop_image is not None:
#     cv2.imshow('autocrop_image', autocrop_image)
if fast_image is not None:
    cv2.imshow('fast_image', fast_image)
# print len(cluster), w, d, h

# cv2.imshow(
#     'Quantized image (64 colors, K-Means)',
#     reshaped)
# colormatrix_image(cluster_centers_, labels, w, h)
if cluster_centers_  is not None:
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
