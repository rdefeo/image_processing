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

import cv2
import matplotlib.pyplot as plt
import numpy as np

n_colors = 5

# reduced_color = self.imageFunctions.improc_reduce(img, 5)
# matrix = self.imageFunctions.improc_matrix(reduced_color)

# Load the Summer Palace photo
# china = load_sample_image("china.jpg")
# cv2.imwrite("test.jpg", china)
test_data = cv2.imread("improc/data/1.jpg")
# china = cv2.imread("test.jpg")

print type(test_data[0][0][0])
# Convert to floats instead of the default 8 bits integer coding. Dividing by
# 255 is important so that plt.imshow behaves works well on float data (need to
# be in the range [0-1]
china = np.array(test_data, dtype=np.float64) / 255
print type(china[0][0][0])
# Load Image and transform to a 2D numpy array.
w, h, d = original_shape = tuple(china.shape)
assert d == 3
image_array = np.reshape(china, (w * h, d))

print("Fitting model on a small sub-sample of the data")
t0 = time()
image_array_sample = shuffle(image_array, random_state=1)[:1000]
kmeans = KMeans(
    n_clusters=n_colors,
    random_state=1,
    precompute_distances=True).fit(image_array_sample)
print("done in %0.3fs." % (time() - t0))

# Get labels for all points
print("Predicting color indices on the full image (k-means)")
t0 = time()
labels = kmeans.predict(image_array)
print("done in %0.3fs." % (time() - t0))
cluster = kmeans.cluster_centers_
cluster = np.array(cluster, dtype=np.uint8) * 255
print cluster[0], cluster[1], cluster[2], "cluster"
# print cluster[labels[1000]]
print labels[1000]

# codebook_random = shuffle(image_array, random_state=0)[:n_colors + 1]
# print("Predicting color indices on the full image (random)")
# t0 = time()
# labels_random = pairwise_distances_argmin(codebook_random,
#                                           image_array,
#                                           axis=0)
# print("done in %0.3fs." % (time() - t0))


def recreate_image(codebook, labels, w, h):
    print "Recreate the (compressed) image from the code book & labels"
    t0 = time()
    d = codebook.shape[1]
    image = np.zeros((w, h, d))
    label_idx = 0
    for i in range(w):
        for j in range(h):
            image[i][j] = codebook[labels[label_idx]]
            label_idx += 1
    print("done in %0.3fs." % (time() - t0))
    return image

cv2.imshow('Original image (96,615 colors)', test_data)
print len(cluster), w, d, h

reshaped = np.reshape(cluster, (w, h))
cv2.imshow(
    'Quantized image (64 colors, K-Means)',
    reshaped)
# cv2.imshow(
#     'Quantized image (64 colors, K-Means)',
#     recreate_image(kmeans.cluster_centers_, labels, w, h))
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
