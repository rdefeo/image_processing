__author__ = 'robdefeo'

import cv2
from improc.color import Matrix_scikit_kmeans, Image_from_matrix
import numpy as np
from matplotlib import pyplot as plt
import operator

img = cv2.imread("samples/spartoo/test_2_0_0_90.jpg")

matrix, cluster_centers_, labels, background_label = Matrix_scikit_kmeans(img, 5)

plt.figure()
plt.axis("off")
plt.imshow(Image_from_matrix(matrix))
plt.show()
print "test"