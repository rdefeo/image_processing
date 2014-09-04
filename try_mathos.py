xfrom improc.data import GetterExtractor
from improc.features.descriptor import HarlickDescriptor, RgbHistogramDescriptor, ZernikeDescriptor
from improc.features.comparator import ChiSquaredComparator, EuclideanComparator, ManhattanComparator, ChebyshevComparator, CosineComparator, HammingComparator
import improc.features.query as feature_query

import mahotas as mh
import mahotas.features
import cv2

d = GetterExtractor()

docs = d.query(limit=100,
  header={
    "statusCode": {
      "$exists": True
    }
  }
  )

descriptor = RgbHistogramDescriptor(preprocess = True)
# size = (250, 250)
items = {}
for doc in docs:
  for i in doc["detail"]["images"]:
    path =  "/Users/rdefeo/Development/getter/detail/data/images/%s" % i["path"]
    key = "%s_%s" % (str(doc["_id"]["_id"]), str(i["_id"]))
    img = mh.imread(path)
    # img = cv2.resize(img, size)
    items[key] = descriptor.describe(img)

name = "536f5a1ea26d15820c9211cb.jpg"
base_path = "/Users/rdefeo/Development/getter/detail/data/images/%s" % name
print base_path
base = mh.imread(base_path)
# base = cv2.resize(base, size)
sample = descriptor.describe(base)

result = feature_query.do(sample, items, ChiSquaredComparator())

for x in result["results"][:10]:

  print x

# f = mh.imread('test_data/1.jpg', as_grey=True)#mh.demos.load('luispedro', as_grey=True)
# img = mahotas.imread('test_data/1.jpg')
# d = mahotas.features.haralick(img).mean(0)
#
#

# # import numpy as np
# # import mahotas
# # import pylab as p
# #
# # img = mahotas.imread('test_data/1.jpg')
# # T_otsu = mahotas.thresholding.otsu(img)
# # seeds,_ = mahotas.label(img > T_otsu)
# # labeled = mahotas.cwatershed(img.max() - img, seeds)
# #
# # p.imshow(labeled)
# # p.show()
#
# from __future__ import print_function
# import numpy as np
# import mahotas as mh
# from mahotas.features import surf, haralick
# import mahotas
# import mahotas.features
# from pylab import *
#
# from os import path
#
# f = mh.imread('test_data/1.jpg', as_grey=True)#mh.demos.load('luispedro', as_grey=True)
#
# img = mahotas.imread('test_data/1.jpg')
# d = mahotas.features.haralick(img).mean(0)
#
#
# f = f.astype(np.uint8)
# spoints = surf.surf(f, 4, 6, 2)
# print("Nr points:", len(spoints))
# print (spoints)
# try:
#     import milk
#     descrs = spoints[:,5:]
#     k = 5
#     values, _  =milk.kmeans(descrs, k)
#     colors = np.array([(255-52*i,25+52*i,37**i % 101) for i in range(k)])
# except:
#     values = np.zeros(100)
#     colors = np.array([(255,0,0)])
#
# print (values)
# print (colors)
# f2 = surf.show_surf(f, spoints[:10], values, colors)
# imshow(f2)
# show()
