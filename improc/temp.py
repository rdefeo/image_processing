import cv2
import numpy as np
from matplotlib import pyplot as plt
from crop import AutoCrop, Crop
from improc.color import Reduce, Hex, Background, Matrix
from shape import Flatten, ScaleWidth, ScaleHeight, ScaleMax, Ratio


if __name__ == '__main__':
  # img = cv2.imread('/Users/rdefeo/Development/getter/detail/data/images/536f5900a26d15820c920baf.jpg')
  img = cv2.imread('/Users/rdefeo/Development/getter/detail/data/images/536f5902a26d15820c920bc5.jpg')
  # reduced = Reduce(img, 4)

  print Ratio(img)
  d, x, y, w, h = AutoCrop(img)
  print x, y, w, h
  cv2.imshow('auto', d)
  scaled = ScaleMax(d, 580, 245)
  print scaled.shape
  cv2.imshow('scaled', scaled)
  # cv2.imshow('crop', Crop(img, x, y, w, h))
  cv2.waitKey(0)
  cv2.destroyAllWindows()

  # Scale(img, max_height = 500)
