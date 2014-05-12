import mahotas.features
import cv2
import numpy as np

class DistanceComparator(object):
  name = None


class ChiSquaredComparator(DistanceComparator):
  def __init__(self, eps = 1e-10):
    # store the number of bins the histogram will use
    self.name = "ChiSquared"
    self.eps = eps

  def compare(self, sampleA, sampleB):
    # compute the chi-squared distance
    d = 0.5 * np.sum([((a - b) ** 2) / (a + b + self.eps)
        for (a, b) in zip(sampleA, sampleB)])

    # return the chi-squared distance
    return d
