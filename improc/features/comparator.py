import mahotas.features
import cv2
import numpy as np

class DistanceComparator(object):
  pass


class ChiSquaredComparator(DistanceComparator):
  def compare(self, sampleA, sampleB, eps = 1e-10):
    # compute the chi-squared distance
    d = 0.5 * np.sum([((a - b) ** 2) / (a + b + eps)
        for (a, b) in zip(sampleA, sampleB)])

    # return the chi-squared distance
    return d
