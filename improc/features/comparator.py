import cv2
import mahotas.features
import numpy as np
import scipy.spatial.distance


class DistanceComparator(object):
    name = None


class ChiSquaredComparator(DistanceComparator):
    def __init__(self, eps=1e-10):
        self.name = "ChiSquared"
        self.eps = eps

    def compare(self, sampleA, sampleB):
        # compute the chi-squared distance
        d = 0.5 * np.sum([((a - b) ** 2) /
                         (a + b + self.eps)
                         for (a, b) in zip(sampleA, sampleB)])

        return d


class EuclideanComparator(DistanceComparator):
    def __init__(self):
        self.name = "Euclidean"

    def compare(self, sampleA, sampleB):
        d = scipy.spatial.distance.euclidean(sampleA, sampleB)

        return d


class ManhattanComparator(DistanceComparator):
    def __init__(self):
        self.name = "Manhattan"

    def compare(self, sampleA, sampleB):
        d = scipy.spatial.distance.cityblock(sampleA, sampleB)

        return d


class ChebyshevComparator(DistanceComparator):
    def __init__(self):
        self.name = "Chebyshev"

    def compare(self, sampleA, sampleB):
        d = scipy.spatial.distance.chebyshev(sampleA, sampleB)

        return d


class CosineComparator(DistanceComparator):
    def __init__(self):
        self.name = "Cosine"

    def compare(self, sampleA, sampleB):
        d = scipy.spatial.distance.cosine(sampleA, sampleB)

        return d


class HammingComparator(DistanceComparator):
    def __init__(self):
        self.name = "Hamming"

    def compare(self, sampleA, sampleB):
        d = scipy.spatial.distance.hamming(sampleA, sampleB)

        return d
