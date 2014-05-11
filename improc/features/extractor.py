import mahotas.features
import cv2

class FeatureExtractor(object):
  pass

class HarlickExtractor(FeatureExtractor):
  def extract(self, img):
    return mahotas.features.haralick(img).mean(0)

class RgbHistogramExtrator(FeatureExtractor):
  def __init__(self, bins = [8, 8, 8]):
		# store the number of bins the histogram will use
		self.bins = bins

  def extract(self, img):
		# compute a 3D histogram in the RGB colorspace,
		# then normalize the histogram so that images
		# with the same content, but either scaled larger
		# or smaller will have (roughly) the same histogram
		hist = cv2.calcHist([img], [0, 1, 2],
			None, self.bins, [0, 256, 0, 256, 0, 256])
		hist = cv2.normalize(hist)

		# return out 3D histogram as a flattened array
		return hist.flatten()
