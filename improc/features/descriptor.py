import mahotas.features
import cv2

class FeatureDescriptor(object):
  pass

class HarlickDescriptor(FeatureDescriptor):
  def describe(self, img):
    return mahotas.features.haralick(img).mean(0)

class RgbHistogramDescriptor(FeatureDescriptor):
  def __init__(self, bins = [8, 8, 8]):
		# store the number of bins the histogram will use
		self.bins = bins

  def describe(self, img):
		# compute a 3D histogram in the RGB colorspace,
		# then normalize the histogram so that images
		# with the same content, but either scaled larger
		# or smaller will have (roughly) the same histogram
		hist = cv2.calcHist([img], [0, 1, 2],
			None, self.bins, [0, 256, 0, 256, 0, 256])
		hist = cv2.normalize(hist)

		# return out 3D histogram as a flattened array
		return hist.flatten()

class ZernikeDescriptor(FeatureDescriptor):
  def __init__(self, radius = 21):
    self.radius = radius

  def describe(self, img):
    processed_image = None
    if len(img.shape) == 3:
      processed_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
      processed_image = img

    return mahotas.features.zernike_moments(processed_image, self.radius)
