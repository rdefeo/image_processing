import mahotas.features
import cv2

class FeatureDescriptor(object):
  name = None
  properties = {}

  def create_spec(self, img):
    spec = {
      "shape": img.shape,
      "name": self.name,
      "properties": self.properties
    }
  def create_result(self, img, value):
    return {
      "spec": self.create_spec(img),
      "value": value
    }

class HarlickDescriptor(FeatureDescriptor):
  def __init__(self, mean = 0):
    self.name = "harlick"
    self.properties["mean"] = mean

  def describe(self, img):
    return self.create_result(img, mahotas.features.haralick(img).mean(self.properties["mean"]))


class RgbHistogramDescriptor(FeatureDescriptor):
  def __init__(self, bins = [8, 8, 8]):
    self.name = "rgb_histogram"
    # store the number of bins the histogram will use
    self.properties["bins"] = bins

  def describe(self, img):
    # compute a 3D histogram in the RGB colorspace,
    # then normalize the histogram so that images
    # with the same content, but either scaled larger
    # or smaller will have (roughly) the same histogram
    hist = cv2.calcHist([img], [0, 1, 2], None, self.properties["bins"], [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist)

    # return out 3D histogram as a flattened array
    return self.create_result(img, hist.flatten())


class ZernikeDescriptor(FeatureDescriptor):
  def __init__(self, radius = 21):
    self.name = "zernike"
    self.properties["radius"] = radius

  def describe(self, img):
    processed_image = None
    if len(img.shape) == 3:
      processed_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
      processed_image = img

    value = mahotas.features.zernike_moments(processed_image, self.properties["radius"])
    return self.create_result(img, value)
