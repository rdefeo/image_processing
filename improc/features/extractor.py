import mahotas.features

class FeatureExtractor(object):
  pass

class HarlickExtractor(FeatureExtractor):
  def extract(self, img):
    return mahotas.features.haralick(img).mean(0)
