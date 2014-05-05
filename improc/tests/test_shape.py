import unittest
from improc.shape import Flatten
import cv2

class FlattenTests(unittest.TestCase):
  def test_white_background_crop(self):
    img = cv2.imread('improc/data/white_background_data.jpg')
    actual = Flatten(img)
    # print "x=%s,y=%s,width=%s,height=%s" % (x, y, width, height)

    self.assertNotEqual(img, None)
    self.assertEqual(actual.shape[0], 2764800)
    self.assertEqual(actual.shape[1], 3)
