import unittest
from improc.crop import AutoCrop, Crop
import numpy as np
import cv2


class AutoCropTests(unittest.TestCase):
  def test_white_background_crop(self):
    img = cv2.imread('improc/data/white_background_data.jpg')
    img, x, y, width, height = AutoCrop(img)
    # print "x=%s,y=%s,width=%s,height=%s" % (x, y, width, height)

    self.assertNotEqual(img, None)
    self.assertEqual(x, 598)
    self.assertEqual(y, 251)
    self.assertEqual(width, 666)
    self.assertEqual(height, 874)

  def test_non_white_background_crop(self):
    img = cv2.imread('improc/data/non_white_background_data.jpg')
    img, x, y, width, height = AutoCrop(img)

    self.assertEqual(img, None)



class CropTests(unittest.TestCase):
  def test_white_background_crop(self):
    img = cv2.imread('improc/data/white_background_data.jpg')
    print img.shape
    actual = Crop(img, 100, 200, 500, 50)
    print actual.shape
    # print "x=%s,y=%s,width=%s,height=%s" % (x, y, width, height)
    # self.assertNotEqual(img, None)
    # self.assertEqual(x, 591)
    # self.assertEqual(y, 239)
    # self.assertEqual(width, 674)
    # self.assertEqual(height, 898)



if __name__ == '__main__':
  unittest.main()
