import unittest
from improc.crop import AutoCrop
import cv2


class AutoCropTests(unittest.TestCase):
  def test_white_background_crop(self):
    img = cv2.imread('improc/data/white_background_data.jpg')
    img, x, y, width, height = AutoCrop(img)
    # print "x=%s,y=%s,width=%s,height=%s" % (x, y, width, height)
    
    self.assertNotEqual(img, None)
    self.assertEqual(x, 590)
    self.assertEqual(y, 238)
    self.assertEqual(width, 676)
    self.assertEqual(height, 900)

  def test_non_white_background_crop(self):
    img = cv2.imread('improc/data/non_white_background_data.jpg')
    img, x, y, width, height = AutoCrop(img)
    
    self.assertEqual(img, None)


if __name__ == '__main__':
  unittest.main()