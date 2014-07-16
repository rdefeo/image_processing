import cProfile
import logging
import pstats
import StringIO
import unittest

import cv2
import numpy as np
from improc.crop import AutoCrop, Crop


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

    def test_1(self):
        img = cv2.imread('improc/data/1.jpg')
        # print cProfile.run("img = cv2.imread('improc/data/1.jpg')\nAutoCrop(img)")
        pr = cProfile.Profile()
        pr.enable()
        img, x, y, width, height = AutoCrop(img)
        pr.disable()

        s = StringIO.StringIO()
        sortby = 'cumulative'
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        print s.getvalue()

        self.assertNotEqual(img, None)
        self.assertEqual(x, 130)
        self.assertEqual(y, 128)
        self.assertEqual(width, 940)
        self.assertEqual(height, 958)

    def test_2(self):
        img = cv2.imread('improc/data/2.jpg')
        img, x, y, width, height = AutoCrop(img)

        self.assertNotEqual(img, None)
        self.assertEqual(x, 241)
        self.assertEqual(y, 15)
        self.assertEqual(width, 720)
        self.assertEqual(height, 1186)

    def test_3(self):
        img = cv2.imread('improc/data/3.jpg')
        img, x, y, width, height = AutoCrop(img)

        self.assertNotEqual(img, None)
        self.assertEqual(x, 85)
        self.assertEqual(y, 289)
        self.assertEqual(width, 1032)
        self.assertEqual(height, 636)

    def test_4(self):
        img = cv2.imread('improc/data/4.jpg')
        img, x, y, width, height = AutoCrop(img)

        self.assertNotEqual(img, None)
        self.assertEqual(x, 15)
        self.assertEqual(y, 175)
        self.assertEqual(width, 1175)
        self.assertEqual(height, 867)

    def test_5(self):
        img = cv2.imread('improc/data/5.jpg')
        img, x, y, width, height = AutoCrop(img)

        self.assertNotEqual(img, None)
        self.assertEqual(x, 33)
        self.assertEqual(y, 287)
        self.assertEqual(width, 702)
        self.assertEqual(height, 521)

    def test_6(self):
        img = cv2.imread('improc/data/6.jpg')
        img, x, y, width, height = AutoCrop(img)

        self.assertNotEqual(img, None)
        self.assertEqual(x, 18)
        self.assertEqual(y, 286)
        self.assertEqual(width, 726)
        self.assertEqual(height, 528)

    def test_7(self):
        img = cv2.imread('improc/data/7.jpg')
        img, x, y, width, height = AutoCrop(img)

        self.assertNotEqual(img, None)
        self.assertEqual(x, 55)
        self.assertEqual(y, 110)
        self.assertEqual(width, 1412)
        self.assertEqual(height, 804)

    def test_8(self):
        img = cv2.imread('improc/data/8.jpg')
        img, x, y, width, height = AutoCrop(img)

        self.assertIsNone(img)
        self.assertIsNone(x)
        self.assertIsNone(y)
        self.assertIsNone(width)
        self.assertIsNone(height)

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
  LOG_FORMAT = ('level=%(levelname)s,ts=%(asctime)s,name=%(name)s,funcName=%(funcName)s,lineno=%(lineno)s'
  ',%(message)s')
  logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)

  unittest.main()
