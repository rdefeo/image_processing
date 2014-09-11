import unittest
from improc.color import IsWhite, Background, Reduce_scikit_kmeans
import cv2
import numpy as np
import logging

class IsWhiteTests(unittest.TestCase):
  def test_white(self):
    self.assertEqual(IsWhite(np.array( [255,255,255] )),True)

  def test_non_white(self):
    self.assertEqual(IsWhite(np.array( [156,34,125] )), False)

class Reduce_scikit_kmeans_tests(unittest.TestCase):
  def test_white(self):
      img = cv2.imread('../data/kmeans_sample_error.jpg')
      actual = Reduce_scikit_kmeans(img, 6)

# class MatrixTests(unittest.TestCase):
#   def test_white(self):
#     img = cv2.imread('improc/data/white_background_data.jpg')
#     actual = Matrix(Reduce(img, 6))
#
#     self.assertEqual(len(actual), 5)
#
#     # c1 = next((x for x in actual if x["hex"] == "0xde8765"), None)
#     # self.assertIsNotNone(c1)
#     # self.assertEqual(c1["percent"], 0.20591951097894637)
#     #
#     # c2 = next((x for x in actual if x["hex"] == "0x7b6d61"), None)
#     # self.assertIsNotNone(c2)
#     # self.assertEqual(c2["percent"], 0.20930917508700225)
#     #
#     # c3 = next((x for x in actual if x["hex"] == "0xfacbb9"), None)
#     # self.assertIsNotNone(c3)
#     # self.assertEqual(c3["percent"], 0.15326862312980707)
#     #
#     # c4 = next((x for x in actual if x["hex"] == "0xeca68a"), None)
#     # self.assertIsNotNone(c4)
#     # self.assertEqual(c4["percent"], 0.22078804523341403)
#     #
#     # c5 = next((x for x in actual if x["hex"] == "0x45443b"), None)
#     # self.assertIsNotNone(c5)
#     # self.assertEqual(c5["percent"], 0.21071464557083028)
#
#   def test_1(self):
#     img = cv2.imread('improc/data/1.jpg')
#     actual = Matrix(Reduce(img, 6))
#
#     self.assertEqual(len(actual), 5)
#
#     # c1 = next((x for x in actual if x["hex"] == "0xd4c9bc"), None)
#     # self.assertIsNotNone(c1)
#     # self.assertEqual(c1["percent"], 0.07519259286050128)
#     #
#     # c2 = next((x for x in actual if x["hex"] == "0xc5ad93"), None)
#     # self.assertIsNotNone(c2)
#     # self.assertEqual(c2["percent"], 0.21116815202336858)
#     #
#     # c3 = next((x for x in actual if x["hex"] == "0xaf947c"), None)
#     # self.assertIsNotNone(c3)
#     # self.assertEqual(c3["percent"], 0.3379783972614689)
#     #
#     # c4 = next((x for x in actual if x["hex"] == "0x6f513e"), None)
#     # self.assertIsNotNone(c4)
#     # self.assertEqual(c4["percent"], 0.0904553510072697)
#     #
#     # c5 = next((x for x in actual if x["hex"] == "0xa1836a"), None)
#     # self.assertIsNotNone(c5)
#     # self.assertEqual(c5["percent"], 0.28520550684739154)
#
#   def test_2(self):
#     img = cv2.imread('improc/data/2.jpg')
#     actual = Matrix(Reduce(img, 6))
#
#     self.assertEqual(len(actual), 5)
#   #
#   #   c1 = next((x for x in actual if x["hex"] == "0x504f4d"), None)
#   #   self.assertIsNotNone(c1)
#   #   self.assertEqual(c1["percent"], 0.4439162137932444)
#   #
#   #   c2 = next((x for x in actual if x["hex"] == "0x6e6d68"), None)
#   #   self.assertIsNotNone(c2)
#   #   self.assertEqual(c2["percent"], 0.08085433196181555)
#   #
#   #   c3 = next((x for x in actual if x["hex"] == "0xd7d6d2"), None)
#   #   self.assertIsNotNone(c3)
#   #   self.assertEqual(c3["percent"], 0.0276454767654763)
#   #
#   #   c4 = next((x for x in actual if x["hex"] == "0x44433f"), None)
#   #   self.assertIsNotNone(c4)
#   #   self.assertEqual(c4["percent"], 0.41919969162751175)
#   #
#   #   c5 = next((x for x in actual if x["hex"] == "0xa4a29c"), None)
#   #   self.assertIsNotNone(c5)
#   #   self.assertEqual(c5["percent"], 0.028384285851952004)
#   #
#   def test_3(self):
#     img = cv2.imread('improc/data/3.jpg')
#     actual = Matrix(Reduce(img, 6))
#
#     self.assertEqual(len(actual), 5)
#
#   #   c1 = next((x for x in actual if x["hex"] == "0xcbc0b1"), None)
#   #   self.assertIsNotNone(c1)
#   #   self.assertEqual(c1["percent"], 0.4113063475406908)
#   #
#   #   c2 = next((x for x in actual if x["hex"] == "0xb4a99d"), None)
#   #   self.assertIsNotNone(c2)
#   #   self.assertEqual(c2["percent"], 0.17457132331599276)
#   #
#   #   c3 = next((x for x in actual if x["hex"] == "0xded7cc"), None)
#   #   self.assertIsNotNone(c3)
#   #   self.assertEqual(c3["percent"], 0.1796813949853759)
#   #
#   #   c4 = next((x for x in actual if x["hex"] == "0x695648"), None)
#   #   self.assertIsNotNone(c4)
#   #   self.assertEqual(c4["percent"], 0.06081850454352631)
#   #
#   #   c5 = next((x for x in actual if x["hex"] == "0xa08671"), None)
#   #   self.assertIsNotNone(c5)
#   #   self.assertEqual(c5["percent"], 0.17362242961441426)
#   #
#   def test_4(self):
#     img = cv2.imread('improc/data/4.jpg')
#     actual = Matrix(Reduce(img, 6))
#
#     self.assertEqual(len(actual), 5)
#   #
#   #   c1 = next((x for x in actual if x["hex"] == "0xd0c7c2"), None)
#   #   self.assertIsNotNone(c1)
#   #   self.assertEqual(c1["percent"], 0.06327767314678372)
#   #
#   #   c2 = next((x for x in actual if x["hex"] == "0x9e8e86"), None)
#   #   self.assertIsNotNone(c2)
#   #   self.assertEqual(c2["percent"], 0.08633764378005952)
#   #
#   #   c3 = next((x for x in actual if x["hex"] == "0x2e1d18"), None)
#   #   self.assertIsNotNone(c3)
#   #   self.assertEqual(c3["percent"], 0.1795047583024037)
#   #
#   #   c4 = next((x for x in actual if x["hex"] == "0x4f372f"), None)
#   #   self.assertIsNotNone(c4)
#   #   self.assertEqual(c4["percent"], 0.4905030955889499)
#   #
#   #   c5 = next((x for x in actual if x["hex"] == "0x785a4b"), None)
#   #   self.assertIsNotNone(c5)
#   #   self.assertEqual(c5["percent"], 0.18037682918180314)
#   #
#   def test_5(self):
#     img = cv2.imread('improc/data/5.jpg')
#     actual = Matrix(Reduce(img, 6))
#
#     self.assertEqual(len(actual), 5)
#   #
#   #   c1 = next((x for x in actual if x["hex"] == "0xd9c6ba"), None)
#   #   self.assertIsNotNone(c1)
#   #   self.assertEqual(c1["percent"], 0.08567011199271694)
#   #
#   #   c2 = next((x for x in actual if x["hex"] == "0xe3d9d2"), None)
#   #   self.assertIsNotNone(c2)
#   #   self.assertEqual(c2["percent"], 0.29192663574506156)
#   #
#   #   c3 = next((x for x in actual if x["hex"] == "0xb29487"), None)
#   #   self.assertIsNotNone(c3)
#   #   self.assertEqual(c3["percent"], 0.03406870246980435)
#   #
#   #   c4 = next((x for x in actual if x["hex"] == "0xe9e8e7"), None)
#   #   self.assertIsNotNone(c4)
#   #   self.assertEqual(c4["percent"], 0.5818049113554493)
#   #
#   #   c5 = next((x for x in actual if x["hex"] == "0xaf5150"), None)
#   #   self.assertIsNotNone(c5)
#   #   self.assertEqual(c5["percent"], 0.0065296384369677996)
#   #
#   def test_6(self):
#     img = cv2.imread('improc/data/6.jpg')
#     actual = Matrix(Reduce(img, 6))
#
#     self.assertEqual(len(actual), 5)
#   #
#   #   c1 = next((x for x in actual if x["hex"] == "0xcac8c0"), None)
#   #   self.assertIsNotNone(c1)
#   #   self.assertEqual(c1["percent"], 0.11577332195020651)
#   #
#   #   c2 = next((x for x in actual if x["hex"] == "0xb0ac9e"), None)
#   #   self.assertIsNotNone(c2)
#   #   self.assertEqual(c2["percent"], 0.06802187406262258)
#   #
#   #   c3 = next((x for x in actual if x["hex"] == "0xe0dfda"), None)
#   #   self.assertIsNotNone(c3)
#   #   self.assertEqual(c3["percent"], 0.2709568748702093)
#   #
#   #   c4 = next((x for x in actual if x["hex"] == "0xeeeeeb"), None)
#   #   self.assertIsNotNone(c4)
#   #   self.assertEqual(c4["percent"], 0.48987055538891067)
#   #
#   #   c5 = next((x for x in actual if x["hex"] == "0x9e7d55"), None)
#   #   self.assertIsNotNone(c5)
#   #   self.assertEqual(c5["percent"], 0.055377373728050946)
#   #
#   def test_7(self):
#     img = cv2.imread('improc/data/7.jpg')
#     actual = Matrix(Reduce(img, 6))
#
#     self.assertEqual(len(actual), 5)
#   #
#   #   c1 = next((x for x in actual if x["hex"] == "0x87373a"), None)
#   #   self.assertIsNotNone(c1)
#   #   self.assertEqual(c1["percent"], 0.30952852464010966)
#   #
#   #   c2 = next((x for x in actual if x["hex"] == "0x9c6765"), None)
#   #   self.assertIsNotNone(c2)
#   #   self.assertEqual(c2["percent"], 0.1013808036734383)
#   #
#   #   c3 = next((x for x in actual if x["hex"] == "0xbca494"), None)
#   #   self.assertIsNotNone(c3)
#   #   self.assertEqual(c3["percent"], 0.0894530102390564)
#   #
#   #   c4 = next((x for x in actual if x["hex"] == "0x4a1f1f"), None)
#   #   self.assertIsNotNone(c4)
#   #   self.assertEqual(c4["percent"], 0.10280621960110116)
#   #
#   #   c5 = next((x for x in actual if x["hex"] == "0x732a2d"), None)
#   #   self.assertIsNotNone(c5)
#   #   self.assertEqual(c5["percent"], 0.3968314418462945)
#   #
#   def test_8(self):
#     img = cv2.imread('improc/data/8.jpg')
#     actual = Matrix(Reduce(img, 6))
#
#     self.assertEqual(len(actual), 5)
#
#   #   c1 = next((x for x in actual if x["hex"] == "0xae9184"), None)
#   #   self.assertIsNotNone(c1)
#   #   self.assertEqual(c1["percent"], 0.2432006811432685)
#   #
#   #   c2 = next((x for x in actual if x["hex"] == "0xd0b6ae"), None)
#   #   self.assertIsNotNone(c2)
#   #   self.assertEqual(c2["percent"], 0.2701891372416005)
#   #
#   #   c3 = next((x for x in actual if x["hex"] == "0x536da1"), None)
#   #   self.assertIsNotNone(c3)
#   #   self.assertEqual(c3["percent"], 0.1983160624750513)
#   #
#   #   c4 = next((x for x in actual if x["hex"] == "0x463c4b"), None)
#   #   self.assertIsNotNone(c4)
#   #   self.assertEqual(c4["percent"], 0.10516995124472471)
#   #
#   #   c5 = next((x for x in actual if x["hex"] == "0x916257"), None)
#   #   self.assertIsNotNone(c5)
#   #   self.assertEqual(c5["percent"], 0.183124167895355)
#
#
# class BackgroundTests(unittest.TestCase):
#   def test_white(self):
#     img = cv2.imread('improc/data/white_background_data.jpg')
#     actual = Background(img)
#     self.assertEqual(actual[0], 255)
#     self.assertEqual(actual[1], 255)
#     self.assertEqual(actual[2], 255)
#
#   def test_non_white(self):
#     img = cv2.imread('improc/data/non_white_background_data.jpg')
#     color = Background(img)
#     self.assertEqual(color, None)
#
# class ReduceTests(unittest.TestCase):
#   def test_white(self):
#     img = cv2.imread('improc/data/white_background_data.jpg')
#     reduced_image = Reduce(img, 8)
#     colors = {}
#     # for x in img.flatten():
#     #   colors[x] = None
#     #
#     # self.assertEqual(len(colors.keys()), 8)
#     # need asserts
#
#   def test_non_white(self):
#     img = cv2.imread('improc/data/non_white_background_data.jpg')
#     reduced_image = Reduce(img, 4)
#     # colors = {}
#     # for x in reduced_image.flatten():
#     #   colors[x] = None
#     #
#     # self.assertEqual(len(colors.keys()), 4)

if __name__ == '__main__':
  LOG_FORMAT = ('level=%(levelname)s,ts=%(asctime)s,name=%(name)s,funcName=%(funcName)s,lineno=%(lineno)s'
  ',%(message)s')
  logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
  unittest.main()
