import unittest
from improc.color import IsWhite, Background, Reduce, Matrix
import cv2
import numpy as np

class IsWhiteTests(unittest.TestCase):
  def test_white(self):
    self.assertEqual(IsWhite(np.array( [255,255,255] )),True)

  def test_non_white(self):
    self.assertEqual(IsWhite(np.array( [156,34,125] )), False)

class MatrixTests(unittest.TestCase):
  def test_white(self):
    img = cv2.imread('improc/data/white_background_data.jpg')
    actual = Matrix(Reduce(img, 6))

    self.assertEqual(len(actual), 5)
    self.assertEqual(actual[0]["percent"], 0.21071464557083028)
    self.assertEqual(actual[0]["hex"], "#45443b")


class BackgroundTests(unittest.TestCase):
  def test_white(self):
    img = cv2.imread('improc/data/white_background_data.jpg')
    actual = Background(img)
    self.assertEqual(actual[0], 255)
    self.assertEqual(actual[1], 255)
    self.assertEqual(actual[2], 255)

  def test_non_white(self):
    img = cv2.imread('improc/data/non_white_background_data.jpg')
    color = Background(img)
    self.assertEqual(color, None)

class ReduceTests(unittest.TestCase):
  def test_white(self):
    img = cv2.imread('improc/data/white_background_data.jpg')
    reduced_image = Reduce(img, 8)
    colors = {}
    # for x in img.flatten():
    #   colors[x] = None
    #
    # self.assertEqual(len(colors.keys()), 8)
    # need asserts

  def test_non_white(self):
    img = cv2.imread('improc/data/non_white_background_data.jpg')
    reduced_image = Reduce(img, 4)
    # colors = {}
    # for x in reduced_image.flatten():
    #   colors[x] = None
    #
    # self.assertEqual(len(colors.keys()), 4)

if __name__ == '__main__':
  unittest.main()
