import unittest
from improc.color import IsWhite
import cv2
import numpy as np

class IsWhiteTests(unittest.TestCase):
  def test_white(self):
    self.assertEqual(IsWhite(np.array( [255,255,255] )),True)
  
  def test_non_white(self):
    self.assertEqual(IsWhite(np.array( [156,34,125] )), False)

if __name__ == '__main__':
  unittest.main()