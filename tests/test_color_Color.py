import logging
from unittest import TestCase
import unittest

from improc import color


class Matrix_scikit_kmeans_tests(TestCase):
    maxDiff = None

    def test_single_looking_color(self):
        from scipy.ndimage import imread

        image = imread('data/color/single_looking_color.jpg')

        (
            matrix,
            cluster_centers_,
            labels,
            background_label
        ) = color.Matrix_scikit_kmeans(image, 5)

        self.assertDictEqual(
            {
                'source': 'scikit_kmeans_reduce',
                'values': [
                    {'percent': 0.4645138928790538, 'decimal': 8485066, 'hex': '0x8178ca', 'rgb': [129, 120, 202]},
                    {'percent': 0.038018909365900494, 'decimal': 3353946, 'hex': '0x332d5a', 'rgb': [51, 45, 90]},
                    {'percent': 0.36205078208262514, 'decimal': 10327004, 'hex': '0x9d93dc', 'rgb': [157, 147, 220]},
                    {'percent': 0.13541641567242055, 'decimal': 6182309, 'hex': '0x5e55a5', 'rgb': [94, 85, 165]}
                ],
                'version': '0.0.2'
            },
            matrix
        )

class Hex_tests(TestCase):
    def test_regular(self):
        actual = color.Hex(127, 120, 33)
        self.assertEqual(
            "0x7f7821",
            actual
        )


class Hex_from_array_Tests(TestCase):
    def test_black(self):
        actual = color.Hex_from_array([0,0,0])
        self.assertEqual(
            "0x000000",
            actual
        )

    def test_white(self):
        actual = color.Hex_from_array([255,255,255])
        self.assertEqual(
            "0xffffff",
            actual
        )

    def test_violet(self):
        actual = color.Hex_from_array([139, 0, 250])
        self.assertEqual(
            "0x8b00fa",
            actual
        )


if __name__ == '__main__':
    LOG_FORMAT = ('level=%(levelname)s,ts=%(asctime)s,name=%(name)s,funcName=%(funcName)s,lineno=%(lineno)s'
                  ',%(message)s')
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
    unittest.main()
