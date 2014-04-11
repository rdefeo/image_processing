import cv2
import numpy as np

def IsWhite(pixel):
  return pixel[0] == 255 and pixel[1] == 255 and pixel[2] == 255
