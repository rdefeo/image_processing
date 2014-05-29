import numpy as np
import cv2

def Flatten(img):
  """
  Keeps the color dimension by merges the X, Y dimensions
  """
  return img.reshape((-1, 3)).take((0,1,2), 1)

def Ratio(img):
  return float(img.shape[1]) / img.shape[0]

def ScaleMax(img, width, height):
  proposedHeight = float(height) / img.shape[0]
  proposedWidth = float(width) / img.shape[1]
  if proposedWidth > width:
    return ScaleWidth(img, width)
  else:
    return ScaleHeight(img, height)

def ScaleHeight(img, height):
  r = float(height) / img.shape[0]
  dim = (int(img.shape[1] * r), height)

  # perform the actual resizing of the image and show it
  resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
  print "action=ScaleWidth,width=%s,height=%s" % (resized.shape[1], resized.shape[0])
  return resized

def ScaleWidth(img, width):
  r = float(width) / img.shape[1]
  dim = (width, int(img.shape[0] * r))

  # perform the actual resizing of the image and show it
  resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
  print "action=ScaleWidth,width=%s,height=%s" % (resized.shape[1], resized.shape[0])
  return resized
