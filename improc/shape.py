import numpy as np

def Flatten(img):
  """
  Keeps the color dimension by merges the X, Y dimensions
  """  
  return img.reshape((-1, 3)).take((0,1,2), 1)
