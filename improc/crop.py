import cv2
import numpy as np
from color import IsWhite

def AutoCrop(img):
  img = img.swapaxes(1,0)
  crop_img_rotated = []
  xfound = False
  # want to have a sligher before
  counterx = -1
  # want to have a sligher before and after
  counterw = 2
  if IsWhite(img[0][0]) and IsWhite(img[len(img)-1][0]) and IsWhite(img[len(img)-1][len(img[0])-1]) and IsWhite(img[0][len(img[0])-1]):
    for row in img:
      if len(np.where(row!=255)[0]) > 0:
        xfound = True
        counterw +=1
        crop_img_rotated.append(row)
      if not xfound:
        counterx +=1

    yfound = False
    # want to have a sligher before
    countery = -1
    # want to have a sligher before and after
    counterh = 2

    img = np.array(crop_img_rotated).swapaxes(1,0)
    crop_img = []
    for row in img:
      if len(np.where(row!=255)[0]) > 0:
        yfound = True
        counterh +=1
        crop_img.append(row)
      if not yfound:
        countery +=1

    return np.array(crop_img), counterx, countery, counterw, counterh
  else:
    return None, None, None, None, None
