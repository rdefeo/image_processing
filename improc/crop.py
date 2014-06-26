import cv2
import numpy as np
import logging
import time
from color import IsWhite

LOGGER = logging.getLogger(__name__)

def ms(start):
  return (time.time() - start) * 1000

def Crop(img, x, y, width, height):
  return img[y:y + height, x:x + width] # Crop from x, y, w, h -> 100, 200, 300, 400
# NOTE: its img[y: y + h, x: x + w] and *not* img[x: x + w, y: y + h]

def AutoCrop(img):
  """
  returns x, y, width, height
  """
  start = time.time()
  img = img.swapaxes(1,0)
  crop_img_rotated = []
  xfound = False
  # want to have a sligher before
  counterx = -1

  return_value = None, None, None, None, None
  if IsWhite(img[0][0]) and IsWhite(img[len(img)-1][0]) and IsWhite(img[len(img)-1][len(img[0])-1]) and IsWhite(img[0][len(img[0])-1]):
    for row in img:
      non_white_pixels = [x for x in row if x[0] < 253 and x[1] < 253 and x[2] < 253]
      if len(non_white_pixels) > 10:
        xfound = True
        crop_img_rotated.append(row)

      elif not xfound:
        counterx += 1

    yfound = False
    # want to have a sligther before
    countery = -1

    img = np.array(crop_img_rotated).swapaxes(1,0)
    crop_img = []
    for row in img:
      non_white_pixels = [x for x in row if x[0] < 253 and x[1] < 253 and x[2] < 253]
      if len(non_white_pixels) > 10:
        yfound = True
        crop_img.append(row)
      elif not yfound:
        countery += 1

    cropped =  np.array(crop_img)

    if countery == -1 or counterx == -1:
      return_value = None, None, None, None, None
    elif not IsWhite(cropped[0][0]) and not IsWhite(cropped[len(cropped)-1][0]) and not IsWhite(cropped[len(cropped)-1][len(cropped[0])-1]) and not IsWhite(cropped[0][len(cropped[0])-1]):
      return_value = None, None, None, None, None
    else:
      return_value = cropped, counterx, countery, cropped.shape[1] + 2, cropped.shape[0] + 2

  else:
    return_value = None, None, None, None, None

  LOGGER.info('ms=%s' % (ms(start)))
  return return_value
