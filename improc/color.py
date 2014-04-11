import cv2
import numpy as np

def IsWhite(pixel):
  return pixel[0] == 255 and pixel[1] == 255 and pixel[2] == 255
  
def Background(img):
  if img[0][0][0] == img[len(img)-1][0][0] == img[len(img)-1][len(img[0])-1][0] == img[0][len(img[0])-1][0] \
    and img[0][0][1] == img[len(img)-1][0][1] == img[len(img)-1][len(img[0])-1][1] == img[0][len(img[0])-1][1] \
    and img[0][0][2] == img[len(img)-1][0][2] == img[len(img)-1][len(img[0])-1][2] == img[0][len(img[0])-1][2]:
    return img[0][0]
  else:
    return None
