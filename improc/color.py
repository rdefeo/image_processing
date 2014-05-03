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

def Reduce(img, number_of_colors):
  ####
  #Reduce image to, number of colors supplied that are most relevant
  ####

  Z = img.reshape((-1,3))
  # convert to np.float32
  Z = np.float32(Z)

  K = number_of_colors
  criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

  ret,label,center=cv2.kmeans(Z,K,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

  # Now convert back into uint8, and make original image
  center = np.uint8(center)
  res = center[label.flatten()]
  res2 = res.reshape((img.shape))

  return res2
