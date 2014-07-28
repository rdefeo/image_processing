import cv2
import numpy as np
from matplotlib import pyplot as plt
from improc.crop import AutoCrop
from improc.color import Reduce, Hex, Background, Matrix
from improc.shape import Flatten

def keypoint_color(kp, image):
  return (image[int(kp.pt[1])][int(kp.pt[0])])

def keypoint_extract(kp, image):
  print "************"
  print "angle %s" % (kp.angle)
  print "pt %s,%s" % (kp.pt[0], kp.pt[1])
  print "octave %s" % (kp.octave)
  print "response %s" % (kp.response)
  print "size %s" % (kp.size)
  print "color %s" % keypoint_color(kp, image)
  print "************"

# def flatten_image(img):
#   # TODO must do the background_image_color_color filter here not in color matrix
#   return img.reshape((-1, 3)).take((0,1,2), 1)

def color_matrix(flattened_image, background_image_color):
  filtered_image = [x for x in flattened_image.tolist() if x[0] != background_image_color[0] and x[1] != background_image_color[1] and x[2] != background_image_color[2]]
  color_ids = [Hex(x[2], x[1], x[0]) for x in filtered_image]
  df = DataFrame(dict(id=color_ids, data=np.ones(len(filtered_image)).tolist()))

  grouped = df.groupby('id')['data']
  # print grouped.sum().percent()
  maximum = float(grouped.sum().max())
  # minimum = float(grouped.sum().min())
  minimum = 0 # ensure no vales end up as 0
  for name, group in grouped:
    norm = float((group.sum() - minimum) / (maximum - minimum))
    percent = float(group.sum() / len(filtered_image))
    print "color=%s,amount=%f,percentage=%f" % (name, norm, percent)


def extract_keypoint_colors(kp, img):
  colors = []
  strengths = []
  for x in kp:
    colors.append(keypoint_color(x, img))
    strengths.append(x.response)

  return np.array(colors), np.array(strengths)


if __name__ == '__main__':
  img = cv2.imread('improc/data/white_background_data.jpg')

  img, x, y, w, h = AutoCrop(img)
  print x, y, w, h
  cv2.imshow('res2',img)
  cv2.waitKey(0)
  cv2.destroyAllWindows()
