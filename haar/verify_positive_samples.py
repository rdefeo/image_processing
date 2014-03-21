#!/usr/bin/env python

'''

Usage:
   verify_positive_samples.py --input data/info.dat
'''
import os, sys
lib_path = os.path.abspath('../categorize')
sys.path.append(lib_path)

from common import mosaic
import numpy as np
import cv2

if __name__ == '__main__':  
  import getopt
  import sys

  print __doc__

  args, _ = getopt.getopt(sys.argv[1:], '', ['input='])
  args = dict(args)
  args.setdefault('--input', 'data/info.dat')
  
  print 'reading file %s' % args['--input']
  lines = [line.strip() for line in open(args['--input'])]
  
  vis = []
  f = open('./positives.txt','w')
  for line in lines:
    rows = line.split()
    source_file_name = rows[0]
    print line
    if len(rows) != 6:
      print line
    # print "%d:%d, %d:%d" % (int(rows[2]), int(rows[4])+ int(rows[2]), int(rows[3]), int(rows[5])+ int(rows[3]))
    img = cv2.imread('/' + source_file_name, cv2.CV_LOAD_IMAGE_GRAYSCALE)

    cr = img[int(rows[3]):int(rows[5])+ int(rows[3]), int(rows[2]):int(rows[4])+ int(rows[2])]
    vis.append(cv2.resize(cr, (50,50)))
    fn = './positive_images/' + source_file_name.split('/')[2]
    f.write(fn + '\n')
    cv2.imwrite(fn, cr)
    
  
  cv2.imwrite('out/test_data.jpg', mosaic(50, np.array(vis)))

  f.close()