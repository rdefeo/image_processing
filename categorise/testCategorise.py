#!/usr/bin/env python

'''

Usage:
   testCategorise.py
'''

import numpy as np
import cv2
from models import KNearest, SVM
from common import preprocess_item, preprocess_hog, idCategory


if __name__ == '__main__':
    import getopt
    import sys
    print  __doc__
     
    models = [
      # RTrees, 
      KNearest, 
      # Boost, 
      SVM, 
      # MLP
    ] # NBayes
    models = dict( [(cls.__name__.lower(), cls) for cls in models] )   

    args, dummy = getopt.getopt(sys.argv[1:], '', ['model=', 'data=', 'input=', 'size='])
    args = dict(args)
    args.setdefault('--model', 'svm')    
    args.setdefault('--data', 'out/shoes_svm_100.dat')
    args.setdefault('--size', 100)
    # args.setdefault('--input', 'letter-recognition.data')
    print 'loading data %s ...' % args['--data']
    
    
    Model = models[args['--model']]
    model = Model()
    model.load(args['--data'])
    
    # model.predict

    testFile = cv2.imread('../data/category/Heels/2.jpg', cv2.CV_LOAD_IMAGE_GRAYSCALE)
    testFile = cv2.resize(testFile, (args['--size'],args['--size']))
    # print testFile
    # d = model.predictSingle(testFile)
    categoryId = model.predictSingle(np.float32(preprocess_item(testFile)))

    print idCategory[categoryId]
    