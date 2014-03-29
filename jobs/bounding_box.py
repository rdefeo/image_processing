#!/usr/bin/env python

'''
Description:
  For images with no bounding box it updates it with the autocrop method
  
Usage:
   ./bounding_box.py --database 'mongodb://localhost' --image-source /data/images
'''
import os, sys
sys.path.append('/usr/local/lib/python2.7/site-packages')
lib_path = os.path.abspath('../categorize')
sys.path.append(lib_path)
import common
# from commmon import contentTypeExtension 

import numpy as np
import cv2
from pymongo import MongoClient


if __name__ == '__main__':
  from optparse import OptionParser
  parser = OptionParser(usage="usage: %prog [options] filename", version="%prog 1.0")
  
  parser.add_option("-l", "--limit",
                    action="store", # optional because action defaults to "store"
                    dest="limit",
                    default="2000",
                    help="Max number of shoes (0) for all",)                    
  parser.add_option("-d", "--database",
                        action="store",
                        dest="database",
                        default='mongodb://localhost',
                        help="Mongo database server address connnection string")
  parser.add_option("-i", "--image-source",
                        action="store",
                        dest="image_source",
                        default='/getter_data/images/',
                        help="source directory of where the images are")  
                    
  (options, args) = parser.parse_args()       
  option_dict = vars(options)


  limit = int(option_dict['limit'])
  databaseUri = option_dict['database']
  image_source = option_dict['image_source']
    
  conn = MongoClient(databaseUri)
  db = conn.getter

  docs = db.shoes.aggregate( [
      { 
        "$project" : { 'shoe.images' : 1 } 
      },
      { 
        "$unwind": "$shoe.images" 
      },
      { 
         "$match": { 
            "shoe.images.features.key": {
              "$ne" : "shoe"
            }
         } 
      },
      { 
        "$limit" : limit
      }
  ])

  for doc in docs["result"]:
    image = doc["shoe"]["images"]
    print "action=create image feature shoe_id=%s,image._id=%s" % (doc["_id"]["_id"], image["_id"])
    f = image_source + str(image["_id"]) + common.contentTypeExtension[image["content-type"]]
    im = cv2.imread(f, cv2.CV_LOAD_IMAGE_GRAYSCALE)
    im, x, y, width, height = common.autoCrop(im)
        
    db.shoes.update({
      "_id._id": doc["_id"]["_id"],
      "shoe.images._id": image["_id"]
    },
    {
      "$push": { 
        "shoe.images.$.features": {
          "source": "autocrop",
          "key": "shoe",
          "x": x,
          "y": y,
          "width": width,
          "height": height
        }
      }
    })
      