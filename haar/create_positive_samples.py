#!/usr/bin/env python

'''

Usage:
   create_positive_samples.py
'''

import os, sys
lib_path = os.path.abspath('../categorize')
sys.path.append(lib_path)

import common
import numpy as np
import cv2
from pymongo import MongoClient

contentTypeExtension = {
  "image/jpeg": ".jpg"
}

  

if __name__ == '__main__':
  print __doc__

  limit = 28
  conn = MongoClient('mongodb://localhost')
  db = conn.getter
  shoes = []
  labels = []
  ids = []
  docs = db.shoes.find({ 
        "shoe.images": { 
          "$elemMatch" : {
            "_id": { 
              "$exists": True 
            }
          }
        },
        "_id.source": "zappos",
        "shoe.categories": { 
          "$in" : [
            "Heels"
            ]
        }
      },
      { 
        "shoe.categories": 1,
        "shoe.images": 1
      }    
    ).limit(25)
  
  f = open('data/info_' + str(limit) + '.dat','w')
  counter = 0
  for doc in docs:
    if len(doc["shoe"]["images"]) == 7:
      for image in doc["shoe"]["images"]:
        if counter < limit and image["shoeCount"] == 1 and image["y"] != 270 and image["y"] != 90 and image["z"] != 0:
          counter+=1
          f.write("/getter_data/images/" + str(image["_id"]) + contentTypeExtension[image["content-type"]] + "\n")
  
  f.close()
