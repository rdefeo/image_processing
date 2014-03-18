#!/usr/bin/env python

'''

Usage:
   create_negative_samples.py
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
          "$nin" : [
            "Heels"
            ]
        }
      },
      { 
        "shoe.categories": 1,
        "shoe.images": 1
      }    
    ).limit(10)
  
  f = open('data/bg.txt','w')
  
  for doc in docs:
    if len(doc["shoe"]["images"]) == 7:
      for image in doc["shoe"]["images"]:
        f.write("/getter_data/images/" + str(image["_id"]) + contentTypeExtension[image["content-type"]] + "\n")
  
  f.close()
  