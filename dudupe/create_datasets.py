__author__ = 'robdefeo'

import cv2
import os
import json
from descriptor_definitions import dd
import descriptor_definitions

def create_ds(key):
    print "creating descriptor %s" % key
    methodToCall = getattr(descriptor_definitions, key)
    descriptor = methodToCall()
    image_files = [x for x in os.listdir("data/") if ".jpg" in x]
    descriptor_data = {
        "properties": descriptor.properties,
        "name": descriptor.name
    }
    data = {}
    for fname in image_files:
        path = "data/%s" % fname
        img = cv2.imread(path)
        # i = descriptor.do_preprocess(img)
        # cv2.imshow("test", i)

        description = descriptor.describe(img)
        if description is not None:
            description["value"] = description["value"].tolist()
            data[fname[:-4]] = description

        print "fname=%s" % fname

    descriptor_data["data"] = data
    descriptor_fname = "datasets/%s" % (key)
    with open(descriptor_fname, 'w') as outfile:
        json.dump(descriptor_data, outfile, indent=2)

for key in dd.keys():
    descriptor_fname = "datasets/%s" % (key)
    if not os.path.isfile(descriptor_fname):
        create_ds(key)


# def create_dataset(create_descriptor, dname):
#     dataset_fname = "datasets/%s" % dname
#     if not os.path.isfile(dataset_fname):
#         data = {}
#         for fname in os.listdir("data/"):
#             print "fname=%s" % fname
#             if ".jpg" in fname:
#                 image_detail ={
#                     "fname": "data/%s" % fname
#                 }
#                 # img = cv2.imread("data/%s" % fname)
#
#                 description = create_descriptor(image_detail)
#                 if description is not None:
#                     description["value"] = description["value"].tolist()
#                     data[fname[:-4]] = description
#
#         with open(dataset_fname, 'w') as outfile:
#             json.dump(data, outfile)
