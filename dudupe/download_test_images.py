__author__ = 'robdefeo'
from prproc.data.product import Product
import os.path
import urllib
import json

p = Product()
p.open_connection()

possible = p.create_db()["product_converse"].find({})
# possible = p.collection.find({})
image_info = {}
for shoe in possible:
    for image in shoe["images"]:

        image_info[str(image["_id"])] = {
            "x": image["x"] if "x" in image else "-",
            "y": image["y"] if "y" in image else "-",
            "z": image["z"] if "z" in image else "-"
        }
        fname = "data/%s.jpg" % str(image["_id"])
        if not os.path.isfile(fname):
            urllib.urlretrieve(image["url"], fname)
            print "downloading %s" % str(image["_id"])


with open("datasets/image_info.json", 'w') as outfile:
    json.dump(image_info, outfile, indent=2)

print "done"

