__author__ = 'robdefeo'
from prproc.data.product import Product
import os.path

p = Product()
p.open_connection()

possible = p.create_db()["product_converse"].find({})
# possible = p.collection.find({})
import urllib

for shoe in possible:
    for image in shoe["images"]:
        fname = "data/%s.jpg" % str(image["_id"])
        if not os.path.isfile(fname):
            urllib.urlretrieve(image["url"], fname)
            print "downloading %s" % str(image["_id"])

print "done"

