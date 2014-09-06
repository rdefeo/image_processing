__author__ = 'robdefeo'
from improc.features.query import do
from improc.features.comparator import ChiSquaredComparator, \
    ChebyshevComparator, CosineComparator, EuclideanComparator, \
    HammingComparator, ManhattanComparator
from descriptor_definitions import dd
from samples import samples_info
import json
import cv2

def process_results(results, sample, key):
    minimum = results["results"][0]["value"]
    maximum = results["results"][-1:][0]["value"]
    total_range = maximum - minimum
    position = None
    actual = None
    for i, x in enumerate(results["results"]):
        if x["key"] == sample["expected"]:
            position = i
            actual = x
        x["norm"] = (x["value"] - minimum ) / total_range

    print "actual_pos=%s,dateset=%s,name=%s,image=%s,min=%s,max=%s," \
          "actual_norm=%s,total_samples=%s" % (
              position, key, sample["name"], sample["sample_name"], minimum,
              maximum,
              actual["norm"], len(results["results"])
    )

    for x in results["results"][:5]:
        pass
        # print x

def load_dataset(fname):
    with open(fname, 'r') as f:
        # read_data = f.read()
        descriptor_info = json.load(f)

    return descriptor_info

def test_dataset(key):
    descriptor_info = load_dataset("datasets/%s" % key)

    for x in samples_info:
        sample_image = cv2.imread(x["sample_name"])
        image_key = "%s_%s" % (key, x["sample_name"])
        cv2.imshow(image_key, dd[key].do_preprocess(sample_image))

        sample_description = dd[key].describe(sample_image)
        res = do(sample_description, descriptor_info["data"],
                 EuclideanComparator())

        process_results(res, x, key)

def test_datasets():
    for key in dd.keys():
        test_dataset(key)

test_datasets()
# test_dataset("zernike_004")
cv2.waitKey()