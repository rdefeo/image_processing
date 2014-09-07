__author__ = 'robdefeo'
from improc.features.query import do
from improc.features.comparator import ChiSquaredComparator, \
    ChebyshevComparator, CosineComparator, EuclideanComparator, \
    HammingComparator, ManhattanComparator
from descriptor_definitions import dd
from samples import samples_info
import json
import cv2
import descriptor_definitions
from matplotlib import pyplot as plt

def process_results(results, sample, key, descriptor):
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

    top_results = 14
    for i, x in enumerate(results["results"][:top_results]):
        # img = cv2.imread("data/%s.jpg" % x["key"])
        # plt.subplot(5, 3, i), plt.imshow(descriptor.do_preprocess(img),"%s_%s" %  (sample["name"], key))
        # plt.title("%s" % (x["norm"]))
        print x

def load_dataset(fname, sample):
    with open(fname, 'r') as f:
        descriptor_info = json.load(f)

    with open("datasets/image_info.json", 'r') as f:
        image_info = json.load(f)

    for key in image_info.keys():
        item = image_info[key]
        if not (
            "x" in item and item["x"] == sample["x"] and
            "y" in item and item["y"] == sample["y"] and
            "z" in item and item["z"] == sample["z"]
        ):
            descriptor_info["data"].pop(key, None)

    return descriptor_info

def test_dataset(key, total):
    descriptor_info = load_dataset("datasets/%s" % key, samples_info[0])

    for i, x in enumerate(samples_info):
        sample_image = cv2.imread(x["sample_name"])
        image_key = "%s_%s" % (key, x["sample_name"][:-10])
        methodToCall = getattr(descriptor_definitions, key)
        descriptor = methodToCall()

        plt.subplot(len(dd.items()), len(samples_info), i + total + 1), plt.imshow(descriptor.do_preprocess(sample_image), 'gray')
        # plt.title("%s_%s" % (key[0], image_key))

        sample_description = descriptor.describe(sample_image)
        res = do(sample_description, descriptor_info["data"],
                 EuclideanComparator())

        process_results(res, x, key, descriptor)

    return i + total


def test_datasets():
    total = -1
    for key in dd.keys():
        total = test_dataset(key, total + 1)

test_datasets()
# test_dataset("zernike_007", 0)
# test_dataset("lbp_001", 0)
plt.show()