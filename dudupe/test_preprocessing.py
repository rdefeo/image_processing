__author__ = 'robdefeo'
from descriptor_definitions import dd
import descriptor_definitions
from samples import samples_info
import cv2
from matplotlib import pyplot as plt
from improc.features.comparator import ChiSquaredComparator, \
    ChebyshevComparator, CosineComparator, EuclideanComparator, \
    HammingComparator, ManhattanComparator

def test_descriptor_preprocessing_across_image(i, key, x):
    img = cv2.imread(x["sample_name"])
    methodToCall = getattr(descriptor_definitions, key)
    descriptor = methodToCall()
    preprocessed = descriptor.do_preprocess(img)

    plt.subplot(2, 3, i + 1),plt.imshow(preprocessed, 'gray')
    plt.title("%s_%s" % (key[0], x["sample_name"]))

def test_descriptor_preprocessing_across_images(key):
    for i, x in enumerate(samples_info):
        test_descriptor_preprocessing_across_image(i, key, x)

def test_image_across_descriptor(image_fname):
    img = cv2.imread(image_fname)
    first_description = None
    for i, key in enumerate(dd.keys()):
        methodToCall = getattr(descriptor_definitions, key)
        descriptor = methodToCall()
        preprocessed = descriptor.do_preprocess(img)
        description = descriptor.describe(img)
        if first_description is None:
            first_description = description
        distance = EuclideanComparator().compare(first_description["value"], description["value"])
        print "distance=%s" % distance
        plt.subplot(2, 3, i + 1), plt.imshow(preprocessed, 'gray')
        plt.title("%s_%s" % (key[0], image_fname))


test_descriptor_preprocessing_across_images("lbp_001")

# test_descriptor_preprocessing_image("zernike_006", samples_info[3])

# test_image_across_descriptor("samples/sarenza/test_1_0_0_90.jpg")


plt.show()
