__author__ = 'robdefeo'

from bson.objectid import ObjectId
from prproc.data.product import Product
import math
import cv2
from prproc.imageFunctions import ImageFunctions

imf = ImageFunctions()


def autocrop(img):
    (
        matrix,
        cluster_centers_,
        labels,
        background_label
    ) = imf.improc_matrix(img, 5)

    (
        img_autocrop,
        x,
        y,
        width,
        height
    ) = imf.improc_autocrop(img, labels, background_label)
    return img_autocrop


all_results = {}
all_results_detail = {}


def detail_result(comparator_type, descriptor_type, sample_url, results):
    minimum = results["results"][0]["value"]
    maximum = results["results"][-1:][0]["value"]
    total_range = maximum - minimum
    position = None
    for i, x in enumerate(results["results"]):
        if x["key"] == sample_url:
            position = i
        x["norm"] = (x["value"] - minimum ) / total_range
        # chebsy : zernite
        # hamming : rgb_histogram
        # chisquared : harlick
        if (
            not math.isnan(x["norm"]) and
            (
                # (
                #     comparator_type == "chebsy" and
                #     descriptor_type == "zernite"
                # )
                #     or
                (
                                comparator_type == "euclidean" and
                                descriptor_type == "zernite"
                )
                #     or
                # (
                #     comparator_type == "hamming" and
                #     descriptor_type == "rgb_histogram"
                # )
                    or
                (
                                comparator_type == "euclidean" and
                                descriptor_type == "rgb_histogram"
                )
                    or
                (
                    comparator_type == "chisquared" and
                    descriptor_type == "harlick"
                )
            )


        ):
            res_detail_val = {
                "key": "%s_%s" % (comparator_type, descriptor_type),
                "value": x["norm"]
            }
            res_val = x["norm"]

            if x["key"] in all_results:
                all_results[x["key"]].append(res_val)
                all_results_detail[x["key"]].append(res_detail_val)
            else:
                all_results[x["key"]] = [res_val]
                all_results_detail[x["key"]] = [res_detail_val]

    print "%s : %s" % (comparator_type, descriptor_type)
    actual = next(
        x for x in results["results"] if x["key"] == sample_url)
    print "min=%s,max=%s,actual_pos=%s,actual_norm=%s" % (
        minimum, maximum, position, actual["norm"])

    for x in results["results"][:10]:
        print x

    print "-------------------"


def compare(comparator_type, sample_url):
    from improc.features.comparator import ChiSquaredComparator, \
        EuclideanComparator, ManhattanComparator, ChebyshevComparator, \
        CosineComparator, HammingComparator
    import improc.features.query as feature_query

    if comparator_type == "euclidean":
        comparator = EuclideanComparator()
    elif comparator_type == "manhattan":
        comparator = ManhattanComparator()
    elif comparator_type == "chisquared":
        comparator = ChiSquaredComparator()
    elif comparator_type == "hamming":
        comparator = HammingComparator()
    elif comparator_type == "chebsy":
        comparator = ChebyshevComparator()
    elif comparator_type == "cosine":
        comparator = CosineComparator()

    else:
        raise Exception(comparator_type)

    result_harlick = feature_query.do(
        sample_harlick, items["harlick"], comparator)

    detail_result(comparator_type, "harlick", sample_url, result_harlick)

    result_rgb_histogram = feature_query.do(
        sample_rgb_histogram, items["rgb_histogram"], comparator)

    detail_result(comparator_type, "rgb_histogram", sample_url,
                  result_rgb_histogram)

    result_zernike = feature_query.do(
        sample_zernike, items["zernike"], comparator)

    detail_result(comparator_type, "zernite", sample_url, result_zernike)


p = Product()
p.open_connection()
x = 0
y = 0
z = 90
test_no = 2
sample_image_data = autocrop(
    cv2.imread("samples/spartoo/test_%s_%s_%s_%s.jpg" % (test_no, x, y, z)))
# print "using: x=%s,y=%s,z=%s %s " % (sample_image["x"], sample_image["y"],
# sample_image["z"], sample_image["url"])
sample_harlick = imf.improc_harlick_describe(sample_image_data)
sample_rgb_histogram = imf.rgb_histogram_describe(sample_image_data)
sample_zernike = imf.improc_zernike_describe(sample_image_data)

possible = p.create_db()["product_converse"].find({})

items = {
    "zernike": {},
    "rgb_histogram": {},
    "harlick": {}
}
for shoe in possible:
    url = shoe["sources"][0]["url"]
    for image in shoe["images"]:
        if (
                                        "x" in image and "y" in image and "z"
                            in image and
                                image["x"] == x and
                            image["y"] == y and
                        image["z"] == z
        ):
            for feature in image["features"]:
                for descriptor in feature["descriptors"]:
                    # key = "%s-%s-%s-%s" % (image["x"] if "x" in image else
                    # "x", image["y"] if "y" in image else "y", image["z"] if
                    # "z" in image else "z", url)
                    key = image["url"]
                    items[descriptor["name"]][key] = descriptor
# test 1
# sample_url = "http://cdn1.sarenza.net/static/_img/productsV4/0000108034"
# "/HD_0000108034_200974_06.jpg?201408271841"
# test 2
sample_url = "http://cdn2.sarenza.net/static/_img/productsV4/0000010180/HD_0000010180_200871_06.jpg?201407281343"

compare("manhattan", sample_url)
compare("euclidean", sample_url)
compare("chisquared", sample_url)
compare("hamming", sample_url)
compare("chebsy", sample_url)
compare("cosine", sample_url)

l = sorted(all_results.items(), key=lambda x: sum(x[1]))

"Print tadaaaa ------------------------"
for x in l[:4]:
    print x[0], sum(x[1])
    for y in all_results_detail[x[0]]:
        print y

