from improc import color
from scipy.ndimage import imread
import matplotlib.pyplot as plt

__author__ = 'robdefeo'


def plot_image_bar(file):
    image = imread(file)
    (
        matrix,
        cluster_centers_,
        labels,
        background_label
    ) = color.Matrix_scikit_kmeans(image, 5)

    img = color.Image_from_matrix(matrix)

    plt.imshow(img)
    plt.show()

plot_image_bar('../../tests/data/color/red_1.jpg')
plot_image_bar('../../tests/data/color/red_2.jpg')
plot_image_bar('../../tests/data/color/red_3.jpg')
plot_image_bar('../../tests/data/color/red_4.jpg')
plot_image_bar('/home/alessio/Desktop/shoe.jpg')

pass