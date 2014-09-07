__author__ = 'robdefeo'
import cv2

def median_blur(img, ksize=5):
    # ksize – aperture linear size; it must be odd and greater than 1, for example: 3, 5, 7
    # Here, the function cv2.medianBlur() takes median of all the pixels under kernel area and central element is replaced with this median value. This is highly effective against salt-and-pepper noise in the images. Interesting thing is that, in the above filters, central element is a newly calculated value which may be a pixel value in the image or a new value. But in median blurring, central element is always replaced by some pixel value in the image. It reduces the noise effectively. Its kernel size should be a positive odd integer.
    return cv2.medianBlur(img, ksize)

def gaussian_blur(img, ksize=(5, 5), sigmaX=0):
    # In this, instead of box filter, gaussian kernel is used. It is done with the function, cv2.GaussianBlur(). We should specify the width and height of kernel which should be positive and odd. We also should specify the standard deviation in X and Y direction, sigmaX and sigmaY respectively. If only sigmaX is specified, sigmaY is taken as same as sigmaX. If both are given as zeros, they are calculated from kernel size. Gaussian blurring is highly effective in removing gaussian noise from the image.
    return cv2.GaussianBlur(img, ksize, sigmaX)