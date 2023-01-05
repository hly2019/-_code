from __future__ import print_function
import cv2 as cv
import numpy as np
import argparse
from PIL import Image

src = None
erosion_size = 0
max_elem = 2
max_kernel_size = 21
def main(image):
    global src
    src = cv.imread(cv.samples.findFile(image))
    if src is None:
        print('Could not open or find the image: ', image)
        exit(0)
    dilatation(0)
# optional mapping of values with morphological shapes
def morph_shape(val):
    if val == 0:
        return cv.MORPH_RECT
    elif val == 1:
        return cv.MORPH_CROSS
    elif val == 2:
        return cv.MORPH_ELLIPSE
def dilatation(image):
    src = cv.imread(cv.samples.findFile(image))
    if src is None:
        print('Could not open or find the image: ', image)
        exit(0)
    dilatation_size = 42
    dilation_shape = morph_shape(2)
    # print("which?{}".format(cv.getTrackbarPos(title_trackbar_element_shape, title_dilation_window)))
    element = cv.getStructuringElement(dilation_shape, (2 * dilatation_size + 1, 2 * dilatation_size + 1),
                                       (dilatation_size, dilatation_size))
    dilatation_dst = cv.dilate(src, element)
    print(dilatation_dst.shape)
    Image.fromarray((dilatation_dst).astype(np.uint8)).save("xxx.jpg")
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Code for Eroding and Dilating tutorial.')
    parser.add_argument('--input', help='Path to input image.', default='./tmp.jpg')
    args = parser.parse_args()
    print(args.input)
    main(args.input)