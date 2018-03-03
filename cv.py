import cv2
from scipy import ndimage

import numpy as np
if __name__ == '__main__':
    img = cv2.imread('ff.png', 0)
    cv2.imshow('safs', img)
    kernel = np.ones((5, 5), dtype=np.float32) / 25
    img_cv = cv2.filter2D(img, -1, kernel)
    kernel2 = np.ones((5, 5), dtype=np.float32) / 25
    # kernel2 = np.empty((5, 5, 3), dtype=np.float32)
    # kernel2[:,:] = [0.2989, 0.5870, 0.1140]
    # kernel2 /= 25
    img2 = ndimage.convolve(img, kernel2)
    print(np.allclose(img_cv, img2, 1))
    img_diff = np.absolute(img2 - img_cv)
    cv2.imshow('ndimage_convolve', img2)
    cv2.imshow('opencv_filter2d', img_cv)
    cv2.imshow('diff', img_diff)
    cv2.waitKey()
