from scipy import ndimage
from matplotlib import pyplot as plt
import cv2
import numpy as np


def show(img):
    plt.figure()
    plt.imshow(img)


if __name__ == '__main__':
    img = ndimage.imread('ff.png')
    kernel = np.ones((5, 5), dtype=np.float32) / 25
    img_cv = cv2.filter2D(img, -1, kernel)
    kernel2 = np.ones((5, 5, 3), dtype=np.float32) / 75
    # kernel2 = np.empty((5, 5, 3), dtype=np.float32)
    # kernel2[:,:] = [0.2989, 0.5870, 0.1140]
    # kernel2 /= 25
    img2 = ndimage.convolve(img, kernel2)
    # compare the two methods
    img_diff = (np.absolute(img_cv - img2)).astype(np.uint8)
    show(img)
    show(img2)
    show(img_cv)
    show(img_diff)
    print(np.allclose(img_cv, img2))
    plt.show()
    plt.waitforbuttonpress()
