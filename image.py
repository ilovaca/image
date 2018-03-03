import numpy as np
import scipy.ndimage as ndimage
import cv2
from scipy.signal import convolve2d

if __name__ == '__main__':
    img = cv2.imread("./aa.jpg", 1)
    # kernel = [[1 / 9, 1 / 9, 1 / 9],
    #           [1 / 9, 1 / 9, 1 / 9],
    #           [1 / 9, 1 / 9, 1 / 9],]
    kernel = np.empty((5, 5))
    # kernel = [[0.25, 0.25],
    #           [0.25, 0.25]]
    # kernel = np.empty((5,5))
    kernel.fill(1/25)
    img2 = convolve2d(img, kernel, mode='same', boundary='wrap').astype(np.uint8)
    # img = cv2.fastNlMeansDenoisingColored(img, None, 10)
    # rows, cols, channels = img.shape
    # if rows % 2 != 0:
    #     rows -= 1
    # if cols % 2 != 0:
    #     cols -= 1
    # img2 = np.zeros([rows // 2, cols // 2, channels]).astype(np.uint8)
    # for i in range(img2.shape[0]):
    #     for j in range(img2.shape[1]):
    #         img2[i, j] = img[2*i, 2*j]
    cv2.imshow('ffff', img)
    cv2.imshow('dkd', img2)
    # img = cv2.fastNlMeansDenoising(img, None, 20).astype(np.float32)
    # img = ndimage.gaussian_filter(img, sigma=(5,5,0), order=0)
    gx = np.array(np.gradient(img, axis=0))
    # gx /= gx.max() / 255.0
    gx2 = np.array(np.gradient(gx, axis=0))
    gy = np.array(np.gradient(img, axis=1))
    # gy /= gy.max() / 255.0

    # gx[np.abs(gx2) < 20] = 255
    gy2 = np.array(np.gradient(gy, axis=1))
    # gy[np.abs(gy2) < 20] = 255
    grad_mag = np.hypot(gx, gy)
    # grad_mag = np.sqrt(gx ** 2 + gy ** 2)
    grad2 = np.hypot(gx2, gy2)
    lapacian = gx2 + gy2
    # grad_mag /= grad_mag.max() / 255.0
    # grad_mag[grad_mag < 200] = 0
    # lap = cv2.Laplacian(img, cv2.CV_64F)
    cv2.imshow('fff', lapacian)
    cv2.waitKey()
