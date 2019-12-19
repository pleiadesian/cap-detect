"""
@ File:     transform.py
@ Author:   wzl
@ Datetime: 2019-12-17 12:33
"""
import cv2
import numpy as np

GAUSSIAN_KERNEL_SIZE = 9 # default 3
BEST_POINT = 100 # default 25

def rotate(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gaussian = cv2.GaussianBlur(gray, (GAUSSIAN_KERNEL_SIZE, GAUSSIAN_KERNEL_SIZE), 0)
    corners = cv2.goodFeaturesToTrack(img_gaussian, BEST_POINT, 0.01, 10)
    corners = np.int0(corners)
    for i in corners:
        x, y = i.ravel()
        cv2.circle(img, (x, y), 3, 255, -1)
    # cv2.imshow("img", img)
    # cv2.waitKey()
    # cv2.destroyAllWindows()

    # img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # img_temp = img_gray.astype(np.float32)
    # img_gaussian = cv2.GaussianBlur(img_temp, (GAUSSIAN_KERNEL_SIZE, GAUSSIAN_KERNEL_SIZE), 0)
    # img_corner = cv2.cornerHarris(img_gaussian, 2, 3, 0.04)
    # img_dilate = cv2.dilate(img_corner, None)
    # img[img_dilate > 0.01 * img_dilate.max()] = [0, 0, 255]
    return img

image = cv2.imread('train/屏幕快照 2019-12-19 下午4.55.54.png')
imgfinal = rotate(image)
cv2.imshow("final", imgfinal)
cv2.waitKey(0)
cv2.destroyAllWindows()