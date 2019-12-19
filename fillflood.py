"""
@ File:     color.py
@ Author:   wzl
@ Datetime: 2019-12-16 19:05
"""
import os
import cv2

CANNY_LOWER_THRESHOLD = 20 # default 50
CANNY_HIGHER_THRESHOLD = 80 # default 150
GAUSSIAN_KERNEL_SIZE = 5 # default 3
DILATE_EDGE_KERNEL_SIZE = 11 # default 3

FILL = 1

def colored_mask(img):
    """
    :param img: grayscale image
    :return: colored mask
    """
    # img = cv2.imread('leaf.png',cv2.IMREAD_GRAYSCALE)
    # img = cv2.resize(img, (512, 512))
    cv2.imshow('origin', img)

    # Canny edge detection
    img_gaussian = cv2.GaussianBlur(img,(GAUSSIAN_KERNEL_SIZE,GAUSSIAN_KERNEL_SIZE),0)
    cv2.imshow('Gaussian', img_gaussian)
    img_canny = cv2.Canny(img_gaussian, CANNY_LOWER_THRESHOLD, CANNY_HIGHER_THRESHOLD)
    cv2.imshow('Canny', img_canny)

    if FILL == 1:
        # morphology dilate for edge
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (DILATE_EDGE_KERNEL_SIZE, DILATE_EDGE_KERNEL_SIZE))
        img_dilate = cv2.morphologyEx(img_canny, cv2.MORPH_DILATE, kernel)
        # cv2.imshow('dilate', img_dilate)

        # fill hole
        img_hole=img_dilate.copy()
        cv2.floodFill(img_hole,None,(0,0),255)
        cv2.imshow('floodfill', img_hole)
        # img_hole = cv2.bitwise_not(img_hole)
        # cv2.imshow('bit not', img_hole)
        # img_fillhole = cv2.bitwise_or(img_dilate, img_hole)
        # cv2.imshow('hole', img_fillhole)
        # img_output = cv2.drawMatches(img, None, img_fillhole, None, None, None, None)
        # cv2.imshow('fillhole', img_output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # return img_fillhole

# for base_path, folder_list, file_list in os.walk('fill'):
#     for file_name in file_list:
#         filename = os.path.join(base_path,file_name)
#         if filename[-4:] != '.png' and filename[-4:] != '.jpg':
#             continue
img_train = cv2.imread('train/屏幕快照 2019-12-19 下午1.17.34.png', cv2.IMREAD_GRAYSCALE)
colored_mask(img_train)