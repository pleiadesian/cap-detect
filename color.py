"""
@ File:     color.py
@ Author:   wzl
@ Datetime: 2019-12-16 19:05
"""
import cv2

CANNY_LOWER_THRESHOLD = 50
CANNY_HIGHER_THRESHOLD = 150

img = cv2.imread('leaf.png',cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (512, 512))
cv2.imshow('origin', img)

# Canny edge detection
img_gaussian = cv2.GaussianBlur(img,(3,3),0)
img_canny = cv2.Canny(img_gaussian, CANNY_LOWER_THRESHOLD, CANNY_HIGHER_THRESHOLD)
cv2.imshow('Canny', img_canny)

# morphology dilate for edge
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
img_dilate = cv2.morphologyEx(img_canny, cv2.MORPH_DILATE, kernel)
cv2.imshow('dilate', img_dilate)

# fill hole
img_hole=img_dilate.copy()
cv2.floodFill(img_hole,None,(0,0),255)
img_hole = cv2.bitwise_not(img_hole)
img_fillhole = cv2.bitwise_or(img_dilate, img_hole)
cv2.imshow('hole', img_fillhole)

cv2.waitKey(0)
cv2.destroyAllWindows()