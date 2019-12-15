"""
@ File:     sift.py
@ Author:   wzl
@ Datetime: 2019-12-15 22:30
"""
import cv2
#这里使用的Python 3
def sift_kp(image):
    gray_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d_SIFT.create()
    kp,des = sift.detectAndCompute(image,None)
    kp_image = cv2.drawKeypoints(gray_image,kp,None)
    return kp_image,kp,des