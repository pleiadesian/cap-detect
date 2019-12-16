"""
@ File:     sift.py
@ Author:   wzl
@ Datetime: 2019-12-15 22:30
"""
import os
import numpy as np
import cv2
from matplotlib import pyplot as plt

# match threshold
MIN_MATCH_COUNT = 10

FRONT = 0
BACK = 1
SIDE = 2
NONE = 3
direct_str = ['FRONT','BACK','SIDE']

# load query images
img = [[], [], []]
img[FRONT].append(cv2.imread('query/front.png',0))
img[FRONT].append(cv2.imread('query/front-1.png', 0))
img[BACK].append(cv2.imread('query/back.png',0))
img[SIDE].append(cv2.imread('query/side.png',0))
img[SIDE].append(cv2.imread('query/side-1.png', 0))

# Initiate SIFT detector
orb = cv2.ORB_create()

# find the keypoints and descriptors with SIFT
kp = [[], [], []]
des = [[], [], []]
for direct in range(FRONT, NONE):
    for img_temp in img[direct]:
        kp_temp, des_temp = orb.detectAndCompute(img_temp, None)
        kp[direct].append(kp_temp)
        des[direct].append(des_temp)

# use FLANN matcher
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)
flann = cv2.FlannBasedMatcher(index_params, search_params)

# for i in range(0, TRAIN_SIZE):
for base_path, folder_list, file_list in os.walk('train'):
    for file_name in file_list:
        # img_train = cv2.imread('train/'+str(i)+'.png',0)      # trainImage
        filename = os.path.join(base_path,file_name)
        img_train = cv2.imread(filename, 0)
        img_train_matched = None

        kp_train, des_train = orb.detectAndCompute(img_train,None)

        matches = [[], [], []]
        for direct in range(FRONT, NONE):
            for des_temp in des[direct]:
                matches[direct].append(
                    flann.knnMatch(np.asarray(des_temp, np.float32), np.asarray(des_train, np.float32), k=2))

        # store all the good matches as per Lowe's ratio test.
        selected = NONE
        max_match = 0
        matchesMask = None
        img_selected = None
        kp_selected = None
        good_selected = None
        for direct in range(FRONT, NONE):
            for img_temp, kp_temp, des_temp, matches_temp in zip(img[direct], kp[direct], des[direct], matches[direct]):
                good = []
                for m,n in matches_temp:
                    if m.distance < 0.7*n.distance:
                        good.append(m)
                if len(good) > MIN_MATCH_COUNT:
                    src_pts = np.float32([ kp_temp[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
                    dst_pts = np.float32([ kp_train[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

                    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
                    matchesMask = mask.ravel().tolist()

                    # no matches is inlier, skip this train image
                    if M is None :
                        continue

                    match_sum = np.sum(matchesMask)
                    if match_sum > max_match:
                        max_match = match_sum
                        selected = direct

                        h,w = img_temp.shape
                        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
                        dst = cv2.perspectiveTransform(pts,M)

                        img_selected = img_temp
                        kp_selected = kp_temp
                        good_selected = good
                        img_train_matched = cv2.polylines(img_train,[np.int32(dst)],True,255,3, cv2.LINE_AA)

        if selected == NONE:
            print("Not enough matches are found")
            matchesMask = None
        else:
            print("%s is %s" % (filename, direct_str[selected]))
            draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                               singlePointColor=None,
                               matchesMask=matchesMask,  # draw only inliers
                               flags=2)
            img_output = cv2.drawMatches(img_selected, kp_selected, img_train_matched, kp_train, good_selected, None, **draw_params)
            plt.imshow(img_output, 'gray'), plt.show()
