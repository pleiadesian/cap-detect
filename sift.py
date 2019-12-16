"""
@ File:     sift.py
@ Author:   wzl
@ Datetime: 2019-12-15 22:30
"""
import numpy as np
import cv2
from matplotlib import pyplot as plt

MIN_MATCH_COUNT = 10

img1 = cv2.imread('car.png',0)          # queryImage
img2 = cv2.imread('findcar.png',0)      # trainImage

# Initiate SIFT detector
orb = cv2.ORB_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = orb.detectAndCompute(img1,None)
kp2, des2 = orb.detectAndCompute(img2,None)

FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)

flann = cv2.FlannBasedMatcher(index_params, search_params)

matches = flann.knnMatch(np.asarray(des1,np.float32),np.asarray(des2,np.float32),k=2)

# store all the good matches as per Lowe's ratio test.
good = []
for m,n in matches:
    if m.distance < 0.7*n.distance:
        good.append(m)

if len(good)>MIN_MATCH_COUNT:
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
    matchesMask = mask.ravel().tolist()

    h,w = img1.shape
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts,M)

    img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)

else:
    print("Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT))
    matchesMask = None

draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                   singlePointColor=None,
                   matchesMask=matchesMask,  # draw only inliers
                   flags=2)

img3 = cv2.drawMatches(img1, kp1, img2, kp2, good, None, **draw_params)

plt.imshow(img3, 'gray'), plt.show()






# # create BFMatcher object
# bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
#
# # Match descriptors.
# matches = bf.match(des1,des2)
#
# # Sort them in the order of their distance.
# matches = sorted(matches, key = lambda x:x.distance)
#
# # Draw first 10 matches.
# img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:10], None, flags=2)
#
# plt.imshow(img3),plt.show()





# import cv2
# import numpy as np
#
#
# def sift_kp():
#     img_gray = cv2.imread('lena.png', cv2.IMREAD_GRAYSCALE)
#     # image = image.astype(np.uint8)
#     # gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#
#     sift = cv2.xfeatures2d_SIFT.create()
#     kp, des = sift.detectAndCompute(img_gray, None)
#     kp_image = cv2.drawKeypoints(gray_image, kp, None)
#
#     # sift = cv2.xfeatures2d.SURF_create()
#     # keypoints = sift.detect(img_gray, None)
#     # img_sift = np.copy(img_gray)
#
#     cv2.drawKeypoints(img_gray, keypoints, img_sift, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
#     cv2.imshow('input_image', img_gray)
#     cv2.imshow('img_sift', img_sift)
#     cv2.waitKey(0)
#
#
# sift_kp()

