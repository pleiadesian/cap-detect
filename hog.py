"""
@ File:     hog.py
@ Author:   wzl
@ Datetime: 2019-12-17 15:23
"""
import numpy as np
import cv2

import matplotlib.pyplot as plt

from skimage.feature import hog
from skimage import data, exposure
from sklearn.decomposition import PCA


# image = data.astronaut()

img_front = cv2.imread('query/front-hog.png')
img_back = cv2.imread('query/back-hog.png')
img_side = cv2.imread('query/side-hog.png')
img_train = cv2.imread('train/front-hog.png')
img_front = cv2.resize(img_front, (512, 512))
img_back = cv2.resize(img_back, (512, 512))
img_side = cv2.resize(img_side, (512, 512))
img_train = cv2.resize(img_train, (512, 512))
# img_test = cv2.imread('train/2.png')

# fd, hog_image = hog(img, orientations=8, pixels_per_cell=(16, 16),
#                     cells_per_block=(1, 1), visualize=True, multichannel=True)
fd_front, hog_front = hog(img_front, orientations=8, pixels_per_cell=(4, 4),
                    cells_per_block=(1, 1), visualize=True, multichannel=True)
fd_back, hog_back = hog(img_back, orientations=8, pixels_per_cell=(4, 4),
                    cells_per_block=(1, 1), visualize=True, multichannel=True)
fd_side, hog_side = hog(img_side, orientations=8, pixels_per_cell=(4, 4),
                    cells_per_block=(1, 1), visualize=True, multichannel=True)
fd_train, hog_train = hog(img_train, orientations=8, pixels_per_cell=(4, 4),
                    cells_per_block=(1, 1), visualize=True, multichannel=True)

'''
L2 distance
'''
op_front=np.linalg.norm(fd_front-fd_train)
op_back=np.linalg.norm(fd_back-fd_train)
op_side = np.linalg.norm(fd_side-fd_train)
softmax_sum = np.exp(op_front)+np.exp(op_back)+np.exp(op_side)
softmax_front = np.exp(op_front)/softmax_sum
softmax_back = np.exp(op_back)/softmax_sum
softmax_side = np.exp(op_side)/softmax_sum
a=1


# '''
# dimension reduction
# '''
# pca = PCA(n_components = 2)




'''
display hog image
'''
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)
#
# ax1.axis('off')
# ax1.imshow(img, cmap=plt.cm.gray)
# ax1.set_title('Input image')
#
# # Rescale histogram for better display
# hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
#
# ax2.axis('off')
# ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
# ax2.set_title('Histogram of Oriented Gradients')
# plt.show()






# #在这里设置参数
# winSize = (128,128)
# blockSize = (64,64)
# blockStride = (8,8)
# cellSize = (16,16)
# nbins = 9
#
# #定义对象hog，同时输入定义的参数，剩下的默认即可
# hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins)
#
# winStride = (8,8)
# padding = (8,8)
# des1 = hog.compute(img, winStride, padding).reshape((-1,))
# des2 = hog.compute(img_test, winStride, padding).reshape((-1,))







# # create BFMatcher object
# bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
#
# # Match descriptors.
# matches = bf.match(np.asarray(des1, np.float32),np.asarray(des2, np.float32))
#
# # Sort them in the order of their distance.
# matches = sorted(matches, key = lambda x:x.distance)
# a = matches





# Draw first 10 matches.
# img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:10], flags=2)

# plt.imshow(img3),plt.show()

# # use FLANN matcher
# FLANN_INDEX_KDTREE = 0
# index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
# search_params = dict(checks = 50)
# flann = cv2.FlannBasedMatcher(index_params, search_params)
#
# matches = flann.knnMatch(np.asarray(hog_img, np.float32), np.asarray(hog_imgtest, np.float32), k=2)
