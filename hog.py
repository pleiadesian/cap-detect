"""
@ File:     hog.py
@ Author:   wzl
@ Datetime: 2019-12-17 15:23
"""
import os
import numpy as np
import cv2

from skimage.feature import hog

FRONT = 0
BACK = 1
SIDE = 2
NONE = 3

direct_lower_str = ['front', 'back', 'side']

def hog_des(img_query):
    """
    :param img_query: query image lists
    :return: HOG descriptor lists
    """
    # evaluate HOG descriptor
    fd = [[], [], []]
    for direct in range(FRONT, NONE):
        for query in img_query[direct]:
            query_temp = cv2.resize(query, (512, 512))
            fd_query, hog_query = hog(query_temp, orientations=8, pixels_per_cell=(4, 4),
                                      cells_per_block=(1, 1), visualize=True, multichannel=True)
            fd[direct].append(fd_query)
    save_fd(fd)
    return fd

def save_fd(fd):
    """
    :param fd: HOG descriptor lists
    """
    for direct in range(FRONT, NONE):
        i = 0
        for fd_array in fd[direct]:
            np.savetxt('./hog/'+direct_lower_str[direct]+str(i)+'.txt', fd_array, delimiter = ',')
            i += 1

def load_fd():
    """
    :return: HOG descriptor lists
    """
    fd = [[], [], []]
    for direct in range(FRONT, NONE):
        i = 0
        filename = './hog/'+direct_lower_str[direct]+str(i)+'.txt'
        while os.path.exists(filename):
            fd_array = np.loadtxt(filename,dtype=np.float64)
            fd[direct].append(fd_array)
            i += 1
            filename = './hog/' + direct_lower_str[direct] + str(i) + '.txt'
    return fd

def hog_match(fd, img_query, img_train):
    """
    :param fd: HOG descriptor lists
    :param img_query: query image lists
    :param img_train: target image
    :return: image type, image matched, softmax list
    """
    train_temp = cv2.resize(img_train, (512, 512))
    fd_train, hog_train = hog(train_temp, orientations=8, pixels_per_cell=(4, 4),
                              cells_per_block=(1, 1), visualize=True, multichannel=True)
    softmax = [[], [], []]
    img_selected = [None, None, None]
    for direct in range(FRONT, NONE):
        op_min = None
        for fd_query, query in zip(fd[direct], img_query[direct]):
            op = np.linalg.norm(fd_query-fd_train)
            if op_min is None or op < op_min:
                op_min = op
                img_selected[direct] = query
            softmax[direct].append(np.exp(op))
    softsum = np.sum(softmax[FRONT])+np.sum(softmax[BACK])+np.sum(softmax[SIDE])
    for direct in range(FRONT, NONE):
        softmax[direct] = softmax[direct] / softsum
    img_type = NONE
    min_softmax = 1
    img_final_selected = None
    for direct in range(FRONT, NONE):
        if min(softmax[direct]) < min_softmax:
            min_softmax =min(softmax[direct])
            img_type = direct
            img_final_selected = img_selected[direct]
    return img_type, img_final_selected, softmax

