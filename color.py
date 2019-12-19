"""
@ File:     color.py
@ Author:   wzl
@ Datetime: 2019-12-19 14:42
"""
import cv2
import json
import matplotlib.pyplot as plt
import numpy as np

from labelme import utils

CANNY_LOWER_THRESHOLD = 20 # default 50
CANNY_HIGHER_THRESHOLD = 80 # default 150
GAUSSIAN_KERNEL_SIZE = 5 # default 3
DILATE_EDGE_KERNEL_SIZE = 11 # default 3

FILL = 1

def colored_mask(filename):
    """
    :param filename: image filename
    :return: colored mask
    """
    data = json.load(open(filename))

    img = utils.img_b64_to_arr(data['imageData'])
    lbl, lbl_names = utils.labelme_shapes_to_label(img.shape, data['shapes'])  # lbl：array 0、1 （区域内的为1，之外为0）
    captions = ['%d: %s' % (l, name) for l, name in enumerate(lbl_names)]
    lbl_viz = utils.draw_label(lbl, img, captions)

    mask = []
    class_id = []
    for i in range(1, len(lbl_names)):  # 若有多个class（物体） 跳过第一个class（默认为背景）
        mask.append((lbl == i).astype(np.uint8))  # 解析出像素值为1的对应，对应第一个对象 mask 为0、1组成的（0为背景，1为对象）
        class_id.append(i)  # mask与clas 一一对应

    mask = np.transpose(np.asarray(mask, np.uint8), [1, 2, 0])  # 转成[h,w,instance count]
    class_id = np.asarray(class_id, np.uint8)  # [instance count,]

    plt.subplot(221)
    plt.imshow(img)
    plt.title("original image")
    plt.axis('off')
    plt.subplot(222)
    plt.imshow(lbl_viz)
    plt.axis('off')
    plt.subplot(223)
    plt.imshow(mask[:, :, 0], 'gray')

    mask2 = np.ones_like(img)
    for i in range(mask2.shape[2]):
        mask2[:, :, i] = mask.squeeze()
    plt.subplot(224)
    plt.imshow(mask2 * img)
    plt.axis('off')
    plt.show()

colored_mask('query/back/green.json')