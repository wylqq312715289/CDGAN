#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/1/12 下午4:28
# @Author  : Kris
# @Site    : 
# @File    : debug_dataset1
# @Software: PyCharm
# @describe: 
import sys

# reload(sys)
# sys.setdefaultencoding('utf8')

import numpy as np
import pandas as pd
import time, os, json, math, re, random
import threading, logging, copy, scipy
from datetime import datetime, timedelta
import collections
import cv2
import matplotlib.pyplot as plt

imsize = 224

def resize_sample(img, label):
    new_img = cv2.resize(img, (imsize, imsize))
    new_label = cv2.resize(label, (imsize, imsize))
    result_label = np.zeros((imsize, imsize, 2))
    for i in range(imsize):
        for j in range(imsize):
            class_id = np.argmax(new_label[i, j, :])
            result_label[i, j, class_id] = 1
    assert np.sum(result_label) == (imsize * imsize), "sum(new_label) != %d * %d" % (imsize, imsize)
    return new_img, result_label

def img_random_transfer(img, label):
    """
    :param img: (512,512,6)
    :param label: (512,512,2)
    :return: img(224,224,6)  label(224,224,2)
    """
    new_img, new_label = img, label
    if random.random() < 0.5: # 随机水平或垂直翻转
        shift_id = random.randint(0,1)
        new_img = cv2.flip(new_img, shift_id)
        new_label = cv2.flip(new_label, shift_id)
    if random.random() < 0.5:  # 随机平移
        old_rows, old_cols = img.shape[:2]
        x_move1, x_move2 = (random.randint(0, 30), random.randint(0, 30))
        y_move1, y_move2 = (random.randint(0, 30), random.randint(0, 30))
        new_img = new_img[x_move1:old_rows-x_move2, y_move1:old_cols-y_move2]
        new_label = new_label[x_move1:old_rows-x_move2, y_move1:old_cols-y_move2]
    if random.random() < 0.5: # 随机旋转
        old_rows, old_cols = img.shape[:2]
        angle = random.randint(-180, 180)
        M = cv2.getRotationMatrix2D((old_cols / 2, old_rows / 2), angle, 1)# 最后一个参数为缩放因子
        new_img = cv2.warpAffine(img, M, (old_rows, old_cols))  # 需要图像、变换矩阵、变换后的大小
        new_label = cv2.warpAffine(label, M, (old_rows, old_cols))
        offset = int((2.0 - math.sqrt(2.0)) / 4.0 * old_rows)
        new_img = new_img[offset:-offset, offset:-offset]
        new_label = new_label[offset:-offset, offset:-offset]
    new_img, new_label = resize_sample(new_img, new_label)
    # print(new_img.shape, np.min(new_img), np.max(new_img))
    # print(new_label.shape,np.min(new_label),np.max(new_label))
    # exit()
    return new_img, new_label



img1_ = cv2.imread("./test_imgs/201801130176_2006.jpg")
img2_ = cv2.imread("./test_imgs/201801130176_2013.jpg")
label_ = cv2.imread("./test_imgs/201801130176_label.jpg",flags=cv2.IMREAD_UNCHANGED)
# label_ =
cont_ = np.concatenate((img1_,img2_,np.reshape(label_,(label_.shape[0],label_.shape[1],1))),axis=2)
label1 = np.where(label_ >= 0.5, 0, 1)
label2 = np.where(label_ >= 0.5, 1, 0)
new_label = np.array([label1, label2], dtype=np.int16).transpose([1, 2, 0]) # size = (512,512,2)

# print(img1.shape)
# print(img2.shape)
# print(label.shape)
# print(cont_.shape
a, b = img_random_transfer(img1_,new_label)
b = (np.argmax(b,axis=2) * 255).astype(int)
print(img1_.shape,b.shape)

plt.figure(1, figsize=(20, 20), dpi=90)

plt.subplot(2,3, 1)
plt.imshow(img1_)
plt.subplot(2, 3, 2)
plt.imshow(img2_)
plt.subplot(2, 3, 3)
plt.imshow(label_)
plt.subplot(2, 3, 4)
plt.imshow(a)
plt.subplot(2, 3, 5)
plt.imshow(b)
plt.show()