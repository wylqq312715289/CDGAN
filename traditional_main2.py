#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/1/7 上午11:06
# @Author  : Kris
# @Site    : 
# @File    : traditional_main2
# @Software: PyCharm
# @describe: 
import sys

import numpy as np
import pandas as pd
import time, os, json, math, re, random
import threading, logging, copy, scipy
from datetime import datetime, timedelta
import collections
from tqdm import tqdm
import cv2
from osgeo import gdal_array

from modules.config import conf
from modules.utils import load_json
from modules.trainer import IOUMetric


map_labely = {
    "mean_acc":"Acc",
    "mean_iou":"mIOU",
    "mean_f1":"mF1",
    "classes_acc":"P",
    "classes_iou":"IOU",
    "classes_f1":"F1",
    "mean_recall":"mR",
    "classes_recall":"R",
}


def get_change(front_gray, behind_gray):
    front_gray = np.where(front_gray >= 128, 1, 0)
    behind_gray = np.where(behind_gray >= 128, 1, 0)
    result = np.where((behind_gray==1) & (front_gray==0), 1, 0)
    return result  # size=(224,224)


def solo_img_segmentation(jpg_file_path):
    ############### step1 读取图像并处理  ####################
    imgA = read_img(jpg_file_path)
    srcArr = imgA
    srcArr = cv2.GaussianBlur(srcArr, (9, 9), 0)  # 高斯滤波
    srcArr = cv2.cvtColor(srcArr, cv2.COLOR_BGR2GRAY)  #转为灰度图
    # BGR = cv2.cvtColor(srcArr, cv2.COLOR_RGB2BGR)
    # srcArr = cv2.pyrMeanShiftFiltering(srcArr, 100, 40)  # 高斯滤波
    # print(srcArr.shape)  # 3*512*512

    ############### step2 读取图像并处理  ####################
    # 根据类别数目将直方图分割成3个颜色区间
    classes = gdal_array.numpy.histogram(srcArr, bins=2)[1]
    # print(classes)
    # 颜色查找表的记录数必须是len(classes)+1，声明RGN元组
    lut = [[255, 0, 0], [0, 0, 0], [255, 255, 255]]
    # 分类初始值
    start = 1
    # 创建一个RGB颜色的JPEG图片输出
    rgb = gdal_array.numpy.zeros((3, srcArr.shape[0], srcArr.shape[1]), gdal_array.numpy.float32)
    # 处理所有类并声明颜色
    for i in range(len(classes)):
        mask = gdal_array.numpy.logical_and(start <= srcArr, srcArr <= classes[i])
        for j in range(len(lut[i])):
            rgb[j] = gdal_array.numpy.choose(mask, (rgb[j], lut[i][j]))
        start = classes[i] + 1
    # 保存图片
    rgb = rgb.astype(gdal_array.numpy.uint8)
    return imgA,np.transpose(rgb,(1,2,0))



def change_detection(jpg_file1_path,jpg_file2_path):
    original_img1, rgb1 = solo_img_segmentation(jpg_file1_path)
    original_img2, rgb2 = solo_img_segmentation(jpg_file2_path)
    gray1 = cv2.cvtColor(rgb1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(rgb2, cv2.COLOR_BGR2GRAY)
    # print(gray1)
    delta = get_change(gray1, gray2)
    return delta


def read_img(abs_file_path, flags=None):
    if flags:
        img = cv2.imread(abs_file_path, flags=flags)
    else:
        img = cv2.imread(abs_file_path)
    try:
        img = cv2.resize(img, (224, 224))
    except:
        print(abs_file_path, img, flags)
        exit()
    return img


if __name__ == '__main__':
    iou_metric = IOUMetric(num_classes=2)
    test_json = load_json(conf.testset_info_file)
    keys = list(test_json.keys())
    for key in tqdm(list(keys)):

        img_2006_filename = test_json.get(key).get("2006")
        img_2013_filename = test_json.get(key).get("2013")
        label = test_json.get(key).get("label")

        assert img_2006_filename and img_2013_filename and label
        front_img = os.path.join(conf.DATA_DIR, img_2006_filename)  # 前期图像
        behind_img = os.path.join(conf.DATA_DIR, img_2013_filename)  # 后期图像
        GT = read_img(os.path.join(conf.DATA_DIR, label), flags=cv2.IMREAD_UNCHANGED) # 人工标注
        img_res = change_detection(front_img, behind_img)  # 变化检测结果 (224, 224)

        img_res = img_res.reshape((1, 224, 224))
        GT = np.where(GT >= 0.5, 1, 0).reshape((1, 224, 224))
        # print(img_res, np.min(img_res), np.max(img_res))
        # print(GT,np.min(GT),np.max(GT))
        # exit()

        iou_metric.add_batch(img_res, GT)
        # print( sub_metric, imgC.shape, img_res.shape)
    print(iou_metric.hist)
    metric = iou_metric.evaluate()
    print(metric)
    for key in map_labely.keys():
        v = metric.get(key)
        print("{} {}".format(map_labely.get(key),v))


