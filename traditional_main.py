#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/11/17 上午10:04
# @Author  : Kris
# @Site    : 
# @File    : traditional_main
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

from tqdm import tqdm
from modules.config import conf
from modules.utils import load_json


def get_pixcel_set(img_np):
    a = []
    for i in range(len(img_np)):
        for j in range(len(img_np[0])):
            a.append(img_np[i, j])
    a = collections.Counter(a)
    a = sorted(a.most_common(len(a)), key=lambda x: x[0])
    print(a)


def get_sub_metric(pred_y, real_y, flag=0):
    """
    :param pred_y:  numpy size = (224,224) value = 0 or 1
    :param real_y:  numpy size = (224,224) value = 0 or 1
    :param target:  numpy size = (batchsize,224,224)
    :param flag:
    :return:
    """
    c, w, h = pred_y.shape
    sub_metric = dict()
    pred_y = pred_y.astype(np.int)
    real_y = real_y.astype(np.int)

    fun = lambda x, y: 1.0 * len(np.where((x == 1) & (y == 1))[0]) / max(1e-5, len(np.where((x == 1) | (y == 1))[0]))
    sub_metric["iou"] = np.sum(list(map(fun, pred_y, real_y)))

    fun = lambda x, y: 1.0 * len(np.where(x == y)[0]) / (w * h)
    sub_metric["pixel_acc"] = np.sum(list(map(fun, pred_y, real_y)))

    fun = lambda x, y: 1.0 * len(np.where((x == 1) & (y == 1))[0]) / max(1e-5, len(np.where(x == 1)[0]))
    sub_metric["mean_acc"] = np.sum(list(map(fun, pred_y, real_y)))
    fun = lambda x, y: 1.0 * len(np.where((x == 0) & (y == 0))[0]) / max(1e-5, len(np.where(x == 0)[0]))
    sub_metric["mean_acc"] += np.sum(list(map(fun, pred_y, real_y)))
    sub_metric["mean_acc"] /= 2

    # 正确率
    fun = lambda x, y: 1.0 * len(np.where((x == 1) & (y == 1))[0]) / max(1e-5, len(np.where(x == 1)[0]))
    sub_metric["acc_detection"] = np.sum(list(map(fun, pred_y, real_y)))

    # 召回率
    fun = lambda x, y: 1.0 * len(np.where((x == 1) & (y == 1))[0]) / max(1e-5, len(np.where(y == 1)[0]))
    sub_metric["recall_detection"] = np.sum(list(map(fun, pred_y, real_y)))

    # 误检率
    fun = lambda x, y: 1.0 * (len(np.where(x == 1)[0]) - len(np.where((x == 1) & (y == 1))[0])) / max(1e-5, len(
        np.where(x == 1)[0]))
    sub_metric["error_detection"] = np.sum(list(map(fun, pred_y, real_y)))

    # 漏检率
    fun = lambda x, y: 1.0 * (len(np.where(y == 1)[0]) - len(np.where((x == 1) & (y == 1))[0])) / max(1e-5, len(
        np.where(y == 1)[0]))
    sub_metric["omit_detection"] = np.sum(list(map(fun, pred_y, real_y)))

    # F1-score
    fun = lambda x, y: 2.0 * len(np.where((x == 1) & (y == 1))[0]) / max(1e-5, len(np.where(x == 1)[0]) + len(
        np.where(y == 1)[0]))

    sub_metric["f1_score"] = np.sum(list(map(fun, pred_y, real_y)))

    return sub_metric



def Img_PCA(delta):
    U, S, V = np.linalg.svd(delta)
    SS = np.zeros(U.shape)
    for i in range(S.shape[0]):
        SS[i][i] = S[i]

    def Pick_k(S):
        sval = np.sum(S)
        for i in range(S.shape[0]):
            if np.sum(S[:i]) >= 0.6 * sval:
                break
        return i + 1

    k = Pick_k(S)
    # print("\nNumber of vectors to reserve: k= ", k)
    Uk = U[:, 0:k]
    Sk = SS[0:k, 0:k]
    Vk = V[0:k, :]
    im = np.dot(np.dot(Uk, Sk), Vk)
    # im=np.dot(im,delta)
    return im


def Find_Threshold(delta):  # OTSU寻找阈值
    val = np.zeros([256])
    print(val.shape)
    for th in range(256):
        loc1 = delta > th
        loc2 = delta <= th
        '''
        delta[loc1] = 255
        delta[loc2] = 0
        '''
        if delta[loc1].size == 0:
            mu1 = 0
            omega1 = 0
        else:
            mu1 = np.mean(delta[loc1])
            omega1 = delta[loc1].size / delta.size

        if delta[loc2].size == 0:
            mu2 = 0
            omega2 = 0
        else:
            mu2 = np.mean(delta[loc2])
            omega2 = delta[loc2].size / delta.size
        val[th] = omega1 * omega2 * np.power((mu1 - mu2), 2)

    # print("val=",val.shape)
    # plt.figure()
    loc = np.where(val == np.max(val))
    # x=np.arange(0,256,1)
    # x=x.reshape([1,256])
    # plt.plot(val)
    # plt.ylabel("Var")
    # plt.xlabel("Threshold")
    # plt.grid("on")
    # plt.show()
    # exit()
    # print("The best OTSU Threshold: ", loc[0])
    return loc[0]


def CD_diff(img1, img2):  # 影像差值法
    delta = np.subtract(img2, img1)
    # delta = np.abs(delta)
    w, h = delta.shape

    # print(delta.min())

    delta += np.abs(delta.min())

    th = Find_Threshold(delta)
    # print(th, np.size(th))
    # exit();

    if np.size(th) > 1:
        th = th[0]

    for i in range(w):
        for j in range(h):
            if delta[i][j] >= th:
                delta[i][j] = 255
            else:
                delta[i][j] = 0
    return delta



def get_change(imgA_np, imgB_np):
    w, h = imgB_np.shape
    result_np = np.zeros((w, h))
    abs = np.abs(imgB_np / imgA_np)
    thread = 0.7
    for i in range(w):
        for j in range(h):
            if abs[i, j] >= 1.0 + thread or abs[i, j] < 1.0 - thread:
                result_np[i, j] = 255
    result_np = result_np.astype(np.uint8)
    return result_np


def get_change_1(grayA, grayB):
    w, h = grayB.shape
    result_window = np.zeros((w, h), dtype=grayB.dtype)
    w_size = 100
    for start_x in range(0, w - w_size, 50):
        for start_y in range(0, h - w_size, 50):
            need_match_window = grayA[start_x:start_x + w_size, start_y:start_y + w_size]
            query_thread = grayB[max(0, (start_x - w_size)):start_x + 2 * w_size,
                           max(0, start_y - w_size):start_y + 2 * w_size]
            match = cv2.matchTemplate(query_thread, need_match_window, cv2.TM_CCOEFF_NORMED)
            _, _, _, max_loc = cv2.minMaxLoc(match)
            matched_window = grayB[max_loc[1]:max_loc[1] + 100, max_loc[0]:max_loc[0] + 100]
            result = cv2.absdiff(need_match_window, matched_window)
            result_window[start_y:start_y + 100, start_x:start_x + 100] = result
    return result_window


def get_change_2(grayA, grayB):
    w, h = grayB.shape
    result_window = np.zeros((w, h), dtype=grayB.dtype)
    w_size = 4
    thread = 3.0
    meanA = np.mean(grayA)
    meanB = np.mean(grayB)
    for start_x in range(0, w, w_size):
        for start_y in range(0, h, w_size):
            winA = grayA[start_x:start_x + w_size, start_y:start_y + w_size]
            winB = grayB[start_x:start_x + w_size, start_y:start_y + w_size]
            # print( winA, winB )
            # abs = np.mean(np.abs(winA-winB))
            abs = np.mean((winA - meanA) / (winB - meanB))
            if abs <= 1.0 - thread or abs > 1.0 + thread:
                result_window[start_x:start_x + w_size, start_y:start_y + w_size] = 255
    return result_window


def get_change_pca(grayA, grayB):
    return CD_diff(Img_PCA(grayA),Img_PCA(grayB))


def filter_img(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (9, 9), 0)  # 高斯滤波
    # gradX = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=1, dy=0)
    # gradY = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=0, dy=1)
    # gradient = cv2.subtract(gradX, gradY)
    # gradient = cv2.convertScaleAbs(gradient)
    # blurred = cv2.GaussianBlur(gradient, (9, 9), 0)
    # (_, thresh) = cv2.threshold(blurred, 90, 255, cv2.THRESH_BINARY)
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    # closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    # plt.imshow(closed)
    # plt.show()
    return gray


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
    test_json = load_json(conf.testset_info_file)
    keys = list(test_json.keys())
    metric = dict()
    columns = ["pixel_acc", "mean_acc", "iou", "loss", "acc_detection", "recall_detection",
               "error_detection", "omit_detection", "f1_score", "d_loss", "g_loss"]
    for column in columns: metric["trad_test_{}".format(column)] = 0.0
    for key in tqdm(list(keys)):
        img_2006_filename = test_json.get(key).get("2006")
        img_2013_filename = test_json.get(key).get("2013")
        label = test_json.get(key).get("label")
        assert img_2006_filename and img_2013_filename and label
        imgA = read_img(os.path.join(conf.DATA_DIR, img_2006_filename))
        imgB = read_img(os.path.join(conf.DATA_DIR, img_2013_filename))
        imgC = read_img(os.path.join(conf.DATA_DIR, label), flags=cv2.IMREAD_UNCHANGED)

        grayA = filter_img(imgA)
        grayB = filter_img(imgB)

        img_res = get_change_pca(grayA, grayB)
        img_res = np.where(img_res >= 128, 1, 0).reshape((1, 224, 224))
        imgC = np.where(imgC >= 128, 1, 0).reshape((1, 224, 224))

        sub_metric = get_sub_metric(img_res, imgC)
        # print( sub_metric, imgC.shape, img_res.shape)

        for column in columns:
            metric["trad_test_{}".format(column)] += sub_metric.get(column, 0.0)

    for column in columns:
        if column == "loss":
            metric["trad_test_{}".format(column)] /= (len(keys) * 256 * 256)
        else:
            metric["trad_test_{}".format(column)] /= len(keys)

    print(metric)
