#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/1/3 上午10:06
# @Author  : Kris
# @Site    : 
# @File    : analysis
# @Software: PyCharm
# @describe: 
import sys

import numpy as np
import pandas as pd
import time, os, json, math, re, random
import threading, logging, copy, scipy
from datetime import datetime, timedelta
import collections
import cv2
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
font_set = FontProperties(fname=r"./data/msyh.ttf", size=15)


plt.rcParams['savefig.dpi'] = 600 #图片像素
plt.rcParams['figure.dpi'] = 600 #分辨率
plt.rcParams['image.interpolation'] = 'nearest'

# 将json文件中的数据读取到dict
def load_json(file_name):
    with open(file_name) as json_file:
        data = json.load(json_file)
        return data

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

out_plot = [
    "vali_mean_acc",
    "vali_mean_iou",
    "vali_mean_f1",
    "vali_classes_acc",
    "vali_classes_iou",
    "vali_classes_f1",
    "vali_mean_recall",
    "vali_classes_recall",
]

map_labely = {
    "vali_mean_acc":"Acc",
    "vali_mean_iou":"mIOU",
    "vali_mean_f1":"mF1",
    "vali_classes_acc":"P",
    "vali_classes_iou":"IOU",
    "vali_classes_f1":"F1",
    "vali_mean_recall":"mR",
    "vali_classes_recall":"R",

}

map_title = {
    "vali_mean_acc":"Acc值变化曲线",
    "vali_mean_iou":"mIOU值变化曲线",
    "vali_mean_f1":"mF1值变化曲线",
    "vali_classes_acc":"P值变化曲线",
    "vali_classes_iou":"IOU值变化曲线",
    "vali_classes_f1":"F1值变化曲线",
    "vali_mean_recall":"mR值变化曲线",
    "vali_classes_recall":"R值变化曲线",

}

def smooth(y):
    every_len = 5
    ans_x, ans_y = [], []
    y = list(y)
    for i in range(0, 200):
        if i + every_len >= len(y): break
        ans_x.append(i)
        ans_y.append(np.mean(y[i:i + every_len]))
    return ans_x, ans_y


# 最完美形式
# metric_source = [
#     ("FCN", "CE", "2019_01_19_16_FCN8_CrossEntropyLoss","red"),
#     ("FCN", "FC", "2019_01_17_16_FCN8_FocalLoss", "green"),
#     ("U-Net", "CE", "2019_01_13_12_Unet_CrossEntropyLoss","blue"),
#     ("U-Net", "FC", "2019_01_13_22_Unet_FocalLoss","orange"),
#     ("CDGAN", "FC", "2019_01_14_04_CDGAN_FocalLoss","black"),
# ]

metric_source = [
    ("FCN", "CE", "2019_03_04_14_FCN8_CrossEntropyLoss","red"),
    ("FCN", "FC", "2019_01_17_16_FCN8_FocalLoss", "green"),
    ("U-Net", "CE", "2019_03_03_17_Unet_CrossEntropyLoss","blue"),
    ("U-Net", "FC", "2019_03_03_23_Unet_FocalLoss","orange"),
    ("CDGAN", "FC", "2019_03_04_18_FCN8_FocalLoss","black"),
]

# 画模型预测曲线
def plot1():
    plt.figure(1, figsize=(14, 8), dpi=90)
    count = 1
    for k, plot_n in enumerate(out_plot,0):
        plt.subplot(330 + count)
        for i,item in enumerate(metric_source,0):
            data_path = item[2]
            data_path = os.path.join(PROJECT_DIR, "CDGAN/cache", data_path,"metric.json")
            data = load_json(data_path)
            data = sorted(data,key=lambda x: x.get("epoch"))[:150]
            iou = list(map(lambda x:x.get(plot_n),data))
            x, y = smooth(iou)
            # x = range(len(iou))
            plt.plot(x, y, c=item[3],label=item[0]+"("+item[1]+")")
            plt.xlabel('epoch')
            plt.ylabel(map_labely.get(plot_n))
            plt.title(map_title.get(plot_n), fontproperties=font_set)
            if k==0: plt.legend()
        count += 1
    plt.tight_layout()
    plt.savefig("./logs/models_line.jpg", dpi=90)
    plt.show()



# 画几个test sample
def plot2():
    # need_imgs = [658, 763, 771, 893]
    need_imgs = [771, 1431, 1460, 1894] # 771,1431,1460,1748
    plt.figure(1, figsize=(14, 8), dpi=90)
    count = 0
    for k, img_id in enumerate(need_imgs,0):
        for i, item in enumerate(metric_source, 0):
            data_path = item[2]
            data_path = os.path.join(PROJECT_DIR, "CDGAN/cache", data_path, "test_output")
            if i ==0:
                # print(data_path)
                img1 = os.path.join(data_path, "20180113%.4d_2006.jpg" % (img_id))
                img2 = os.path.join(data_path, "20180113%.4d_2013.jpg" % (img_id))
                gt   = os.path.join(data_path, "20180113%.4d_label.jpg" % (img_id))
                img1 = cv2.imread(img1)
                img2 = cv2.imread(img2)
                gt = cv2.imread(gt, flags=cv2.IMREAD_UNCHANGED)
                gt = cv2.cvtColor(gt, cv2.COLOR_GRAY2BGR)

                plt.subplot(len(need_imgs), len(metric_source) + 3, count + 1)
                plt.imshow(img1)
                if k==0: plt.title("2006")#.set_fontsize(fontsize=22)
                # plt.axis('off')  # 不显示坐标轴
                plt.xticks([])
                plt.yticks([])
                count += 1
                # plt.tight_layout()

                if count%(len(metric_source)+3) == 1:
                    plt.ylabel("样本{}".format(int(1.0*count/(len(metric_source)+3))+1), fontproperties=font_set)

                plt.subplot(len(need_imgs), len(metric_source) + 3, count + 1)
                plt.imshow(img2)
                if k==0: plt.title("2013")#.set_fontsize(fontsize=22)
                # plt.axis('off')  # 不显示坐标轴
                plt.xticks([])
                plt.yticks([])
                count += 1
                # plt.tight_layout()


                plt.subplot(len(need_imgs), len(metric_source) + 3, count + 1)
                plt.imshow(gt)
                if k==0: plt.title("GT")#.set_fontsize(fontsize=22)
                # plt.axis('off')  # 不显示坐标轴
                plt.xticks([])
                plt.yticks([])
                count += 1
                # plt.tight_layout()



            pred = os.path.join(data_path, "20180113%.4d_pred.jpg" % (img_id))
            pred = cv2.imread(pred, flags=cv2.IMREAD_UNCHANGED)
            pred = cv2.cvtColor(pred, cv2.COLOR_GRAY2BGR)
            plt.subplot(len(need_imgs), len(metric_source) + 3, count + 1)
            plt.imshow(pred)
            if k == 0: plt.title(item[0]+"("+item[1]+')')#.set_fontsize(fontsize=22)
            # plt.axis('off')  # 不显示坐标轴
            plt.xticks([])
            plt.yticks([])
            count += 1
            # plt.tight_layout()

    plt.tight_layout()
    plt.savefig("./logs/model_predict.jpg", dpi=90)
    plt.show()


if __name__ == '__main__':
    plot1()
    # plot2()