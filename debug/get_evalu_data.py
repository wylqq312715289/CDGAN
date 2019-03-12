#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/3/1 下午5:50
# @Author  : Kris
# @Site    : 
# @File    : get_evalu_data
# @Software: PyCharm
# @describe: 
import sys
import numpy as np
import pandas as pd
import time, os, json, math, re, random
import threading, logging, copy, scipy
from datetime import datetime, timedelta
import collections

# 将json文件中的数据读取到dict
def load_json(file_name):
    with open(file_name) as json_file:
        data = json.load(json_file)
        return data

# 将dict数据保存到json
def store_json(file_name, data):
    with open(file_name, 'w') as json_file:
        json_file.write(json.dumps(data, indent=4, ensure_ascii=False))


PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


metric_source = [
    ("FCN", "CE", "2019_01_19_16_FCN8_CrossEntropyLoss","red"),
    ("FCN", "FC", "2019_01_17_16_FCN8_FocalLoss", "green"),
    ("U-Net", "CE", "2019_01_13_12_Unet_CrossEntropyLoss","blue"),
    ("U-Net", "FC", "2019_01_13_22_Unet_FocalLoss","orange"),
    ("CDGAN", "FC", "2019_01_14_04_CDGAN_FocalLoss","black"),
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

def plot1():
    count = 1
    new_data = dict()
    for i,item in enumerate(metric_source,0):
        data_path = item[2]
        data_path = os.path.join(PROJECT_DIR, "CDGAN/cache", data_path,"metric.json")
        data = load_json(data_path)
        data = sorted(data,key=lambda x: x.get("epoch"))[-1]
        id = "{}({})".format(item[0],item[1])
        new_data[id] = dict()
        for key in map_labely.keys():
            new_data[id][map_labely.get(key)] =  "%.4lf"%data.get(key)

    store_json("test.json",new_data)
    sub_keys = [ "Acc", "mIOU", "mF1", "P", "IOU", "F1", "mR", "R" ]
    for sub_key in sub_keys:
        sum = sub_key
        for key in new_data.keys():
            sum = sum + " & %s"%(new_data.get(key).get(sub_key))
        print(sum + "  \\\\")
    # print(new_data)


if __name__ == '__main__':
    plot1()