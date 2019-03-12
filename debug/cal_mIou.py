#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/1/13 上午10:35
# @Author  : Kris
# @Site    : 
# @File    : cal_mIou
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

class IOUMetric:
    """ Class to calculate mean-iou using fast_hist method """
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.hist = np.zeros((num_classes, num_classes))
        self.sub_metric = dict()

    def _fast_hist(self, label_pred, label_true):
        mask = (label_true >= 0) & (label_true < self.num_classes)
        ###### 统计各个值出现的次数
        hist = np.bincount(
            self.num_classes * label_true[mask].astype(int) +
            label_pred[mask], minlength=self.num_classes ** 2).reshape(self.num_classes, self.num_classes)
        return hist

    def add_batch(self, predictions, gts):
        for lp, lt in zip(predictions, gts):
            self.hist += self._fast_hist(lp.flatten(), lt.flatten())

    def evaluate(self):
        acc = np.diag(self.hist).sum() / self.hist.sum()
        mean_acc = np.diag(self.hist) / self.hist.sum(axis=1)
        mean_recall = np.diag(self.hist) / self.hist.sum(axis=0)
        mean_f1 = 2.0*mean_acc*mean_recall/(mean_acc+mean_recall)

        mean_acc = np.nanmean(mean_acc)
        mean_recall = np.nanmean(mean_recall)
        mean_f1 = np.nanmean(mean_f1)

        iou = np.diag(self.hist) / (self.hist.sum(axis=1) + self.hist.sum(axis=0) - np.diag(self.hist))
        mean_iou = np.nanmean(iou)

        return acc, mean_acc, mean_recall, mean_f1, iou, mean_iou


m = IOUMetric(num_classes=2)
pic1 = np.array([[[1,1,0],[1,1,0],[0,0,0]],[[0,1,1],[0,1,1],[0,0,0]]])
pic2 = np.array([[[0,0,0],[0,1,1],[0,1,1]],[[0,0,0],[0,1,1],[0,1,1]]])
m.add_batch(pic1,pic2)
m.add_batch(pic1,pic2)
metric = m.evaluate()
print(metric)











