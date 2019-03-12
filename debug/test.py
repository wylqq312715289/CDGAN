#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/11/20 上午10:48
# @Author  : Kris
# @Site    : 
# @File    : test
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

import nltk
from nltk.corpus import stopwords
# nltk.download('averaged_perceptron_tagger')
# text = nltk.word_tokenize("what does the fox say")
# print(text)
# print(nltk.pos_tag(text))
# print(stopwords.words("english"))
# print(len(stopwords.words("english")))
# print(len(stopwords.words("chinese")))

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data.dataset import Dataset
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn.functional as functional
import time, copy, os, cv2
import pandas as pd


# a = torch.rand(2,3,4)
# b = torch.rand(2,3,4)
# print(a.log())
# # exit()
# print(a)
# # print(a*b)
# indices = torch.LongTensor([0, 1])
# a = torch.index_select(a,1,indices)
# print(a)
#
# print( list(range(0,100,int(100/4))) )
#
# for i in range(100):
#     a =random.randint(0,1)
#     print(a)

# a = "/asdas/afasf/fasf"
#
# print(re.sub(r'^[\\/]',r"",a))

# 将json文件中的数据读取到dict
# def load_json(file_name):
#     with open(file_name) as json_file:
#         data = json.load(json_file)
#         return data
#
# a = load_json("../data/dataset-v6/dataset.json")
# a = load_json("../data/dataset-v6/train.json")
# a = load_json("../data/rscg-dataset/test.json")
# print( len(a.keys()) )

# pic1 = np.array([[[1,1,0],[1,1,0],[0,0,0]],[[0,1,1],[0,1,1],[0,0,0]]])
# pic2 = np.array([[[0,0,0],[0,1,1],[0,1,1]],[[0,0,0],[0,1,1],[0,1,1]]])
# print(pic1.shape,pic2.shape)
# # print(pic1,pic2)
# print(len(np.where((pic2[0] == 1) & ( pic1[0] == 1))[0]))
# print(len(np.where((pic2[0] == 1) | ( pic1[0] == 1))[0]))
# print(len(np.where((pic2[1] == 1) & ( pic1[1] == 1))[0]))
# print(len(np.where((pic2[1] == 1) | ( pic1[1] == 1))[0]))
#
# fun = lambda x, y: 1.0 * len(np.where((x == 1) & (y == 1))[0]) / max(1e-5, len(np.where((x == 1) | (y == 1))[0]))
# a = np.sum(list(map(fun, pic1, pic2)))
#
# print(a)

# a = "dataset-v6/imgs/201801130329_2006.jpg"
# b = re.sub(r'([/\\])(label\.jpg)', r'(0)', a)
# b = re.sub(r"label",r"pred",re.split(r"[\\/]",a)[-1])
# print(b)


a = 0.979793273
b = 0.930538513
print(2.0*a*b/(a+b))



















