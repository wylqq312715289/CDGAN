#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/10/29 下午5:02
# @Author  : Kris
# @Site    : 
# @File    : config
# @Software: PyCharm
# @describe:

import sys
import numpy as np
import pandas as pd
import time, os, json, math, re, random
import threading, logging, copy, scipy
from datetime import datetime, timedelta
import collections
from easydict import EasyDict as edict

from .trainer import BasicGanTrainer, BasicNNTrainer, BasicSSGanTrainer

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if not os.path.exists(os.path.join(PROJECT_DIR, "data")): os.mkdir(os.path.join(PROJECT_DIR, "data"))
if not os.path.exists(os.path.join(PROJECT_DIR, "cache")): os.mkdir(os.path.join(PROJECT_DIR, "cache"))
if not os.path.exists(os.path.join(PROJECT_DIR, "example")): os.mkdir(os.path.join(PROJECT_DIR, "example"))

train_infos = (
    ("Unet", "CrossEntropyLoss", BasicNNTrainer),
    ("Unet", "FocalLoss", BasicNNTrainer),
    ("CDGAN", "FocalLoss", BasicGanTrainer),
    ("FCN8", "CrossEntropyLoss", BasicNNTrainer),
    ("FCN8", "FocalLoss", BasicNNTrainer),
    # ("SSGAN", "FocalLoss", BasicSSGanTrainer),
)



test_infos = (
    # ("Unet", "CrossEntropyLoss", BasicNNTrainer, "cache/2018_11_19_11_Unet_CrossEntropyLoss/unet-121.pkl"),
    ("Unet", "FocalLoss", BasicNNTrainer, "cache/2019_01_09_20_Unet_FocalLoss/model_parameters.pkl"),
    # ("CDGAN", "FocalLoss", BasicGanTrainer, "cache/2019_01_09_20_CDGAN_FocalLoss/generator-101.pkl"),
    # ("FCN8", "CrossEntropyLoss", BasicNNTrainer, "cache/2018_11_20_10_FCN8_CrossEntropyLoss/model_parameters.pkl"),
    # ("FCN8", "FocalLoss", BasicNNTrainer, "cache/2018_11_20_15_FCN8_FocalLoss/model_parameters.pkl"),
    # ("SSGAN", "FocalLoss", BasicSSGanTrainer),
)



__C = edict()
conf = __C

__C.DEBUG = False  # 是否DEBUG 整个程序, 小数据集测试
# Dataset name: flowers, birds
__C.PROJECT_DIR = PROJECT_DIR
__C.DATA_DIR = os.path.join(PROJECT_DIR, "data")
__C.SAMPLE_PATH = os.path.join(PROJECT_DIR, "example")
__C.dataset_info_file = os.path.join(PROJECT_DIR, "data", "dataset-v6", "dataset.json")
__C.unlabeled_dataset_info_file = os.path.join(PROJECT_DIR, "data", "unlabeled-dataset", "dataset.json")
__C.trainset_info_file = os.path.join(PROJECT_DIR, "data", "dataset-v6", "train.json")
__C.testset_info_file = os.path.join(PROJECT_DIR, "data", "dataset-v6", "test.json")

__C.IMSIZE = 224

__C.USE_GPU = True
__C.GPU_ID = "0"  # 使用哪一块GPU


__C.WORKERS = 32
__C.CLASS_NUM = 2

__C.TRAIN = edict()
__C.TRAIN.G_LR = 0.0002
__C.TRAIN.D_LR = 0.0002
__C.TRAIN.UNET_LR = 0.0002
__C.TRAIN.FCN_LR = 0.0002

__C.TRAIN.BATCH_SIZE = 4
__C.TRAIN.MAX_EPOCHS = 160


__C.TRAIN.EARLY_STOP = 30
__C.TRAIN.LR_DECAY_STEP = 30
