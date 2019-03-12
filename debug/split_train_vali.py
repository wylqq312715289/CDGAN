#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/11/17 上午10:24
# @Author  : Kris
# @Site    : 
# @File    : split_train_vali
# @Software: PyCharm
# @describe: 
import sys

import numpy as np
import pandas as pd
import time, os, json, math, re, random
import threading, logging, copy, scipy
from datetime import datetime, timedelta
import collections

def load_json(file_name):
    with open(file_name) as json_file:
        data = json.load(json_file)
        return data

def store_json(file_name, data):
    with open(file_name, 'w') as json_file:
        json_file.write(json.dumps(data, indent=4, ensure_ascii=False))


dataset_name = "dataset-v6"

all = load_json("../data/%s/dataset.json"%dataset_name)
keys = all.keys()
# print(len(keys))
# exit();
keys = sorted(keys)
train_keys = keys[:311*3+1]
test_keys  = keys[311*3+1:]
train_json = dict()
test_json = dict()



[ train_json.update({k:all.get(k)}) for k in train_keys]
[ test_json.update({k:all.get(k)}) for k in test_keys]
store_json("../data/%s/train.json"%dataset_name,train_json)
store_json("../data/%s/test.json"%dataset_name,test_json)



print(len(train_keys),len(test_keys))