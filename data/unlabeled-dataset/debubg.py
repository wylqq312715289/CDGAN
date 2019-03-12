#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/12/20 下午4:39
# @Author  : Kris
# @Site    : 
# @File    : debubg
# @Software: PyCharm
# @describe: 
import sys

import numpy as np
import pandas as pd
import time, os, json, math, re, random
import threading, logging, copy, scipy
from datetime import datetime, timedelta
import collections

# 将dict数据保存到json
def store_json(file_name, data):
    with open(file_name, 'w') as json_file:
        json_file.write(json.dumps(data, indent=4, ensure_ascii=False))



imgs = os.listdir("./imgs/")
root_path = os.path.join("unlabeled-dataset","imgs")
data = dict()
for file_name in imgs:
    key = file_name.split("-")[0]
    year = file_name.split("-")[1].split(".")[0]
    if data.get(key) is None:
        data[key] = dict()
    data[key][year] = os.path.join(root_path,file_name)

print(len(imgs))
store_json("dataset.json", data)
