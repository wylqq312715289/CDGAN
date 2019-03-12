#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/1/7 下午3:20
# @Author  : Kris
# @Site    : 
# @File    : leaf
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
import os, time
import torch.optim as optim
from torchvision import datasets, transforms
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data.dataset import Dataset
import torch.nn.functional as F
from torchvision import models
from torch.autograd import Variable
import time, copy, os, cv2
import random
import json
from torch.nn.parallel import data_parallel
from torch.autograd import grad

def f(x):
    """x^2 * e^x"""
    y = x ** 2 * torch.exp(x)
    return y


def gradf(x):
    """2*x*e^x + x^2*e^x"""
    dx = 2 * x * torch.exp(x) + x ** 2 * torch.exp(x)
    return dx


x = Variable(torch.randn(3, 4), requires_grad=True)
y = f(x)
y.backward(torch.ones(y.size()))
print(x.grad)
print(gradf(x))