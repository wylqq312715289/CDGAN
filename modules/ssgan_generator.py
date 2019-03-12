#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/12/20 上午10:43
# @Author  : Kris
# @Site    : 
# @File    : ssgan_generator
# @Software: PyCharm
# @describe: 
# -*- coding:utf-8 -*-
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
import pandas as pd
import random
import json
import logging, sys



class SegmentationG(nn.Module):
    def __init__(self, input_size=(-1,256,7,7), output_size=(-1,6,224,224)):
        super(SegmentationG, self).__init__()

        self.relu = nn.ReLU(inplace=True)
        self.sigm = nn.Sigmoid()
        self.deconv1 = nn.ConvTranspose2d(input_size[1], 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.BatchNorm1 = nn.BatchNorm2d(512)

        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.BatchNorm2 = nn.BatchNorm2d(256)

        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.BatchNorm3 = nn.BatchNorm2d(128)

        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.BatchNorm4 = nn.BatchNorm2d(64)

        self.deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.BatchNorm5 = nn.BatchNorm2d(32)
        self.classifier = nn.Conv2d(32, output_size[1], kernel_size=1)

    def forward(self, x):
        out = self.relu(self.BatchNorm1(self.deconv1(x)))    # size=(N, 512, x.H/16, x.W/16)
        out = self.relu(self.BatchNorm2(self.deconv2(out)))  # size=(N, 256, x.H/8, x.W/8)
        out = self.relu(self.BatchNorm3(self.deconv3(out)))  # size=(N, 128, x.H/4, x.W/4)
        out = self.relu(self.BatchNorm4(self.deconv4(out)))  # size=(N, 64, x.H/2, x.W/2)
        out = self.relu(self.BatchNorm5(self.deconv5(out)))  # size=(N, 32, x.H, x.W)
        out = self.sigm(self.classifier(out))  # size=(N, n_class, x.H/1, x.W/1)
        return out


if __name__ == '__main__':
    input =  Variable(torch.randn(30,256,7,7))
    net = SegmentationG()
    out = net(input)
    print(out)