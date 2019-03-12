#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/12/21 上午10:35
# @Author  : Kris
# @Site    : 
# @File    : unlabeled_GAN_loss
# @Software: PyCharm
# @describe: 
import sys
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

class Semi_Supervised_Unlabeled_Loss(nn.Module):
    def __init__(self, class_num=2+1, size_average=True):
        super(Semi_Supervised_Unlabeled_Loss, self).__init__()
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs):
        # inputs size = (-1,3,w,h) 并且已经经过softmax layer
        input_size = tuple(inputs.size())
        mask_size = (input_size[0], 1, input_size[2], input_size[3])

        # 将target转换为(-1,3,w,h)
        class_mask = np.zeros(input_size)
        class_mask[:, -1, :, :] = 1.0
        class_mask = Variable(torch.FloatTensor(class_mask))
        if torch.cuda.is_available():
            class_mask = class_mask.cuda()
        probs = (1.0 - (inputs * class_mask).sum(1)).view(-1,1) # 不是第三类的概率
        batch_loss = -1.0 * probs.log()
        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss

if __name__ == '__main__':
    fake_y = (np.ones((8, 512, 512)) * 2).astype(np.int)
    print(fake_y.shape)



    # input = Variable(torch.randn(11,3,46,46))
    #
    #
    # net = Semi_Supervised_Unlabeled_Loss()
    # out = net(input)
    # print(out)


