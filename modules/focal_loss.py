# -*- coding:utf-8 -*-
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

class FocalLoss(nn.Module):
    def __init__(self, class_num=2, alpha=None, gamma=2, device_id=0, size_average=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num,1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.use_gpu = torch.cuda.is_available()  # 判断是否有GPU加速
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average
        self.device_id = device_id

    def forward(self, inputs, targets):
        # print(type(inputs.data),type(targets.data))
        # 针对语义分割问题 input为(-1,class_num,512,512) targets=(-1,512,512)
        # P = nn.Softmax2d()(inputs)

        input_size = tuple( inputs.size() )
        mask_size = (input_size[0],1,input_size[2],input_size[3])

        # 将target转换为(-1,class_num,512,512)
        class_mask = torch.zeros(input_size)
        class_mask = Variable(class_mask)
        # if self.use_gpu: class_mask = class_mask.cuda()
        if self.use_gpu:
            class_mask = class_mask.cuda(self.device_id)
        class_mask.scatter_(1, targets.data.view(mask_size), 1.) # 用1.去覆盖 class_mask 相当于one-hot


        probs = (inputs*class_mask).sum(1).view(-1,1)
        log_p = probs.log()
        batch_loss = -1.0 * torch.pow( (1.0-probs), self.gamma) * log_p

        if self.size_average: loss = batch_loss.mean()
        else: loss = batch_loss.sum()

        return loss

