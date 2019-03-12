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
from torch.nn.parallel import data_parallel
from torch.autograd import grad


class Discriminator(nn.Module):
    # initializers
    def __init__(self, in_channels, d=32):
        super(Discriminator, self).__init__()
        self.d = d
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, d, kernel_size=3, padding=1),
            # nn.BatchNorm2d(d),
            nn.LeakyReLU(0.2),
            # nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(d, 2 * d, kernel_size=3, padding=1),
            nn.BatchNorm2d(2 * d),
            nn.LeakyReLU(0.2),
            # nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(2 * d, 4 * d, kernel_size=3, padding=1),
            nn.BatchNorm2d(4 * d),
            nn.LeakyReLU(0.2),
            # nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(4 * d, 8 * d, kernel_size=3, padding=1),
            nn.BatchNorm2d(8 * d),
            nn.LeakyReLU(0.2),
            # nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(8 * d, 8 * d, kernel_size=3, padding=1),
            nn.BatchNorm2d(8 * d),
            nn.LeakyReLU(0.2),
            # nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc = nn.Sequential(
            nn.Linear(self.d * 8 * 7 * 7, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 2),  # fake/real
            nn.Softmax(),
        )

    # forward method
    def forward(self, input):
        # print(input.size())
        x = self.conv(input)
        x_size = list(x.size())
        # print(list(x_size));
        # exit();
        x = x.view(-1, x_size[1] * x_size[2] * x_size[3])
        x = self.fc(x)
        return x


def calc_gradient_penalty(netD, real_data, fake_data):
    # 梯度惩罚
    print("=======")
    batch_size,c,w,h = real_data.size()
    alpha = torch.randn((batch_size, 1))
    alpha = alpha.expand(batch_size, int(real_data.nelement() / batch_size)).contiguous()
    alpha = alpha.view(real_data.size())

    # alpha = self.tensor2variable(alpha)
    alpha = Variable(alpha)
    interpolates = alpha * real_data.detach() + ((1 - alpha) * fake_data.detach())


    # print(interpolates.requires_grad)
    # interpolates = torch.FloatTensor(interpolates, requires_grad=True)
    # interpolates.requires_grad = True

    disc_interpolates = netD(interpolates)  # 多GPU训练网络
    print(disc_interpolates.requires_grad)
    print("=======",disc_interpolates.size())

    gradients = grad(outputs=disc_interpolates, inputs=interpolates,
                    grad_outputs=torch.ones(disc_interpolates.size()),
                    create_graph=True, retain_graph=True, only_inputs=True)[0]
    print(gradients)
    gradients = gradients.view(gradients.size(0), -1)
    print("=======")
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * 10
    print("=======")
    print("=======", gradient_penalty)

    return gradient_penalty


if __name__ == '__main__':
    real_data = Variable(torch.randn(11, 8, 224, 224))
    fake_data = Variable(torch.randn(11, 8, 224, 224))
    netD = Discriminator(8)
    calc_gradient_penalty(netD,real_data,fake_data)

