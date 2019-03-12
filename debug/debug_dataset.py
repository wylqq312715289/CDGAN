# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd
import random
import os, cv2
import torch.utils.data as data
from PIL import Image
import PIL
import os
import os.path
import pickle
import random
import cv2
import re
import torch
from torchvision.utils import save_image, make_grid
from torch.utils.data.dataset import Dataset
from torchvision import transforms
# from ..modules.utils import load_json
from PIL import Image, ImageOps, ImageEnhance
import matplotlib.pyplot as plt



img1_ = cv2.imread("./test_imgs/201801130068_2006.jpg")
img2_ = cv2.imread("./test_imgs/201801130068_2013.jpg")
label_ = cv2.imread("./test_imgs/201801130068_label.jpg",flags=cv2.IMREAD_UNCHANGED)
# label_ =
cont_ = np.concatenate((img1_,img2_,np.reshape(label_,(label_.shape[0],label_.shape[1],1))),axis=2)
# print(img1.shape)
# print(img2.shape)
# print(label.shape)
# print(cont_.shape)

# img1 = Image.open("./test_imgs/201801130068_2006.jpg")
# img2 = Image.open("./test_imgs/201801130068_2013.jpg")
# label = Image.open("./test_imgs/201801130068_label.jpg")





img1 = Image.fromarray(img1_)
img2 = Image.fromarray(img2_)
label = Image.fromarray(label_)
# cont = Image.fromarray(cont_)

# print(img1.mode,img1.size)
# print(img2.mode,img2.size)
# print(label.mode,label.size)
# print(label.mode,label.size)

transformer = transforms.Compose([
    # transforms.RandomSizedCrop(512),  # 先将给定的PIL.Image随机切
    transforms.RandomHorizontalFlip(),  # 0.5的概率随机翻转
    transforms.Scale(224 + 30),  # 改变原图(长边,短边)大小为(size*长边/短边,size)
    # transforms.CenterCrop(self.imsize),  # 给定的图片进行中心切割, 变成正方形
    transforms.RandomCrop(224),  # 给定的图片进行随机切割成正方形
    # transforms.ToTensor(),  # 将图片转换成tensor range [0, 255] -> [0.0,1.0]
    # transforms.Normalize(  # 根据参数把Tensor正则化。
    #     mean=[0.5, 0.5, 0.5],
    #     std=[0.5, 0.5, 0.5],
    # )
])

tensor1 = transformer((img1, img2, label))

tensor2 = transformer(img2)
tensor_label = transformer(label)
# tensor_label = transformer(cont_)
# print(np.asarray(tensor1))
# tensor1.show()




tensor1 = np.asarray(tensor1)
tensor1 = tensor1.astype('uint8')

tensor2 = np.asarray(tensor2)
tensor2 = tensor2.astype('uint8')

tensor_label = np.asarray(tensor_label)
tensor_label = tensor_label.astype('uint8')

# tensor1 = np.transpose(tensor1, (1,2,0))
# cv2.imshow('img_1', tensor1)
plt.figure(1, figsize=(60, 10), dpi=90)

plt.subplot(3, 5, 1)
plt.imshow(img1_)
plt.subplot(3, 5, 6)
plt.imshow(img2_)
plt.subplot(3, 5, 11)
plt.imshow(label_)
plt.subplot(3, 5, 2)
plt.imshow(tensor1)
plt.subplot(3, 5, 7)
plt.imshow(tensor2)
plt.subplot(3, 5, 12)
plt.imshow(tensor_label)



plt.show()
# cv2.waitKey()

