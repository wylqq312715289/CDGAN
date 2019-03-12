# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd
import random, math
import os, cv2
import torch.utils.data as data
from PIL import Image
import PIL
import re
import torch
from torchvision.utils import save_image, make_grid
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from .utils import load_json

random.seed(20180122)
np.random.seed(20180122)



# 根据构造的图像信息存储格式读取list
def load_train_test_items(train_json_file, test_json_file):
    train_data_dict = load_json(train_json_file)
    test_data_dict = load_json(test_json_file)
    return list(train_data_dict.values()), list(test_data_dict.values())



# 根据构造的图像信息存储格式读取list
def load_semi_supervised_train_test_items(train_json_file, test_json_file,unlabel_json_file):
    train_data_dict = load_json(train_json_file)
    test_data_dict = load_json(test_json_file)
    unlabel_data_dict = load_json(unlabel_json_file)
    return list(train_data_dict.values()), list(test_data_dict.values()), list(unlabel_data_dict.values())


# 重构数据集器
class RSCGDataset(data.Dataset):
    def __init__(self, conf, img_items=None, debug=False, enhance=False):
        self.conf = conf
        self.debug = debug
        self.imsize = conf.IMSIZE
        self.data_dir = conf.DATA_DIR
        self.img_items = img_items[:20] if debug else img_items  # :list
        self.enhance = enhance

    def __len__(self):
        return len(self.img_items)

    def __getitem__(self, index):
        sample_info = self.img_items[index]
        img_1_file = sample_info.get("2006")
        img_2_file = sample_info.get("2013")
        img_label_file = sample_info.get("label")
        img_1 = self.load_img(os.path.join(self.data_dir, img_1_file), False)
        img_2 = self.load_img(os.path.join(self.data_dir, img_2_file), False)
        img_label = self.load_img(os.path.join(self.data_dir, img_label_file), True)
        # print(img_1.shape,img_2.shape)
        img = np.concatenate((img_1, img_2), axis=2)
        img, img_label = self.img_random_transfer(img, img_label)
        # img.size=(224,224,6)  img_label.size=(224,224,2)
        # print(np.min(img_label),np.max(img_label))
        assert (np.max(img) < 1.0 + 1e-3) and (np.min(img)> -1e-3), "img range not in [0,1]"
        assert (np.max(img_label) < 1.0 + 1e-3) and (np.min(img_label) > -1e-3), "img_label range not in [0,1]"
        img = np.transpose(img, (2, 0, 1))
        img_label = np.transpose(img_label, (2, 0, 1))
        img_label = img_label.astype(int)

        img = torch.FloatTensor(img)
        img_label = torch.LongTensor(img_label)
        # print(img.size(),img_label.size())
        # print(sample_info)
        # exit();
        return img, img_label, sample_info

    def resize_sample(self, img, label):
        new_img = cv2.resize(img, (self.imsize, self.imsize))
        new_label = cv2.resize(label, (self.imsize, self.imsize))
        result_label = np.zeros((self.imsize, self.imsize, 2))
        for i in range(self.imsize):
            for j in range(self.imsize):
                class_id = np.argmax(new_label[i, j, :])
                result_label[i, j, class_id] = 1
        assert np.sum(result_label) == (self.imsize * self.imsize), "sum(new_label) != %d * %d" % (self.imsize, self.imsize)
        return new_img, result_label

    def load_img(self, img_path, is_label=None):
        """
        :param img_path:
        :param is_label:
        :return: img(224,224,3) or img(224,224,2)
        """
        if is_label:
            label = cv2.imread(img_path, flags=cv2.IMREAD_UNCHANGED) # size = (512,512)
            label1 = np.where(label >= 0.5, 0, 1)
            label2 = np.where(label >= 0.5, 1, 0)
            img = np.array([label1, label2], dtype=np.int16).transpose([1, 2, 0]) # size = (512,512,2)
        else:
            img = cv2.imread(img_path)  # size = (512,512,3)
            img = 1.0 * img / 255
        assert not img is None, "{} is None".format(img_path)
        return img

    # 一张图像随机变换
    def img_random_transfer(self, img, label):
        """
        :param img: (512,512,6)
        :param label: (512,512,2)
        :return: img(224,224,6)  label(224,224,2)
        """
        new_img, new_label = img, label
        if self.enhance and random.random() < 0.5: # 随机水平或垂直翻转
            shift_id = random.randint(0,1)
            new_img = cv2.flip(new_img, shift_id)
            new_label = cv2.flip(new_label, shift_id)
        if self.enhance and random.random() < 0.5:  # 随机平移
            old_rows, old_cols = img.shape[:2]
            x_move1, x_move2 = (random.randint(0, 30), random.randint(0, 30))
            y_move1, y_move2 = (random.randint(0, 30), random.randint(0, 30))
            new_img = new_img[x_move1:old_rows-x_move2, y_move1:old_cols-y_move2]
            new_label = new_label[x_move1:old_rows-x_move2, y_move1:old_cols-y_move2]
        if self.enhance and random.random() < 0.5: # 随机旋转
            old_rows, old_cols = img.shape[:2]
            angle = random.randint(-180, 180)
            M = cv2.getRotationMatrix2D((old_cols / 2, old_rows / 2), angle, 1)# 最后一个参数为缩放因子
            new_img = cv2.warpAffine(img, M, (old_rows, old_cols))  # 需要图像、变换矩阵、变换后的大小
            new_label = cv2.warpAffine(label, M, (old_rows, old_cols))
            offset = int((2.0 - math.sqrt(2.0)) / 4.0 * old_rows)
            new_img = new_img[offset:-offset, offset:-offset]
            new_label = new_label[offset:-offset, offset:-offset]
        new_img, new_label = self.resize_sample(new_img, new_label)
        # print(new_img.shape, np.min(new_img), np.max(new_img))
        # print(new_label.shape,np.min(new_label),np.max(new_label))
        # exit()
        return new_img, new_label



class S_RSCGDataset(RSCGDataset):
    def __init__(self, conf, img_items=[], unlabeled_items=[], debug=False,enhance=True):
        super(S_RSCGDataset, self).__init__(conf, img_items, debug, enhance)
        self.unlabeled_items = unlabeled_items  # :list

    def get_unlabeled_img(self):
        unlabeled_idx = random.randint(0, len(self.unlabeled_items) - 1)
        # print(len(self.unlabeled_items),unlabeled_idx)
        item = self.unlabeled_items[unlabeled_idx]
        img_1_file = item.get("2010")
        img_2_file = item.get("2018")
        img_1 = self.load_img(os.path.join(self.data_dir, img_1_file), False)
        img_2 = self.load_img(os.path.join(self.data_dir, img_2_file), False)
        img = np.concatenate((img_1, img_2), axis=2)
        img = cv2.resize(img, (self.imsize, self.imsize))
        img = np.transpose(img, (2, 0, 1))
        img = torch.FloatTensor(img)
        return img

    def __getitem__(self, index):
        sample_info = self.img_items[index]
        img_1_file = sample_info.get("2006")
        img_2_file = sample_info.get("2013")
        img_label_file = sample_info.get("label")
        img_1 = self.load_img(os.path.join(self.data_dir, img_1_file), False)
        img_2 = self.load_img(os.path.join(self.data_dir, img_2_file), False)
        img_label = self.load_img(os.path.join(self.data_dir, img_label_file), True)
        # print(img_1.shape,img_2.shape)
        img = np.concatenate((img_1, img_2), axis=2)
        img, img_label = self.img_random_transfer(img, img_label)
        # img.size=(224,224,6)  img_label.size=(224,224,2)
        # print(np.min(img_label),np.max(img_label))
        img = np.transpose(img, (2, 0, 1))
        img_label = np.transpose(img_label, (2, 0, 1))
        img_label = img_label.astype(int)

        img = torch.FloatTensor(img)
        img_label = torch.LongTensor(img_label)
        img_unlabel = self.get_unlabeled_img()
        # print(img.size(),img_label.size())
        return img, img_label, img_unlabel, sample_info

