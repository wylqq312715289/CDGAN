# -*- coding:utf-8 -*-
# from __future__ import print_function #把下一个新版本的特性导入到当前版本
import os, sys, h5py, gc, argparse, codecs, shutil
import numpy as np
import pandas as pd
import random, logging
import json, cv2, time

logging.basicConfig(level=logging.INFO)

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import shuffle

import torch
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from datetime import datetime, timedelta
import dateutil
import dateutil.tz
from multiprocessing import Pool
from sklearn.ensemble import RandomForestClassifier

sys.path.append(".")

from modules.dataset import RSCGDataset
from modules.dataset import S_RSCGDataset
from modules.dataset import load_train_test_items
from modules.dataset import load_semi_supervised_train_test_items
from modules.config import conf, train_infos, test_infos

RANDOM_SEED = 20180122
DEBUG = False
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

def work(args, infos):
    if args.train:
        model_name, loss_name, trainer = infos
    else:
        model_name, loss_name, trainer, best_model_dict_path = infos
    if "ssgan" in model_name.lower():
        train_items, vali_items, unlabeled_items = load_semi_supervised_train_test_items(
            conf.trainset_info_file,
            conf.testset_info_file,
            conf.unlabeled_dataset_info_file
        )
        train_dataset = S_RSCGDataset(
            conf=conf,
            img_items=train_items,
            unlabeled_items=unlabeled_items,
            debug=args.debug,
            enhance=True,
        )
    else:
        train_items, vali_items = load_train_test_items(conf.trainset_info_file, conf.testset_info_file)
        train_dataset = RSCGDataset(
            conf=conf,
            img_items=train_items,
            debug=args.debug,
            enhance=True,
        )

    print("len(train set)={}".format(len(train_items)))
    print("len(vali set)={}".format(len(vali_items)))

    vali_dataset = RSCGDataset(
        conf=conf,
        img_items=vali_items,
        debug=args.debug,
        enhance=False,
    )
    assert train_dataset, "Dataset is NULL"
    train_loader = DataLoader(
        train_dataset,
        batch_size=conf.TRAIN.BATCH_SIZE,
        drop_last=True,
        shuffle=True,
        num_workers=int(conf.WORKERS)
    )
    vali_loader = DataLoader(
        vali_dataset,
        batch_size=conf.TRAIN.BATCH_SIZE,
        drop_last=True,
        shuffle=False,
        num_workers=int(conf.WORKERS)
    )
    now = datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H')
    model_path = "{}_{}_{}".format(timestamp, model_name, loss_name)
    if args.debug == 1:
        model_path += "_debug"
    model_path = os.path.join(PROJECT_DIR, "cache", model_path)
    print("Model Path = {}".format(model_path))
    conf['loss_name'] = loss_name
    conf['model_name'] = model_name
    if args.train:
        trainer = trainer(conf, model_path)
        trainer.train(train_loader, vali_loader)
    else:
        trainer = trainer(conf, model_path)
        trainer.predict(vali_loader, best_model_dict_path)

def get_argparse():
    args = argparse.ArgumentParser()
    args.add_argument('--debug', dest='debug', type=int, default=1)  # 是否使用小数据集测试程序
    args.add_argument('--train', dest='train', type=int, default=1)  # 是否使用小数据集测试程序
    args = args.parse_args()
    return args

def main(args):
    if args.train:
        for infos in train_infos:
            work(args, infos)
    else:
        for infos in test_infos:
            work(args, infos)

if __name__ == '__main__':
    args = get_argparse()
    main(args)
