#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/10/29 下午11:16
# @Author  : Kris
# @Site    :
# @File    : trainer
# @Software: PyCharm
# @describe:

import torch.optim as optim
import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import time, copy, os, re, cv2
from torch.autograd import grad
from torchvision.utils import save_image
from torch.nn.parallel import data_parallel
from tensorboardX import SummaryWriter

from .fcn import FCN32, FCN16, FCN8
from .focal_loss import FocalLoss
from .cdgan_discriminator import Discriminator
from .ssgan_generator import SegmentationG
from .u_net import U_net
from .utils import store_json,moveFileto
from .unlabeled_GAN_loss import Semi_Supervised_Unlabeled_Loss



class IOUMetric(object):
    """ Class to calculate mean-iou using fast_hist method """
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.hist = np.zeros((num_classes, num_classes))
        self.sub_metric = dict()

    def _fast_hist(self, label_pred, label_true):
        mask = (label_true >= 0) & (label_true < self.num_classes)
        ###### 统计各个值出现的次数
        hist = np.bincount(
            self.num_classes * label_true[mask].astype(int) +
            label_pred[mask], minlength=self.num_classes ** 2).reshape(self.num_classes, self.num_classes)
        return hist

    def add_batch(self, predictions, gts):
        for lp, lt in zip(predictions, gts):
            self.hist += self._fast_hist(lp.flatten(), lt.flatten())

    def evaluate(self):
        acc = np.diag(self.hist).sum() / self.hist.sum()
        mean_acc = np.diag(self.hist) / self.hist.sum(axis=1)
        mean_recall = np.diag(self.hist) / self.hist.sum(axis=0)
        mean_f1 = 2.0*mean_acc*mean_recall/(mean_acc+mean_recall)

        self.sub_metric["acc"] = acc
        self.sub_metric["mean_acc"] = np.nanmean(mean_acc)
        self.sub_metric["mean_recall"] = np.nanmean(mean_recall)
        self.sub_metric["mean_f1"] = np.nanmean(mean_f1)
        iou = np.diag(self.hist) / (self.hist.sum(axis=1) + self.hist.sum(axis=0) - np.diag(self.hist))
        self.sub_metric["iou"] = iou
        self.sub_metric["mean_iou"] = np.nanmean(iou)
        return self.sub_metric


class BaseMetricHandle(object):
    default_device_id = 0
    def __init__(self):
        self.img_w = None
        self.img_h = None
        self.model = None
        self.conf = None
        self.model_path = None
        self.lr_decay_step = 0
        self.optimizer = None
        self.g_optimizer = None
        self.d_optimizer = None
        self.max_epochs = int(1e8)
        self.lr = 0.0002
        self.g_lr = 0.0002
        self.d_lr = 0.0002
        self.iou_metric = None
        self.board_writer = SummaryWriter()
        self.tensorboard_data_dir = "./logs"
        self.best_model_path = None

    def get_sub_metric(self, pred_y, real_y, flag=0):
        """
        :param output:  numpy size = (batchsize,224,224)
        :param target:  numpy size = (batchsize,224,224)
        :param flag:
        :return:
        """
        sub_metric = dict()
        bs,img_w,img_h = real_y.shape
        batch_size = pred_y.shape[0]
        pred_y = pred_y.astype(np.int)
        real_y = real_y.astype(np.int)
        # print(np.min(pred_y),np.min(real_y),np.max(pred_y),np.max(real_y))


        # fun = lambda x, y: 1.0 * max(1.0,len(np.where((x == 1) & (y == 1))[0])) / max(1.0, len(np.where((x == 1) | (y == 1))[0]))
        fun = lambda x, y: 1.0 * len(np.where(x == y)[0]) / (1.0 * img_w * img_h)
        sub_metric["iou"] = np.sum(list(map(fun, pred_y, real_y)))


        fun = lambda x, y: 1.0 * len(np.where(x == y)[0]) / (img_w * img_h)
        sub_metric["pixel_acc"] = np.sum(list(map(fun, pred_y, real_y)))

        fun = lambda x, y: 1.0 * max(1.0,len(np.where((x == 1) & (y == 1))[0])) / max(1.0, len(np.where(x == 1)[0]))
        sub_metric["mean_acc"] = np.sum(list(map(fun, pred_y, real_y)))
        fun = lambda x, y: 1.0 * max(1.0,len(np.where((x == 0) & (y == 0))[0])) / max(1.0, len(np.where(x == 0)[0]))
        sub_metric["mean_acc"] += np.sum(list(map(fun, pred_y, real_y)))
        sub_metric["mean_acc"] /= 2

        # 正确率
        fun = lambda x, y: 1.0 * max(1.0,len(np.where((x == 1) & (y == 1))[0])) / max(1.0, len(np.where(x == 1)[0]))
        sub_metric["acc_detection"] = np.sum(list(map(fun, pred_y, real_y)))

        # 召回率
        fun = lambda x, y: 1.0 * max(1.0,len(np.where((x == 1) & (y == 1))[0])) / max(1.0, len(np.where(y == 1)[0]))
        sub_metric["recall_detection"] = np.sum(list(map(fun, pred_y, real_y)))

        # 误检率
        fun = lambda x, y: 1.0 * max(1.0,len(np.where(x == 1)[0]) - len(np.where((x == 1) & (y == 1))[0])) / max(1.0, len(
            np.where(x == 1)[0]))
        sub_metric["error_detection"] = np.sum(list(map(fun, pred_y, real_y)))

        # 漏检率
        fun = lambda x, y: 1.0 * max(1.0,len(np.where(y == 1)[0]) - len(np.where((x == 1) & (y == 1))[0])) / max(1.0, len(
            np.where(y == 1)[0]))
        sub_metric["omit_detection"] = np.sum(list(map(fun, pred_y, real_y)))

        # F1-score
        fun = lambda x, y: 2.0 * max(1.0,len(np.where((x == 1) & (y == 1))[0])) / max(1.0, len(np.where(x == 1)[0]) + len(
            np.where(y == 1)[0]))

        sub_metric["f1_score"] = np.sum(list(map(fun, pred_y, real_y)))

        return sub_metric


    def epoch_step(self, loader, forward_fun, prefix=""):
        metric = dict()
        keys = ["acc","mean_acc","mean_recall","iou","loss","f1_score","d_loss","g_loss"]
        for key in keys: metric["{}_{}".format(prefix,key)] = 0.0
        sample_num = 0
        # 分批训练样本
        for i, data in enumerate(loader, 1):
            # 通常情况下 train_x, train_y = data
            sample_num += len(data[0])
            metric = forward_fun(data, metric, batch_id=i)
        for key in keys:
            if key=="loss":
                metric["{}_{}".format(prefix, key)] /= (sample_num * self.img_w * self.img_h)
        sub_metric = self.iou_metric.evaluate()
        for key in sub_metric.keys():
            metric["%s_%s"%(prefix,key)] = sub_metric[key]
        return metric


    def metric2borad(self, metric, prefix="train",n_iter=0):
        """
        :param metric: type=dict {"*_acc":***,"_mean_acc":***,...}
        :param flag: type=str 'train_metric' or 'vali_metric'
        :param n_iter: epoch_id
        :return:
        """
        for board_key in metric.keys():
            self.board_writer.add_scalar(board_key, metric.get("board_key"), n_iter)


    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.02)  # 随机初始化模型参数  第一个参数为均值，第二个参数为方差
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)  # 随机初始化模型参数  第一个参数为均值，第二个参数为方差
            m.bias.data.fill_(0)  # 全部用0填充参数
        elif classname.find('Linear') != -1:
            m.weight.data.normal_(0.0, 0.02)  # 随机初始化模型参数  第一个参数为均值，第二个参数为方差
            if m.bias is not None:
                m.bias.data.fill_(0.0)  # 全部用0填充参数


    # torch模型在训练时的学习率调整
    def check_and_adjust_lr(self, epoch):
        #########################  一定轮数后调整学习率  #############################
        # [0,25,50,75]
        adjust_list = list(range(0,self.max_epochs,int(self.max_epochs/4)))
        if epoch == 0 or not epoch in adjust_list: return
        if not self.optimizer is None:
            self.lr *= 0.5
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.lr
        if not self.g_optimizer is None:
            self.g_lr *= 0.5
            for param_group in self.g_optimizer.param_groups:
                param_group['lr'] = self.g_lr
            self.d_lr *= 0.5
            for param_group in self.d_optimizer.param_groups:
                param_group['lr'] = self.d_lr

    # 将矩阵保存成图片
    def save_imgs(self, preds, sample_infos):
        # print(sample_infos)
        """
        :param pred: (batch_size, 224, 224)
        :param sample_info: {"2006":"dataset-v6/imgs/201801130329_2006.jpg","2013":...,"label":...}
        :return:
        """
        for i in range(len(sample_infos)):
            img = preds[i] * 255
            # print(np.max(img),np.min(img))
            input_img1_path = os.path.join(self.conf.DATA_DIR,sample_infos.get("2006")[i])
            input_img2_path = os.path.join(self.conf.DATA_DIR,sample_infos.get("2013")[i])
            input_label_path = os.path.join(self.conf.DATA_DIR,sample_infos.get("label")[i])
            pred_path = os.path.join(self.best_model_path,"test_output")
            if pred_path and not os.path.exists(pred_path): os.mkdir(pred_path)
            moveFileto(input_img1_path, os.path.join(pred_path, input_img1_path.split("/")[-1]))
            moveFileto(input_img2_path, os.path.join(pred_path, input_img2_path.split("/")[-1]))
            moveFileto(input_label_path, os.path.join(pred_path, input_label_path.split("/")[-1]))
            output_file_name = re.sub(r"label",r"pred",re.split(r"[\\/]",input_label_path)[-1])
            output_pred_path = os.path.join(pred_path, output_file_name)
            cv2.imwrite(output_pred_path, img)


    def tensor2variable(self, x):
        """Convert tensor to variable."""
        if torch.cuda.is_available():
            x = x.cuda(self.default_device_id)
        return Variable(x)


    def variable2numpy(self, x):
        """Convert variable to tensor."""
        if torch.cuda.is_available():
            x = x.cpu()
        return x.data.numpy()


    def denorm(self, x):
        """Convert range (-1, 1) to (0, 1)"""
        out = (x + 1) / 2
        return out.clamp(0, 1)



class BasicGanTrainer(BaseMetricHandle):
    LAMBDA = 10  # Gradient penalty lambda hyperparameter
    def __init__(self, conf, model_path=None):
        super(BasicGanTrainer, self).__init__()
        self.conf = conf
        self.class_num = conf.CLASS_NUM
        self.img_w = conf.IMSIZE
        self.img_h = conf.IMSIZE
        self.use_gpu = torch.cuda.is_available()  # 判断是否有GPU加速
        self.batch_size = conf.TRAIN.BATCH_SIZE
        self.max_epochs = self.conf.TRAIN.MAX_EPOCHS
        if model_path and not os.path.exists(model_path): os.mkdir(model_path)
        self.model_path = model_path
        # self.train_d_lambda = 0.6  # D_loss 占比
        # self.train_g_lambda = 0.8  # G_loss 占比
        self.g_lr = self.conf.TRAIN.G_LR
        self.d_lr = self.conf.TRAIN.D_LR
        self.lr_decay_step = self.conf.TRAIN.LR_DECAY_STEP
        self.sample_path = conf.SAMPLE_PATH
        self.iou_metric = IOUMetric(num_classes=self.class_num)
        s_gpus = conf.GPU_ID.split(',')  # 使用gpu id的list
        self.gpus = [int(ix) for ix in s_gpus]  # 使用gpu id的list
        self.default_device_id = self.gpus[0]
        self.board_writer = SummaryWriter(log_dir=model_path)
        self.init_GD()

    def init_GD(self):
        self.CrossE_criterion = FocalLoss(device_id=self.default_device_id)
        self.BCE_criterion = nn.BCELoss()
        self.D = Discriminator(in_channels=6+2)
        self.G = U_net(in_channels=6, n_classes=2)
        if self.use_gpu:
            self.D = self.D.cuda(self.default_device_id)
            self.G = self.G.cuda(self.default_device_id)
        self.g_optimizer = optim.Adam(self.G.parameters(), lr=self.g_lr, betas=(0.5, 0.999))
        self.d_optimizer = optim.Adam(self.D.parameters(), lr=self.d_lr, betas=(0.5, 0.999))
        self.G.apply(self.weights_init)
        self.D.apply(self.weights_init)


    def evalue_step(self, data, metric, batch_id):
        train_x, train_y, sample_infos = data
        mini_batch = len(train_y)
        self.G.eval()
        data_x = self.tensor2variable(train_x)
        _, data_y = torch.max(train_y, 1)
        data_y = self.tensor2variable(data_y.long())
        G_out = data_parallel(self.G, data_x, self.gpus)  # 多GPU训练网络
        G_unet_loss = self.CrossE_criterion(G_out, data_y)
        metric["vali_loss"] += (G_unet_loss.item() * mini_batch * self.img_w * self.img_h)
        _, pred = torch.max(G_out, 1)
        sub_metric = self.get_sub_metric(pred.data.cpu().numpy(), data_y.data.cpu().numpy())
        return metric, sub_metric


    def calc_gradient_penalty(self, netD, real_data, fake_data):
        # 梯度惩罚
        batch_size, c, w, h = real_data.size()
        alpha = torch.randn((batch_size, 1))
        alpha = alpha.expand(batch_size, int(real_data.nelement() / batch_size)).contiguous()
        alpha = alpha.view(real_data.size())
        alpha = self.tensor2variable(alpha)
        interpolates = alpha * real_data.detach() + ((1 - alpha) * fake_data.detach())

        # print(interpolates.requires_grad)
        # interpolates = torch.FloatTensor(interpolates, requires_grad=True)
        # interpolates.requires_grad_(True)
        interpolates.requires_grad = True

        # disc_interpolates = netD(interpolates)  # 多GPU训练网络
        disc_interpolates = data_parallel(netD, interpolates, self.gpus)  # 多GPU训练网络
        gradients = grad(outputs=disc_interpolates, inputs=interpolates,
                         grad_outputs=torch.ones(disc_interpolates.size()).to(self.default_device_id),
                         create_graph=True, retain_graph=True, only_inputs=True)[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * self.LAMBDA
        return gradient_penalty


    def train_step(self, data, metric, batch_id):
        train_x, train_y, sample_infos = data
        mini_batch = len(train_y)
        self.G.train()
        self.D.train()
        ################# define Variable ###############
        real_y = np.zeros((mini_batch, 2))
        real_y[:, 1] = 1
        real_y = self.tensor2variable(torch.FloatTensor(real_y))

        fake_y = np.zeros((mini_batch, 2))
        fake_y[:, 0] = 1
        fake_y = self.tensor2variable(torch.FloatTensor(fake_y))

        data_x = self.tensor2variable(train_x)
        _, data_y = torch.max(train_y, 1)
        # print(train_y.size(),torch.sum(train_y),torch.sum(data_y)); exit();

        data_y = self.tensor2variable(data_y.long())
        ###################################################################
        ############################# train D #############################
        ###################################################################
        self.d_optimizer.zero_grad()
        self.g_optimizer.zero_grad()
        for p in self.D.parameters():
            p.requires_grad = True  # reset requires_grad
        for p in self.G.parameters():
            p.requires_grad = False  # to avoid computation


        # forward real data
        D_real_input = torch.cat((train_x, train_y.float()), dim=1)
        D_real_input = self.tensor2variable(D_real_input)
        D_real_out = data_parallel(self.D, D_real_input, self.gpus)  # 多GPU训练网络
        D_real_loss = self.BCE_criterion(D_real_out, real_y)

        # forward fake data
        G_out = data_parallel(self.G, data_x, self.gpus)  # 多GPU训练网络
        D_fake_input = torch.cat((train_x, G_out.data.cpu()), dim=1)
        D_fake_input = self.tensor2variable(D_fake_input)
        D_fake_out = data_parallel(self.D, D_fake_input.detach(), self.gpus)  # 多GPU训练网络
        D_fake_loss = self.BCE_criterion(D_fake_out, fake_y)

        # backward
        gradient_penalty = self.calc_gradient_penalty(self.D,D_real_input,D_fake_input)
        D_train_loss = D_real_loss + D_fake_loss + gradient_penalty
        metric["train_d_loss"] += (D_train_loss.item() * mini_batch)

        D_train_loss.backward()
        self.d_optimizer.step()

        ###################################################################
        ############################# train G #############################
        ###################################################################
        self.d_optimizer.zero_grad()
        self.g_optimizer.zero_grad()
        for p in self.D.parameters():
            p.requires_grad = False  # to avoid computation
        for p in self.G.parameters():
            p.requires_grad = True  # to avoid computation

        G_out = data_parallel(self.G, data_x, self.gpus)  # 多GPU训练网络
        G_unet_loss = self.CrossE_criterion(G_out, data_y)
        D_fake_input = torch.cat((train_x, G_out.data.cpu()), dim=1)
        D_fake_input = self.tensor2variable(D_fake_input)

        D_fake_out = data_parallel(self.D, D_fake_input, self.gpus)  # 多GPU训练网络
        G_D_loss = self.BCE_criterion(D_fake_out, real_y)
        G_train_loss = G_unet_loss + 0.1 * G_D_loss
        metric["train_g_loss"] += (G_train_loss.item() * mini_batch)
        G_train_loss.backward()
        self.g_optimizer.step()
        # print("G loss",np.sum(G_train_loss.data.cpu().numpy()))

        ################ eval train ##############
        metric["train_loss"] += (G_unet_loss.item() * mini_batch * self.img_w * self.img_h)
        _, pred = torch.max(G_out, 1)
        sub_metric = self.get_sub_metric(pred.data.cpu().numpy(), data_y.data.cpu().numpy())
        return metric, sub_metric

    def predict_step(self, data, metric, batch_id):
        test_x, test_y, sample_infos = data
        self.G.eval()
        data_x = self.tensor2variable(test_x)
        _, data_y = torch.max(test_y, 1)
        data_y = self.tensor2variable(data_y)
        G_out = self.G(data_x)
        _, pred = torch.max(G_out, 1)
        sub_metric = self.get_sub_metric(pred.data.cpu().numpy(), data_y.data.cpu().numpy())
        # sample_png_fule_name = os.path.join(self.sample_path, 'predict_samples_%.3d.jpg' % batch_id)
        # save_image(self.denorm(G_out.data), sample_png_fule_name, nrow=4, padding=2)
        self.save_imgs(pred.data.cpu().numpy(), sample_infos)
        return metric, sub_metric

    def train(self, train_loader, vali_loader):
        log_data = []
        for epoch in range(self.max_epochs):
            st_time = time.time()
            self.check_and_adjust_lr(epoch)  # 调整优化器的学习率
            train_metric = self.epoch_step(train_loader, self.train_step, prefix="train")
            vali_metric = self.epoch_step(vali_loader, self.evalue_step, prefix="vali")
            print('[%.3d / %.3d] times use %.4f' % (epoch, self.conf.TRAIN.MAX_EPOCHS, time.time() - st_time,), train_metric, vali_metric)
            g_path = os.path.join(self.model_path, 'generator.pkl')
            d_path = os.path.join(self.model_path, 'discriminator.pkl')
            torch.save(self.G.state_dict(), g_path)
            torch.save(self.D.state_dict(), d_path)
            train_metric.update(vali_metric)
            train_metric["epoch"] = epoch
            log_data.append(train_metric)
            for board_key in train_metric.keys():
                self.board_writer.add_scalar(board_key, train_metric.get(board_key), epoch)
            for board_key in vali_metric.keys():
                self.board_writer.add_scalar(board_key, vali_metric.get(board_key), epoch)
            metric_path = os.path.join(self.model_path, 'metric.json')
            store_json(metric_path, log_data)
        self.board_writer.close()
        self.best_model_path = self.model_path
        test_metric = self.epoch_step(vali_loader, self.predict_step, prefix="test")
        print("test_metric", test_metric)


    # 模型预测
    def predict(self, test_loader):
        self.G.load_state_dict(torch.load(self.conf["BEST_CDGAN_MODEL_PATH"]))
        self.G.eval()
        test_metric = self.epoch_step(test_loader, self.predict_step, prefix="test")
        print("test_metric", test_metric)



class BasicNNTrainer(BaseMetricHandle):
    def __init__(self, conf, model_path):
        super(BasicNNTrainer, self).__init__()
        self.conf = conf
        self.img_w = conf.IMSIZE
        self.img_h = conf.IMSIZE
        self.class_num = conf.CLASS_NUM
        self.use_gpu = torch.cuda.is_available()
        if model_path and not os.path.exists(model_path): os.mkdir(model_path)
        self.model_path = model_path
        self.lr = conf.TRAIN.UNET_LR
        self.lr_decay_step = conf.TRAIN.LR_DECAY_STEP
        self.sample_path = conf.SAMPLE_PATH
        s_gpus = conf.GPU_ID.split(',')  # 使用gpu id的list
        self.gpus = [int(ix) for ix in s_gpus]  # 使用gpu id的list
        self.default_device_id = self.gpus[0]
        self.board_writer = SummaryWriter(log_dir=model_path)
        self.init_model()


    def init_model(self):
        if self.conf.model_name=="FCN8":
            self.model = FCN8(in_channels=6, n_classes=self.conf.CLASS_NUM, pretrained=False)
            print(self.model)
        elif self.conf.model_name=="FCN16":
            self.model = FCN16(in_channels=6, n_classes=self.conf.CLASS_NUM, pretrained=False)
        elif self.conf.model_name=="FCN32":
            self.model = FCN32(in_channels=6, n_classes=self.conf.CLASS_NUM, pretrained=False)
        else:
            self.model =  U_net(in_channels=6, n_classes=self.conf.CLASS_NUM)
        if self.conf.loss_name == "CrossEntropyLoss":
            self.criterion = nn.CrossEntropyLoss()
        else:
            self.criterion = FocalLoss(device_id=self.default_device_id)
        if self.use_gpu:
            self.criterion = self.criterion.cuda(self.default_device_id )
            self.model = self.model.cuda(self.default_device_id )
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, betas=(0.5, 0.999))
        self.model.apply(self.weights_init)


    def evalue_step(self, data, metric, batch_id):
        self.model.eval()
        train_x, train_y, sample_infos = data
        mini_batch = len(train_y)
        data_x = self.tensor2variable(train_x)
        _, data_y = torch.max(train_y, 1)
        data_y = self.tensor2variable(data_y.long())
        outputs = self.model(data_x)
        loss = self.criterion(outputs, data_y)
        metric["vali_loss"] += (loss.item() * mini_batch * self.img_w * self.img_h)
        _, pred = torch.max(outputs, 1)
        self.iou_metric.add_batch(pred.data.cpu().numpy(), data_y.data.cpu().numpy())
        return metric


    def train_step(self, data, metric, batch_id):
        self.model.train()
        train_x, train_y, sample_infos = data
        mini_batch = len(train_y)
        data_x = self.tensor2variable(train_x)
        _, data_y = torch.max(train_y, 1)
        data_y = self.tensor2variable(data_y.long())

        # forward
        outputs = self.model(data_x)  # 正向传播
        loss = self.criterion(outputs, data_y)  # 使用交叉熵计算损失

        # backward
        self.optimizer.zero_grad()
        loss.backward()  # 反向传播
        self.optimizer.step()  # 用优化器更新参数

        # 计算该轮误差
        metric["train_loss"] += (loss.item() * mini_batch * self.img_w * self.img_h)
        _, pred = torch.max(outputs, 1)
        sub_metric = self.get_sub_metric(pred.data.cpu().numpy(), data_y.data.cpu().numpy())
        return metric, sub_metric


    def predict_step(self, data, metric, batch_id):
        self.model.eval()
        train_x, train_y, sample_infos = data
        data_x = self.tensor2variable(train_x)
        _, data_y = torch.max(train_y, 1)
        data_y = self.tensor2variable(data_y.long())
        # print(data_x.size())
        # print(self.model)
        outputs = self.model(data_x)
        _, pred = torch.max(outputs, 1)
        sub_metric = self.get_sub_metric(pred.data.cpu().numpy(), data_y.data.cpu().numpy())
        self.save_imgs(pred.data.cpu().numpy(),sample_infos)
        return metric, sub_metric


    def train(self, train_loader, vali_loader):
        log_data = []
        self.best_model = copy.deepcopy(self.model)
        for epoch in range(self.conf.TRAIN.MAX_EPOCHS):
            st_time = time.time()
            self.check_and_adjust_lr(epoch)  # 调整优化器的学习率
            train_metric = self.epoch_step(train_loader, self.train_step, prefix="train")
            vali_metric = self.epoch_step(vali_loader, self.evalue_step, prefix="vali")
            print('[%.3d / %.3d] times use %.4f' % (epoch, self.conf.TRAIN.MAX_EPOCHS, time.time() - st_time,),
                  train_metric, vali_metric)
            model_path = os.path.join(self.model_path, 'model_parameters.pkl')
            torch.save(self.model.state_dict(), model_path)
            train_metric.update(vali_metric)
            train_metric["epoch"] = epoch
            log_data.append(train_metric)
            for board_key in train_metric.keys():
                self.board_writer.add_scalar(board_key, train_metric.get(board_key), epoch)
            for board_key in vali_metric.keys():
                self.board_writer.add_scalar(board_key, vali_metric.get(board_key), epoch)
            metric_path = os.path.join(self.model_path, 'metric.json')
            store_json(metric_path, log_data)
        self.board_writer.close()
        self.best_model_path = self.model_path
        test_metric = self.epoch_step(vali_loader, self.predict_step, prefix="test")
        print("test_metric", test_metric)


    # 模型预测
    def predict(self, test_loader, best_model_file):
        self.best_model_path = os.path.dirname(best_model_file)
        self.model.load_state_dict(torch.load(best_model_file))
        test_metric = self.epoch_step(test_loader, self.predict_step, prefix="test")
        print("test_metric", test_metric)



class BasicSSGanTrainer(BaseMetricHandle):
    def __init__(self, conf, model_path=None):
        super(BasicSSGanTrainer, self).__init__()
        self.conf = conf
        self.class_num = conf.CLASS_NUM
        self.img_w = conf.IMSIZE
        self.img_h = conf.IMSIZE
        self.use_gpu = torch.cuda.is_available()  # 判断是否有GPU加速
        self.batch_size = conf.TRAIN.BATCH_SIZE
        self.max_epochs = conf.TRAIN.MAX_EPOCHS
        if model_path and not os.path.exists(model_path): os.mkdir(model_path)
        self.model_path = model_path
        # self.CrossE_criterion = nn.CrossEntropyLoss()
        self.CrossE_criterion = FocalLoss(class_num=self.class_num+1)
        # self.BCE_criterion = nn.BCELoss()
        self.unlabeled_loss_fun = Semi_Supervised_Unlabeled_Loss()
        self._lambda = 0.8  # D_loss 占比
        self.g_lr = self.conf.TRAIN.G_LR
        self.d_lr = self.conf.TRAIN.D_LR
        self.lr_decay_step = self.conf.TRAIN.LR_DECAY_STEP
        self.sample_path = conf.SAMPLE_PATH
        s_gpus = conf.GPU_ID.split(',')  # 使用gpu id的list
        self.gpus = [int(ix) for ix in s_gpus]  # 使用gpu id的list
        self.default_device_id = self.gpus[0]
        self.board_writer = SummaryWriter(log_dir=model_path)
        self.init_GD()

    def init_GD(self):
        self.G = SegmentationG()
        self.D = U_net(in_channels=6, n_classes=self.class_num+1)
        if self.use_gpu:
            self.D = self.D.cuda(self.default_device_id)
            self.G = self.G.cuda(self.default_device_id)
        # print(self.G)
        # print(self.D)
        self.g_optimizer = optim.Adam(self.G.parameters(), lr=self.g_lr, betas=(0.5, 0.999))
        self.d_optimizer = optim.Adam(self.D.parameters(), lr=self.d_lr, betas=(0.5, 0.999))
        # self.g_optimizer = optim.SGD(self.G.parameters(), lr=self.g_lr, momentum=0.8)
        # self.d_optimizer = optim.SGD(self.D.parameters(), lr=self.d_lr, momentum=0.8)
        self.G.apply(self.weights_init)
        self.D.apply(self.weights_init)

    def evalue_step(self, data, metric, batch_id):
        train_x, train_y, sample_infos = data
        self.D.eval()
        mini_batch = len(train_y)
        data_x = self.tensor2variable(train_x)
        _, data_y = torch.max(train_y, 1)
        data_y = self.tensor2variable(data_y.long())
        D_out = data_parallel(self.D, data_x, self.gpus)  # 多GPU训练网络
        D_unet_loss = self.CrossE_criterion(D_out, data_y)
        metric["vali_loss"] += (D_unet_loss.item() * mini_batch * self.img_w * self.img_h)
        indices = torch.LongTensor([0, 1])
        if self.use_gpu: indices = indices.cuda()
        D_labeled_out = torch.index_select(D_out, 1, indices)
        _, pred = torch.max(D_labeled_out, 1)
        sub_metric = self.get_sub_metric(pred.data.cpu().numpy(), data_y.data.cpu().numpy())
        return metric, sub_metric


    def train_step(self, data, metric, batch_id):
        train_x, train_y, unlabeled_train_x, sample_infos = data
        mini_batch = len(train_y)
        self.D.train()
        self.G.train()
        # print(unlabeled_train_x.size())
        # exit()
        ################# define Variable ###############
        _, real_y = torch.max(train_y, 1) # (bt,2,w,h) => (bt,w,h)
        fake_y = self.tensor2variable(torch.ones(real_y.size()).long()*self.class_num) # 下标从0开始
        real_y = self.tensor2variable(real_y.long())
        labeled_x = self.tensor2variable(train_x)
        unlabeled_x = self.tensor2variable(unlabeled_train_x)
        noise = self.tensor2variable(torch.randn(mini_batch,256,7,7))
        ###################################################################
        ############################# train D #############################
        ###################################################################
        self.d_optimizer.zero_grad()
        self.g_optimizer.zero_grad()
        for p in self.D.parameters():
            p.requires_grad = True
        for p in self.G.parameters():
            p.requires_grad = False  # to avoid computation

        #################    forward real data    ################
        D_labeled_out = data_parallel(self.D, labeled_x, self.gpus)  # 多GPU训练网络
        D_real_loss = self.CrossE_criterion(D_labeled_out, real_y)

        #################    forward fake data    ################
        G_fake_out = data_parallel(self.G, noise, self.gpus)  # 多GPU训练网络
        D_fake_out = data_parallel(self.D, G_fake_out.detach(), self.gpus)  # 多GPU训练网络
        D_fake_loss = self.CrossE_criterion(D_fake_out, fake_y)

        #################    forward unlabeled data   ################
        D_unlabeled_out = data_parallel(self.D, unlabeled_x, self.gpus)  # 多GPU训练网络
        D_unlabeled_loss = self.unlabeled_loss_fun(D_unlabeled_out)

        # backward
        D_train_loss = D_real_loss +  0.01 * D_fake_loss + 0.5 * D_unlabeled_loss
        # print(D_real_loss.item(), D_fake_loss.item(), D_unlabeled_loss.item())
        metric["train_d_loss"] += (D_train_loss.item() * mini_batch)

        D_train_loss.backward()
        self.d_optimizer.step()

        ###################################################################
        ############################# train G #############################
        ###################################################################
        self.d_optimizer.zero_grad()
        self.g_optimizer.zero_grad()
        for p in self.D.parameters():
            p.requires_grad = False  # to avoid computation
        for p in self.G.parameters():
            p.requires_grad = True  # to avoid computation

        noise = self.tensor2variable(torch.randn(mini_batch, 256, 7, 7))
        G_fake_out = data_parallel(self.G, noise, self.gpus)  # 多GPU训练网络
        D_fake_out = data_parallel(self.D, G_fake_out, self.gpus)  # 多GPU训练网络
        D_fake_loss = self.unlabeled_loss_fun(D_fake_out)
        G_train_loss = D_fake_loss
        metric["train_g_loss"] += (G_train_loss.item() * mini_batch)
        G_train_loss.backward()
        self.g_optimizer.step()
        # print("G loss",np.sum(G_train_loss.data.cpu().numpy()))

        ################ eval train ##############
        metric["train_loss"] += (D_real_loss.item() * mini_batch * self.img_w * self.img_h)
        indices = torch.LongTensor([0, 1])
        if self.use_gpu: indices = indices.cuda()
        D_labeled_out = torch.index_select(D_labeled_out, 1, indices)
        _, pred = torch.max(D_labeled_out, 1)
        sub_metric = self.get_sub_metric(pred.data.cpu().numpy(), real_y.data.cpu().numpy())
        return metric, sub_metric


    def predict_step(self, data, metric, batch_id):
        test_x, test_y, sample_infos = data
        self.D.eval()
        data_x = self.tensor2variable(test_x)
        _, data_y = torch.max(test_y, 1)
        data_y = self.tensor2variable(data_y)
        D_out = self.D(data_x)
        indices = torch.LongTensor([0, 1])
        if self.use_gpu: indices = indices.cuda()
        D_fake_out = torch.index_select(D_out, 1, indices)
        _, pred = torch.max(D_fake_out, 1)
        sub_metric = self.get_sub_metric(pred.data.cpu().numpy(), data_y.data.cpu().numpy())
        sample_png_fule_name = os.path.join(self.sample_path, 'predict_samples_%.3d.jpg' % batch_id)
        save_image(self.denorm(D_out.data), sample_png_fule_name, nrow=4, padding=2)
        return metric, sub_metric


    def train(self, train_loader, vali_loader):
        log_data = []
        for epoch in range(self.max_epochs):
            st_time = time.time()
            self.check_and_adjust_lr(epoch)  # 调整优化器的学习率
            train_metric = self.epoch_step(train_loader, self.train_step, prefix="train")
            vali_metric = self.epoch_step(vali_loader, self.evalue_step, prefix="vali")
            print('[%.3d / %.3d] times use %.4f' % (epoch, self.conf.TRAIN.MAX_EPOCHS, time.time() - st_time,), train_metric, vali_metric)
            g_path = os.path.join(self.model_path, 'generator.pkl')
            d_path = os.path.join(self.model_path, 'discriminator.pkl')
            torch.save(self.G.state_dict(), g_path)
            torch.save(self.D.state_dict(), d_path)
            train_metric.update(vali_metric)
            train_metric["epoch"] = epoch
            log_data.append(train_metric)
            for board_key in train_metric.keys():
                self.board_writer.add_scalar(board_key, train_metric.get(board_key), epoch)
            for board_key in vali_metric.keys():
                self.board_writer.add_scalar(board_key, vali_metric.get(board_key), epoch)
            metric_path = os.path.join(self.model_path, 'metric.json')
            store_json(metric_path, log_data)
        self.board_writer.close()

    # 模型预测
    def predict(self, test_loader):
        self.G.load_state_dict(torch.load(self.conf["BEST_CDGAN_MODEL_PATH"]))
        self.G.eval()
        test_metric = self.epoch_step(test_loader, self.predict_step, prefix="test")
        print("test_metric", test_metric)














