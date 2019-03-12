#-*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

from torch.utils import model_zoo
from torchvision import models


# 最复杂
class FCN8(nn.Module):
    def __init__(self, in_channels=6, n_classes=2,pretrained=True):
        super(FCN8, self).__init__()
        feats = list(models.vgg16(pretrained=pretrained).features.children())

        self.first_layer = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1) #改造VGG 第一层channel为6
        self.feats = nn.Sequential(*feats[1:9])
        self.feat3 = nn.Sequential(*feats[10:16])
        self.feat4 = nn.Sequential(*feats[17:23])
        self.feat5 = nn.Sequential(*feats[24:30])

        # first_conv2d_flag = 0
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         if first_conv2d_flag==0:
        #             first_conv2d_flag = 1
        #             continue
        #         m.requires_grad = False

        self.fconn = nn.Sequential(
            nn.Conv2d(512, 4096, 7),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Conv2d(4096, 4096, 1),
            nn.ReLU(inplace=True),
            nn.Dropout(),
        )
        self.score_feat3 = nn.Conv2d(256, n_classes, 1)
        self.score_feat4 = nn.Conv2d(512, n_classes, 1)
        self.score_fconn = nn.Conv2d(4096, n_classes, 1)

    def forward(self, x):
        feats = self.first_layer(x)
        print(feats.size())
        feats = self.feats(feats)
        feat3 = self.feat3(feats) # size=(N, 512, x.H/8, x.W/8)
        feat4 = self.feat4(feat3) # size=(N, 512, x.H/16, x.W/16)
        feat5 = self.feat5(feat4) # size=(N, 512, x.H/32, x.W/32)
        print(feat5.size())
        fconn = self.fconn(feat5)

        score_feat3 = self.score_feat3(feat3)
        score_feat4 = self.score_feat4(feat4)
        score_fconn = self.score_fconn(fconn)

        print(score_fconn.size(),score_feat4.size())
        score = F.upsample(score_fconn, score_feat4.size()[2:])

        score += score_feat4
        score = F.upsample(score, score_feat3.size()[2:])
        score += score_feat3
        out = F.upsample(score, x.size()[2:])
        # print( out.size() )
        return out

class FCN16(nn.Module):
    def __init__(self, in_channels=6, n_classes=2, pretrained=True):
        super(FCN16, self).__init__()
        feats = list(models.vgg16(pretrained=pretrained).features.children())

        self.feats = nn.Sequential(*feats[0:16])
        self.feat4 = nn.Sequential(*feats[17:23])
        self.feat5 = nn.Sequential(*feats[24:30])
        self.fconn = nn.Sequential(
            nn.Conv2d(512, 4096, 7),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Conv2d(4096, 4096, 1),
            nn.ReLU(inplace=True),
            nn.Dropout(),
        )
        self.score_fconn = nn.Conv2d(4096, n_classes, 1)
        self.score_feat4 = nn.Conv2d(512, n_classes, 1)

    def forward(self, x):
        feats = self.feats(x)
        feat4 = self.feat4(feats)
        feat5 = self.feat5(feat4)
        fconn = self.fconn(feat5)

        score_feat4 = self.score_feat4(feat4)
        score_fconn = self.score_fconn(fconn)

        score = F.upsample(score_fconn, score_feat4.size()[2:])
        score += score_feat4

        return F.upsample(score, x.size()[2:])

class FCN32(nn.Module):
    def __init__(self, in_channels=6, n_classes=2, pretrained=True):
        super(FCN32, self).__init__()
        self.feats = models.vgg16(pretrained=pretrained).features
        self.fconn = nn.Sequential(
            nn.Conv2d(512, 4096, 7),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Conv2d(4096, 4096, 1),
            nn.ReLU(inplace=True),
            nn.Dropout(),
        )
        self.score = nn.Conv2d(4096, n_classes, 1)

    def forward(self, x):
        feats = self.feats(x)
        fconn = self.fconn(feats)
        score = self.score(fconn)

        return F.upsample(score, x.size()[2:])


if __name__ == '__main__':
    model = models.vgg16(pretrained=False)
    print(model)