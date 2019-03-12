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
        self.feat3 = nn.Sequential(*feats[1:17])
        self.feat4 = nn.Sequential(*feats[17:24])
        self.feat5 = nn.Sequential(*feats[24:31])

        self.relu = nn.ReLU(inplace=True)
        self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn1 = nn.BatchNorm2d(512)

        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn2 = nn.BatchNorm2d(256)

        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn4 = nn.BatchNorm2d(64)

        self.deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn5 = nn.BatchNorm2d(32)
        self.classifier = nn.Conv2d(32, n_classes, kernel_size=1)

    def forward(self, x):
        feats = self.first_layer(x)
        feat3 = self.feat3(feats) # size=(N, 512, x.H/8, x.W/8)
        feat4 = self.feat4(feat3) # size=(N, 512, x.H/16, x.W/16)
        feat5 = self.feat5(feat4) # size=(N, 512, x.H/32, x.W/32)

        score = self.relu(self.deconv1(feat5))  # size=(N, 512, x.H/16, x.W/16)
        score = self.bn1(score + feat4)  # element-wise add, size=(N, 512, x.H/16, x.W/16)
        score = self.relu(self.deconv2(score))  # size=(N, 256, x.H/8, x.W/8)
        score = self.bn2(score + feat3)  # element-wise add, size=(N, 256, x.H/8, x.W/8)
        score = self.bn3(self.relu(self.deconv3(score)))  # size=(N, 128, x.H/4, x.W/4)
        score = self.bn4(self.relu(self.deconv4(score)))  # size=(N, 64, x.H/2, x.W/2)
        score = self.bn5(self.relu(self.deconv5(score)))  # size=(N, 32, x.H, x.W)
        score = self.classifier(score)  # size=(N, n_class, x.H/1, x.W/1)

        return score

class FCN16(nn.Module):
    def __init__(self, in_channels=6, n_classes=2, pretrained=True):
        super(FCN16, self).__init__()
        feats = list(models.vgg16(pretrained=pretrained).features.children())
        self.first_layer = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1) #改造VGG 第一层channel为6
        self.feat4 = nn.Sequential(*feats[1:24])
        self.feat5 = nn.Sequential(*feats[24:31])

        self.relu = nn.ReLU(inplace=True)
        self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn1 = nn.BatchNorm2d(512)
        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn5 = nn.BatchNorm2d(32)
        self.classifier = nn.Conv2d(32, n_classes, kernel_size=1)

    def forward(self, x):
        feats = self.first_layer(x)
        feat4 = self.feats(feats)
        feat5 = self.feat5(feat4)

        score = self.relu(self.deconv1(feat5))  # size=(N, 512, x.H/16, x.W/16)
        score = self.bn1(score + feat4)  # element-wise add, size=(N, 512, x.H/16, x.W/16)
        score = self.bn2(self.relu(self.deconv2(score)))  # size=(N, 256, x.H/8, x.W/8)
        score = self.bn3(self.relu(self.deconv3(score)))  # size=(N, 128, x.H/4, x.W/4)
        score = self.bn4(self.relu(self.deconv4(score)))  # size=(N, 64, x.H/2, x.W/2)
        score = self.bn5(self.relu(self.deconv5(score)))  # size=(N, 32, x.H, x.W)
        score = self.classifier(score)  # size=(N, n_class, x.H/1, x.W/1)

        return score  # size=(N, n_class, x.H/1, x.W/1)

class FCN32(nn.Module):
    def __init__(self, in_channels=6, n_classes=2, pretrained=True):
        super(FCN32, self).__init__()
        feats = list(models.vgg16(pretrained=pretrained).features.children())
        self.first_layer = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)  # 改造VGG 第一层channel为6
        self.feat5 = nn.Sequential(*feats[1:31])

        self.first_layer = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1) #改造VGG 第一层channel为6
        self.relu = nn.ReLU(inplace=True)
        self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn1 = nn.BatchNorm2d(512)
        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn5 = nn.BatchNorm2d(32)
        self.classifier = nn.Conv2d(32, n_classes, kernel_size=1)

    def forward(self, x):
        feats = self.first_layer(x)
        feat5 = self.feat5(feats)

        score = self.bn1(self.relu(self.deconv1(feat5)))  # size=(N, 512, x.H/16, x.W/16)
        score = self.bn2(self.relu(self.deconv2(score)))  # size=(N, 256, x.H/8, x.W/8)
        score = self.bn3(self.relu(self.deconv3(score)))  # size=(N, 128, x.H/4, x.W/4)
        score = self.bn4(self.relu(self.deconv4(score)))  # size=(N, 64, x.H/2, x.W/2)
        score = self.bn5(self.relu(self.deconv5(score)))  # size=(N, 32, x.H, x.W)
        score = self.classifier(score)  # size=(N, n_class, x.H/1, x.W/1)

        return score  # size=(N, n_class, x.H/1, x.W/1)


if __name__ == '__main__':
    # model = models.vgg16(pretrained=False)
    x = torch.FloatTensor(torch.randn(3, 5, 224, 224))


    print(model)