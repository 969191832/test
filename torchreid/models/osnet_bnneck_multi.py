from __future__ import absolute_import
from __future__ import division

__all__ = ['osnet_bnneck_multi']

import torch
from torch import nn
from torch.nn import functional as F
import torchvision
import copy
import random
import math

if __name__ == '__main__':  # for debug
    from osnet import *
    from layer import *
else:
    from .osnet import *
    from .layer import *


class Conv1x1Linear(nn.Module):
    """1x1 convolution + bn (w/o non-linearity)."""
    def __init__(self, in_channels, out_channels, stride=1):
        super(Conv1x1Linear, self).__init__()
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.conv = nn.Conv2d(in_channels,
                              out_channels,
                              1,
                              stride=stride,
                              padding=0,
                              bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self._init_params()

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.maxpool(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class OSNetMod(nn.Module):
    def __init__(self,
                 num_classes,
                 fc_dims=None,
                 loss=None,
                 with_attention=True,
                 **kwargs):
        super(OSNetMod, self).__init__()

        osnet = osnet_x1_0(pretrained=True)
        self.loss = loss
        self.with_attention = with_attention

        self.layer0 = nn.Sequential(osnet.conv1, osnet.maxpool)
        self.layer1 = osnet.conv2
        self.layer2 = osnet.conv3
        self.layer3 = osnet.conv4
        self.layer4 = osnet.conv5

        self.conv_0 = Conv1x1Linear(64, 32)
        self.conv_1 = Conv1x1Linear(256, 128)
        self.conv_2 = Conv1x1Linear(384, 192)
        self.conv_3 = Conv1x1Linear(288, 144)
        self.conv_4 = Conv1x1Linear(656, 328)
        self.conv_5 = Conv1x1Linear(1032, 516)
        # self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        # self.global_pool = nn.AdaptiveMaxPool2d((1, 1))
        self.global_pool = GeM()
        self.gem2 = GeM()
        #self.batchdrop = BatchDrop(0.3, 1.0)
        self.bn_fea = nn.BatchNorm1d(896)
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(896, num_classes, bias=False)

        self._init_param()

    def _init_param(self):
        self.bn_fea.bias.requires_grad_(False)
        self.bn_fea.apply(weights_init_kaiming)
        self.classifier.apply(weights_init_classifier)

    def featuremaps(self, x):
        x0 = self.layer0(x)  # 64
        print('x0.shape:', x0.shape)
        x1 = self.layer1(x0)  # 256
        print('x1.shape:', x1.shape)
        x2 = self.layer2(x1)  # 384
        print('x2.shape:', x2.shape)
        x3 = self.layer3(x2)  # 512
        print('x3.shape:', x3.shape)
        x4 = self.layer4(x3)
        print('x4.shape:', x4.shape)

        x1_0 = self.conv_0(x0)  # 32
        x2_0 = self.conv_1(x1)  # 128
        x3_0 = self.conv_2(x2)  # 192

        x2_1 = self.conv_4(torch.cat([x1, x1_0], dim=1))  # 256 + 32 --> 144
        x3_1 = self.conv_5(torch.cat([x2, x2_0, x2_1],
                                     dim=1))  # 384 + 128 + 144 ---> 328
        x4_1 = self.conv_6(torch.cat([x3, x3_0, x3_1],
                                     dim=1))  # 512 + 192 + 328 ---> 516

        return x4_1, x4

    def forward(self, x):
        x2, f = self.featuremaps(x)

        v = self.global_pool(f)
        v = v.view(v.size(0), -1)

        fea = torch.cat([v, x2], dim=1)
        fea_bn = self.bn_fea(fea)

        if not self.training:
            vv = F.normalize(fea_bn, p=2, dim=1)
            return vv

        fea_bn = self.dropout(fea_bn)
        y = self.classifier(fea_bn)
        y = [y]
        if self.loss == 'softmax':
            return y
        elif self.loss == 'triplet':
            return y, fea
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))


def osnet_bnneck_multi(num_classes,
                       loss='softmax',
                       pretrained=True,
                       with_attention=True,
                       **kwargs):
    model = OSNetMod(num_classes=num_classes,
                     fc_dims=512,
                     loss=loss,
                     with_attention=with_attention,
                     **kwargs)
    return model


if __name__ == '__main__':
    from torchsummary import summary
    model = osnet_bnneck_multi(100)
    print(model)
    print(summary(model, (3, 256, 128), device='cpu'))
