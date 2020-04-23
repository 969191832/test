from __future__ import absolute_import
from __future__ import division

__all__ = ['osnet_bnneck']

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

        # self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        # self.global_pool = nn.AdaptiveMaxPool2d((1, 1))
        self.global_pool = GeM()
        #self.batchdrop = BatchDrop(0.3, 1.0)
        self.bn = nn.BatchNorm1d(512)
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(512, num_classes, bias=False)

        self._init_param()

    def _init_param(self):
        self.bn.bias.requires_grad_(False)
        self.bn.apply(weights_init_kaiming)
        self.classifier.apply(weights_init_classifier)
        
    def featuremaps(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def forward(self, x):
        f = self.featuremaps(x)
        v = self.global_pool(f)
        v = v.view(v.size(0), -1)

        fea = [v]
        v = self.bn(v)
        if not self.training:
            v = F.normalize(v, p=2, dim=1)
            # v = torch.cat([v1, v2], dim=1)
            return v

        v = self.dropout(v)
        y = self.classifier(v)
        y = [y]
        if self.loss == 'softmax':
            return y
        elif self.loss == 'triplet':
            return y, fea
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))


def osnet_bnneck(num_classes,
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
    model = osnet_x1_0_mod(100)
    print(model)
    print(summary(model, (3, 256, 128), device='cpu'))
