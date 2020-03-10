from __future__ import absolute_import
from __future__ import division

__all__ = ['osnet_x1_0_mod']

import torch
from torch import nn
from torch.nn import functional as F
import torchvision
import copy
import random
import math

if __name__ == '__main__': # for debug
    from osnet import *
    from layer import *
else:
    from .osnet import *
    from .layer import *


class OSNetMod(nn.Module):

    def __init__(
        self,
        num_classes,
        fc_dims=None,
        loss=None,
        with_attention=True,
        **kwargs
    ):
        super(OSNetMod, self).__init__()

        osnet = osnet_x1_0(pretrained=True)
        self.loss = loss
        self.with_attention = with_attention

        self.layer0 = nn.Sequential(osnet.conv1, osnet.maxpool)
        self.layer1 = osnet.conv2
        if with_attention:
            self.attention_module1 = Attention_Module(256)
        self.layer2 = osnet.conv3
        if with_attention:
            self.attention_module2 = Attention_Module(384)
        self.layer3 = osnet.conv4
        self.layer4 = osnet.conv5

        self.global_avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.global_maxpool = nn.AdaptiveMaxPool2d((1, 1))
        self.gem = GeM()
        #self.batchdrop = BatchDrop(0.3, 1.0)
        if with_attention:
            self.se_module = SEModule(512, 16)
            self.se_module1 = SEModule(512, 16)
            self.se_module2 = SEModule(512, 16)

        # self.fc = nn.Linear(fc_dims * 2, 512, bias=False)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(512)
        self.bn1.bias.requires_grad_(False)
        self.bn2.bias.requires_grad_(False)

        #self.dropout = nn.Dropout(0.5)

        self.classifier1 = nn.Linear(512, num_classes, bias=False)
        self.classifier2 = nn.Linear(512, num_classes, bias=False)

        #self.fc.apply(weights_init_classifier)
        self.bn1.apply(weights_init_kaiming)
        self.bn2.apply(weights_init_kaiming)
        self.classifier1.apply(weights_init_classifier)
        self.classifier2.apply(weights_init_classifier)

    def featuremaps(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        if self.with_attention:
            x = self.attention_module1(x)
        x = self.layer2(x)
        if self.with_attention:
            x = self.attention_module2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def forward(self, x):
        f = self.featuremaps(x)
        #if self.training:
        #    f = self.batchdrop(f)
        # c = f.size(1) // 2;
        v_avg = self.global_avgpool(f)
        v_gem = self.gem(f)
        # v = self.global_avgpool(f)

        if self.with_attention:
            v_avg = self.se_module1(v_avg)
            v_gem = self.se_module2(v_gem)

        v_avg = v_avg.view(v_avg.size(0), -1)
        v_gem = v_gem.view(v_gem.size(0), -1)

        fea = [v_avg, v_gem]

        v1 = self.bn1(v_avg)
        v2 = self.bn2(v_gem)
        # v = torch.cat([v_avg, v_gem], dim=1)
        # fea = self.fc(v)
        # v = self.bn(fea)
        if not self.training:
            v1 = F.normalize(v1, p=2, dim=1)
            v2 = F.normalize(v2, p=2, dim=1)
            v = torch.cat([v1, v2], dim=1)
            return v

        #v = self.dropout(v)
        y1 = self.classifier1(v1)
        y2 = self.classifier2(v2)
        y = [y1, y2]
        if self.loss == 'softmax':
            return y
        elif self.loss == 'triplet':
            return y, fea
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))


def osnet_x1_0_mod(
    num_classes,
    loss='softmax',
    pretrained=True,
    with_attention=True,
    **kwargs
):
    model = OSNetMod(
        num_classes=num_classes,
        fc_dims=512,
        loss=loss,
        with_attention=with_attention,
        **kwargs
    )
    return model


if __name__ == '__main__':
    from torchsummary import summary
    model = osnet_x1_0_mod(100)
    print(model)
    print(summary(model, (3, 256, 128), device='cpu'))
