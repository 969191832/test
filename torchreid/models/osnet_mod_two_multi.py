from __future__ import absolute_import
from __future__ import division

__all__ = ['osnet_mod_two_multi']

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
        self.conv = nn.Conv2d(in_channels,
                              out_channels,
                              1,
                              stride=stride,
                              padding=0,
                              bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
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
        x = self.conv(x)
        x = self.bn(x)
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

        if self.with_attention:
            self.pam = PAM_Module(512)
            self.cam = CAM_Module(512)

        self.global_avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.global_maxpool = nn.AdaptiveMaxPool2d((1, 1))
        self.gem = GeM()
        self.gem2 = GeM()

        self.conv_2 = Conv1x1Linear(384, 128)

        if self.with_attention:
            self.se_module1 = SEModule(512, 16)
            self.se_module2 = SEModule(512, 16)

        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(640)
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)

        self.classifier1 = nn.Linear(512, num_classes, bias=False)
        self.classifier2 = nn.Linear(640, num_classes, bias=False)

        self._init_param()

    def _init_param(self):
        self.bn1.bias.requires_grad_(False)
        self.bn2.bias.requires_grad_(False)
        self.bn1.apply(weights_init_kaiming)
        self.bn2.apply(weights_init_kaiming)
        self.classifier1.apply(weights_init_classifier)
        self.classifier2.apply(weights_init_classifier)

    def featuremaps(self, x):
        x0 = self.layer0(x)
        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        return x2, x4

    def forward(self, x):
        x2, f = self.featuremaps(x)  ## 64, 512, 16, 8

        x2 = self.conv_2(x2)
        x2 = self.gem2(x2)
        x2 = x2.view(x2.size(0), -1)

        if self.with_attention:
            f1 = self.pam(f)
            f2 = self.cam(f)
            v_avg = self.global_avgpool(f1)
            v_gem = self.gem(f2)
            v_avg = self.se_module1(v_avg)
            v_gem = self.se_module2(v_gem)
        else:
            v_avg = self.global_avgpool(f)
            v_gem = self.gem(f)

        v_avg = v_avg.view(v_avg.size(0), -1)
        v_gem = v_gem.view(v_gem.size(0), -1)

        v_down = torch.cat([v_gem, x2], dim=1)

        fea = [v_avg, v_down]

        v1 = self.bn1(v_avg)
        v2 = self.bn2(v_down)

        if not self.training:
            v1 = F.normalize(v1, p=2, dim=1)
            v2 = F.normalize(v2, p=2, dim=1)
            v = torch.cat([v1, v2], dim=1)
            return v

        v1 = self.dropout1(v1)
        v2 = self.dropout2(v2)
        y1 = self.classifier1(v1)
        y2 = self.classifier2(v2)
        y = [y1, y2]
        if self.loss == 'softmax':
            return y
        elif self.loss == 'triplet':
            return y, fea
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))


def osnet_mod_two_multi(num_classes,
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
    model = osnet_mod_two_multi(100)
    print(model)
    import torch
    a = torch.rand(64, 3, 256, 128)
    model(a)
    # print(summary(model, (3, 256, 128), device='cpu'))
