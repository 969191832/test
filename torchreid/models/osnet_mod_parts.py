from __future__ import absolute_import
from __future__ import division

__all__ = ['osnet_mod_parts']

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

        # self.non_local = Non_local(in_channels=512, reduc_ratio=8)

        # self.pam = PAM_Module(512)
        # self.cam1 = CAM_Module(512)
        # self.cam2 = CAM_Module(512)

        self.global_avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.global_maxpool = nn.AdaptiveMaxPool2d((1, 1))
        # self.gem1 = GeM()
        # self.gem2 = GeM()
        self.gem = GeM()
        # self.se_module1 = SEModule(512, 16)

        # self.fc = nn.Linear(fc_dims * 2, 512, bias=False)
        self.bn = nn.BatchNorm1d(1536)
        self.bn.bias.requires_grad_(False)

        self.dropout = nn.Dropout(0.5)

        self.classifier = nn.Linear(1536, num_classes, bias=False)

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
        f = self.featuremaps(x)  ## 64, 512, 16, 8
        B, C, H, W = f.size()

        f_up = f[:, :, :H // 2, :]
        f_down = f[:, :, H // 2:, :]

        v_all = self.global_avgpool(f)

        v_up = self.global_avgpool(f_up)
        # v_down = self.global_maxpool(f_down)
        v_down = self.gem(f_down)

        v_all = v_all.view(v_all.size(0), -1)
        fea_up = v_up.view(v_up.size(0), -1)
        fea_down = v_down.view(v_down.size(0), -1)

        fea = torch.cat([v_all, fea_up, fea_down], dim=1)

        v = self.bn(fea)
        if not self.training:
            v = F.normalize(v, p=2, dim=1)
            return v
            return torch.cat([v_all, fea_up, fea_down], dim=1)

        v = self.dropout(v)
        y1 = self.classifier(v)
        y = [y1]
        if self.loss == 'softmax':
            return y
        elif self.loss == 'triplet':
            return y, fea
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))


def osnet_mod_parts(num_classes,
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
    model = osnet_mod_parts(100)
    print(model)
    print(summary(model, (3, 256, 128), device='cpu'))
