from __future__ import absolute_import
from __future__ import division

__all__ = ['resnet_mod_two_branch']

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models.resnet import resnet50, Bottleneck
import torchvision
import copy
import random
import math

if __name__ == '__main__':  # for debug
    from layer import *
else:
    from .layer import *


class ReNetMod(nn.Module):
    def __init__(self,
                 num_classes,
                 fc_dims=None,
                 loss=None,
                 with_attention=True,
                 **kwargs):
        super(ReNetMod, self).__init__()

        resnet_ = resnet50(pretrained=True)

        self.loss = loss

        self.layer0 = nn.Sequential(resnet_.conv1, resnet_.bn1, resnet_.relu,
                                    resnet_.maxpool)
        self.layer1 = resnet_.layer1
        self.layer2 = resnet_.layer2
        self.layer3 = resnet_.layer3
        self.layer4 = nn.Sequential(
            Bottleneck(1024,
                       512,
                       downsample=nn.Sequential(
                           nn.Conv2d(1024, 2048, 1, bias=False),
                           nn.BatchNorm2d(2048))), Bottleneck(2048, 512),
            Bottleneck(2048, 512))
        self.layer4.load_state_dict(resnet_.layer4.state_dict())

        self.pam = PAM_Module(2048)
        self.cam = CAM_Module(2048)

        self.global_avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.global_maxpool = nn.AdaptiveMaxPool2d((1, 1))
        self.gem = GeM()

        self.se_module1 = SEModule(2048, 16)
        self.se_module2 = SEModule(2048, 16)

        self.bn1 = nn.BatchNorm1d(2048)
        self.bn2 = nn.BatchNorm1d(2048)
        self.bn1.bias.requires_grad_(False)
        self.bn2.bias.requires_grad_(False)

        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        #self.dropout = nn.Dropout(0.5)

        self.classifier1 = nn.Linear(2048, num_classes, bias=False)
        self.classifier2 = nn.Linear(2048, num_classes, bias=False)

        self.bn1.apply(weights_init_kaiming)
        self.bn2.apply(weights_init_kaiming)
        self.classifier1.apply(weights_init_classifier)
        self.classifier2.apply(weights_init_classifier)

    def featuremaps(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def forward(self, x):
        f = self.featuremaps(x)  ## 64, 512, 16, 8

        f1 = self.pam(f)
        f2 = self.cam(f)

        v_avg = self.global_avgpool(f1)
        v_gem = self.gem(f2)

        v_avg = self.se_module1(v_avg)
        v_gem = self.se_module2(v_gem)

        v_avg = v_avg.view(v_avg.size(0), -1)
        v_gem = v_gem.view(v_gem.size(0), -1)

        fea = [v_avg, v_gem]

        v1 = self.bn1(v_avg)
        v2 = self.bn2(v_gem)

        if not self.training:
            v1 = F.normalize(v1, p=2, dim=1)
            v2 = F.normalize(v2, p=2, dim=1)
            v = torch.cat([v1, v2], dim=1)
            return v2

        #v = self.dropout(v)
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


def resnet_mod_two_branch(num_classes,
                          loss='softmax',
                          pretrained=True,
                          with_attention=True,
                          **kwargs):
    model = ReNetMod(num_classes=num_classes,
                     fc_dims=2048,
                     loss=loss,
                     with_attention=with_attention,
                     **kwargs)
    return model


if __name__ == '__main__':
    from torchsummary import summary
    model = resnet_mod_two_branch(100)
    print(model)
    print(summary(model, (3, 256, 128), device='cpu'))
