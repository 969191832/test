"""
Code source: https://github.com/pytorch/vision
"""
from __future__ import absolute_import
from __future__ import division

__all__ = ['resnet_ori']

import torch
from torch import nn
from torch.nn import functional as F
import torchvision
import torch.utils.model_zoo as model_zoo
from torchvision.models.resnet import resnet50, Bottleneck
import copy


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.normal_(m.weight, 1.0, 0.02)
            nn.init.constant_(m.bias, 0.0)


class ResNet(nn.Module):
    def __init__(self,
                 num_classes,
                 fc_dims=None,
                 loss=None,
                 dropout_p=None,
                 **kwargs):
        super(ResNet, self).__init__()

        resnet_ = resnet50(pretrained=True)

        self.loss = loss

        self.layer0 = nn.Sequential(resnet_.conv1, resnet_.bn1, resnet_.relu,
                                    resnet_.maxpool)
        self.layer1 = resnet_.layer1
        self.layer2 = resnet_.layer2
        self.layer3 = resnet_.layer3
        layer4 = nn.Sequential(
            Bottleneck(1024,
                       512,
                       downsample=nn.Sequential(
                           nn.Conv2d(1024, 2048, 1, bias=False),
                           nn.BatchNorm2d(2048))), Bottleneck(2048, 512),
            Bottleneck(2048, 512))
        layer4.load_state_dict(resnet_.layer4.state_dict())

        self.layer40 = nn.Sequential(copy.deepcopy(layer4))
        #self.layer41 = nn.Sequential(copy.deepcopy(layer4))

        self.global_avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.bn0 = nn.BatchNorm1d(2048)
        self.classifier0 = nn.Linear(2048, num_classes, bias=False)

        self.bn0.bias.requires_grad_(False)
        self.bn0.apply(weights_init_kaiming)
        self.classifier0.apply(weights_init_kaiming)

    def featuremaps(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x1 = self.layer40(x)
        return x1

    def forward(self, x):
        f1 = self.featuremaps(x)
        v0 = self.global_avgpool(f1)
        v0 = v0.view(v0.size(0), -1)
        v0 = self.bn0(v0)

        if not self.training:
            v0 = F.normalize(v0, p=2, dim=1)
            return v0

        y0 = self.classifier0(v0)

        if self.loss == 'softmax':
            return y0
        elif self.loss == 'triplet':
            return y0, v0
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))


def resnet_ori(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = ResNet(num_classes=num_classes,
                   fc_dims=None,
                   loss=loss,
                   dropout_p=None,
                   **kwargs)
    return model
