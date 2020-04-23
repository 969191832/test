from __future__ import print_function, division, absolute_import
import torch
import torch.nn as nn
from .gem_pooling import GeM

__all__ = ['SEModule']


class SEModule(nn.Module):

    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # self.gem = GeM()
        self.fc1 = nn.Conv2d(
            channels, channels // reduction, kernel_size=1, padding=0
        )
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(
            channels // reduction, channels, kernel_size=1, padding=0
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        # x = self.gem(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x


if __name__ == '__main__':
    a = SEModule(256, 16)
    print(a)
