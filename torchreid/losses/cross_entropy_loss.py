from __future__ import division, absolute_import
import torch
import torch.nn as nn


class CrossEntropyLoss(nn.Module):

    def __init__(
        self, num_classes, epsilon=0.1, use_gpu=True, label_smooth=True
    ):
        super(CrossEntropyLoss, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon if label_smooth else 0
        self.use_gpu = use_gpu
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        log_probs = self.logsoftmax(inputs)
        zeros = torch.zeros(log_probs.size())
        targets = zeros.scatter_(1, targets.unsqueeze(1).data.cpu(), 1)
        if self.use_gpu:
            targets = targets.cuda()
        targets = (
            1 - self.epsilon
        ) * targets + self.epsilon / self.num_classes
        return (-targets * log_probs).mean(0).sum()
