from typing import Tuple

import torch
import torch.nn as nn


class DistributionHead(nn.Module):
    """
    Detection head with distribution learning
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        num_anchors: int = 1,
        reg_max: int = 16,
    ):
        super(DistributionHead, self).__init__()
        self.num_classes = num_classes
        self.reg_max = reg_max

        self.cls_conv = nn.Conv2d(in_channels, num_classes * num_anchors, 1)
        self.reg_conv = nn.Conv2d(in_channels, 4 * reg_max * num_anchors, 1)
        self.obj_conv = nn.Conv2d(in_channels, num_anchors, 1)

        self.distribution = nn.Parameter(torch.linspace(0, self.reg_max + 1))

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = x.shape[0]
        cls_score = self.cls_conv(x)
        reg_feat = self.reg_conv(x)
        obj_score = self.obj_conv(x)

        reg_feat = reg_feat.reshape(batch_size, 4, self.reg_max + 1, -1)
        reg_dist = reg_feat.softmax(dim=2)
        reg_pred = (reg_dist * self.distribution.reshape(1, 1, -1, 1)).sum(dim=2)

        return cls_score, reg_pred, obj_score
