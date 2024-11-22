from typing import List, Tuple
from neck import SPPFNeck
from head import DistributionHead
from backbone import ConvNormAct, CSPBlock

import torch
import torch.nn as nn


class YOLO(nn.Module):
    def __init__(
        self,
        num_classes: int,
        in_channels: int = 3,
        backbone_channels: List[int] = [64, 128, 256, 512],
    ):
        super(YOLO, self).__init__()

        self.backbone = self._build_backbone(in_channels, backbone_channels)
        self.neck = SPPFNeck(backbone_channels[-1], backbone_channels[-1])
        self.head = DistributionHead(backbone_channels[-1], num_classes)

    def _build_backbone(self, in_channels: int, channels: List[int]) -> nn.ModuleList:
        """
        Builds the backbone of the model using the provided channels.

        Args:
          in_channels (int): Number of input channels.
          channels (List[int]): List of channels for each layer.

        Returns:
          nn.ModuleList: List of backbone layers.
        """

        layers = []
        curr_channels = in_channels

        for out_channels in channels:
            layers.append(
                nn.Sequential(
                    ConvNormAct(curr_channels, out_channels),
                    CSPBlock(out_channels, out_channels),
                )
            )

            curr_channels = out_channels

        return nn.ModuleList(layers)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of the model that returns the output of the backbone, neck and head.

        Args:
          x (torch.Tensor): Input tensor.

        Returns:
          Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Output of the backbone, neck and head.
        """

        for layer in self.backbone:
            x = layer(x)

        x = self.neck(x)
        return self.head(x)
