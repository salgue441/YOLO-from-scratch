from backbone import ConvNormAct


import torch
import torch.nn as nn


class SPPFNeck(nn.Module):
    """
    Spatial Pyramid Pooling - Fast Neck
    """

    def __init__(self, in_channels: int, out_channels: int):
        super(SPPFNeck, self).__init__()
        hidden_channels = in_channels // 2

        self.conv1 = ConvNormAct(in_channels, hidden_channels, kernel_size=1)
        self.pool = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)
        self.conv2 = ConvNormAct(hidden_channels * 4, hidden_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forwad pass of the neck

        Args:
            x (torch.Tensor): input tensor

        Returns:
            torch.Tensor: output tensor
        """

        x = self.conv1(x)
        y1 = self.pool(x)
        y2 = self.pool(y1)
        y3 = self.pool(y2)

        return self.conv2(torch.cat([x, y1, y2, y3], dim=1))
