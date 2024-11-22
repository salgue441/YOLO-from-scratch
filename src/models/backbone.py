import torch
import torch.nn as nn


class ConvNormAct(nn.Module):
    """
    Base convolution block with normalization and activation.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int | None = None,
        groups: int = 1,
        bias: bool = False,
        norm_layer: nn.Module = nn.BatchNorm2d,
        act_layer: nn.Module = nn.ReLU,
    ):
        super(ConvNormAct, self).__init__()
        padding = padding or kernel_size // 2

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=bias,
        )

        self.norm = norm_layer(out_channels)
        self.act = act_layer()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the block.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor
        """

        return self.act(self.norm(self.conv(x)))


class CSPBlock(nn.Module):
    """
    Cross Stage Partial block with modern improvements.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        expansion: float = 0.5,
        use_residual: bool = True,
    ):
        super(CSPBlock, self).__init__()
        hidden_channels = int(out_channels * expansion)

        self.conv1 = ConvNormAct(in_channels, hidden_channels)
        self.conv2 = ConvNormAct(
            hidden_channels, out_channels=out_channels, kernel_size=1
        )
        self.use_residual = use_residual and in_channels == out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the block.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor
        """

        identity = x
        x = self.conv2(self.conv1(x))

        if self.use_residual:
            x += identity

        return x
