from src.models.backbone import ConvNormAct, CSPBlock
from src.models.head import DistributionHead

import pytest
import torch


def test_conv_norm_act():
    layer = ConvNormAct(3, 64)
    x = torch.randn(1, 3, 32, 32)
    out = layer(x)

    assert out.shape == (1, 64, 32, 32)
    assert not torch.isnan(out).any()


def test_csp_block():
    block = CSPBlock(64, 64)
    x = torch.randn(1, 64, 32, 32)
    out = block(x)

    assert out.shape == (1, 64, 32, 32)
    assert not torch.isnan(out).any()


def test_distribution_head():
    head = DistributionHead(512, 80)
    x = torch.randn(1, 512, 20, 20)
    cls_score, reg_pred, obj_score = head(x)

    assert cls_score.shape[1] == 80
    assert reg_pred.shape[1] == 4
    assert obj_score.shape[1] == 1
