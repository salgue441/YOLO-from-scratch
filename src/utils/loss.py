from typing import Tuple

import torch
import torch.nn as nn


class YOLOLoss(nn.Module):
    def __init__(self, num_classes: int, reg_max: int = 16, use_focal: bool = True):
        super(YOLOLoss, self).__init__()
        self.num_classes = num_classes
        self.reg_max = reg_max
        self.use_focal = use_focal

    def forward(
        self,
        pred: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass of YOLO Loss

        Args:
            pred (Tuple[torch.Tensor, torch.Tensor, torch.Tensor]): Prediction from the model
            targets (torch.Tensor): Ground truth targets

        Returns:
            torch.Tensor: Loss value
        """

        cls_score, reg_pred, obj_score = pred

        cls_loss = self._focal_loss(cls_score, targets[..., 0])
        reg_loss = self._distribution_loss(reg_pred, targets[..., 1:5])
        obj_loss = self._objectness_loss(obj_score, targets[..., -1])

        return cls_loss + reg_loss + obj_loss

    def _focal_loss(
        self, cls_score: torch.Tensor, cls_target: torch.Tensor
    ) -> torch.Tensor:
        """
        Focal loss for classification

        Args:
            cls_score (torch.Tensor): Predicted class scores
            cls_target (torch.Tensor): Ground truth class targets

        Returns:
            torch.Tensor: Classification loss
        """
        if self.use_focal:
            cls_loss = nn.functional.binary_cross_entropy_with_logits(
                cls_score, cls_target, reduction="none"
            )
            cls_loss = (cls_loss * (cls_loss > 0.5).float()).mean()
        else:
            cls_loss = nn.functional.binary_cross_entropy_with_logits(
                cls_score, cls_target
            )

        return cls_loss

    def _distribution_loss(
        self, reg_pred: torch.Tensor, reg_target: torch.Tensor
    ) -> torch.Tensor:
        """
        Distribution loss for regression

        Args:
            reg_pred (torch.Tensor): Predicted regression values
            reg_target (torch.Tensor): Ground truth regression targets

        Returns:
            torch.Tensor: Regression loss
        """
        reg_loss = nn.functional.smooth_l1_loss(reg_pred, reg_target, reduction="none")
        reg_loss = reg_loss.sum(-1).mean()

        return reg_loss

    def _objectness_loss(
        self, obj_score: torch.Tensor, obj_target: torch.Tensor
    ) -> torch.Tensor:
        """
        Objectness loss

        Args:
            obj_score (torch.Tensor): Predicted objectness score
            obj_target (torch.Tensor): Ground truth objectness target

        Returns:
            torch.Tensor: Objectness loss
        """
        obj_loss = nn.functional.binary_cross_entropy_with_logits(obj_score, obj_target)

        return obj_loss
