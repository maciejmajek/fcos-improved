import torch
import torch.nn as nn
import torch.nn.functional as F


class IOULoss(nn.Module):
    def __init__(self, loss_type="iou"):
        super(IOULoss, self).__init__()
        self.loss_type = loss_type

    def forward(self, pred, target, weight=None):
        pred_left = pred[:, 0]
        pred_top = pred[:, 1]
        pred_right = pred[:, 2]
        pred_bottom = pred[:, 3]

        target_left = target[:, 0]
        target_top = target[:, 1]
        target_right = target[:, 2]
        target_bottom = target[:, 3]

        target_area = (target_left + target_right) * (target_top + target_bottom)
        pred_area = (pred_left + pred_right) * (pred_top + pred_bottom)

        w_intersect = torch.min(pred_left, target_left) + torch.min(
            pred_right, target_right
        )
        g_w_intersect = torch.max(pred_left, target_left) + torch.max(
            pred_right, target_right
        )
        h_intersect = torch.min(pred_bottom, target_bottom) + torch.min(
            pred_top, target_top
        )
        g_h_intersect = torch.max(pred_bottom, target_bottom) + torch.max(
            pred_top, target_top
        )
        ac_uion = g_w_intersect * g_h_intersect + 1e-7
        area_intersect = w_intersect * h_intersect
        area_union = target_area + pred_area - area_intersect
        ious = (area_intersect + 1.0) / (area_union + 1.0)
        gious = ious - (ac_uion - area_union) / ac_uion
        if self.loss_type == "iou":
            losses = -torch.log(ious)
        elif self.loss_type == "linear_iou":
            losses = 1 - ious
        elif self.loss_type == "giou":
            losses = 1 - gious
        else:
            raise NotImplementedError

        if weight is not None and weight.sum() > 0:
            return (losses * weight).sum()
        else:
            assert losses.numel() != 0
            return losses.mean()


class FocalLoss(nn.Module):
    def __init__(self, reduction="sum", alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy(
            inputs.squeeze(), targets.float(), reduction=self.reduction
        )
        loss = self.alpha * (1 - torch.exp(-bce_loss)) ** self.gamma * bce_loss
        if self.reduction == "sum":
            if targets.sum() > 0:
                loss = loss / targets.sum()
            else:
                loss = loss / targets.numel()
        return loss


class MulticlassFocalLoss(nn.Module):
    def __init__(self, reduction="sum", alpha=0.25, gamma=2.0):
        super(MulticlassFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        cle_loss = F.cross_entropy(
            inputs.float(), targets.long(), reduction=self.reduction
        )
        loss = self.alpha * (1 - torch.exp(-cle_loss)) ** self.gamma * cle_loss
        if self.reduction == "sum":
            if targets.sum() > 0:
                loss = loss / (targets > 0).sum()
            else:
                loss = loss / targets.numel()
        return loss
