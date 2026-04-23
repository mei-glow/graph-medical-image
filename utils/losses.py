from __future__ import annotations

import torch
import torch.nn as nn


def dice_loss(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    probs = probs.flatten(1)
    targets = targets.flatten(1)
    intersection = (probs * targets).sum(dim=1)
    denominator = probs.sum(dim=1) + targets.sum(dim=1)
    dice = (2.0 * intersection + eps) / (denominator + eps)
    return 1.0 - dice.mean()


class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = self.bce(logits, targets)
        dice = dice_loss(logits, targets)
        return self.bce_weight * bce + self.dice_weight * dice


class ProbabilityBCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCELoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        size = pred.size(0)
        pred_ = pred.view(size, -1)
        target_ = target.view(size, -1)
        return self.bce(pred_, target_)


class ProbabilityDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        smooth = 1.0
        size = pred.size(0)
        pred_ = pred.view(size, -1)
        target_ = target.view(size, -1)
        intersection = pred_ * target_
        dice_score = (2 * intersection.sum(1) + smooth) / (pred_.sum(1) + target_.sum(1) + smooth)
        return 1 - dice_score.sum() / size


class BceDiceLoss(nn.Module):
    def __init__(self, wb: float = 1.0, wd: float = 1.0):
        super().__init__()
        self.bce = ProbabilityBCELoss()
        self.dice = ProbabilityDiceLoss()
        self.wb = wb
        self.wd = wd

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.wd * self.dice(pred, target) + self.wb * self.bce(pred, target)


class GTBceDiceLoss(nn.Module):
    def __init__(self, wb: float = 1.0, wd: float = 1.0):
        super().__init__()
        self.bcedice = BceDiceLoss(wb=wb, wd=wd)

    def forward(self, output, target: torch.Tensor) -> torch.Tensor:
        if not isinstance(output, tuple) or len(output) != 2:
            raise TypeError("GT_BceDiceLoss expects EGE-UNet output as ((gt_pre5..gt_pre1), out)")

        gt_pre, out = output
        if not isinstance(gt_pre, (tuple, list)) or len(gt_pre) != 5:
            raise TypeError("GT_BceDiceLoss expects five deep-supervision predictions")

        loss = self.bcedice(out, target)
        weights = [0.1, 0.2, 0.3, 0.4, 0.5]
        for w, pred in zip(weights, gt_pre):
            loss = loss + self.bcedice(pred, target) * w
        return loss


class SwinDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if logits.ndim != 4 or target.ndim != 4:
            raise ValueError("SwinDiceLoss expects tensors with shape [B, C, H, W]")

        if logits.size(1) == 1:
            probs = torch.sigmoid(logits)
            target_fg = target.float()
        elif logits.size(1) == 2:
            probs = torch.softmax(logits, dim=1)[:, 1:2]
            target_fg = target.float()
        else:
            raise ValueError("SwinDiceLoss supports binary outputs with 1 or 2 channels")

        smooth = 1e-5
        probs = probs.flatten(1)
        target_fg = target_fg.flatten(1)
        intersect = torch.sum(probs * target_fg, dim=1)
        y_sum = torch.sum(target_fg * target_fg, dim=1)
        z_sum = torch.sum(probs * probs, dim=1)
        dice = (2.0 * intersect + smooth) / (z_sum + y_sum + smooth)
        return 1.0 - dice.mean()


class TransUNetDiceLoss(nn.Module):
    def __init__(self, n_classes: int):
        super().__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor: torch.Tensor) -> torch.Tensor:
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(
        self,
        inputs: torch.Tensor,
        target: torch.Tensor,
        weight=None,
        softmax: bool = False,
    ) -> torch.Tensor:
        if softmax:
            inputs = torch.softmax(inputs, dim=1)

        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes

        assert inputs.size() == target.size(), f"predict {inputs.size()} & target {target.size()} shape do not match"

        loss = 0.0
        for i in range(self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            loss += dice * weight[i]
        return loss / self.n_classes


class SwinCEDiceLoss(nn.Module):
    def __init__(self, ce_weight: float = 0.4, dice_weight: float = 0.6):
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.ce = nn.CrossEntropyLoss()
        self.dice = SwinDiceLoss()

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if logits.size(1) == 1:
            ce_term = self.bce(logits, target.float())
        elif logits.size(1) == 2:
            ce_term = self.ce(logits, target[:, 0].long())
        else:
            raise ValueError("SwinCEDiceLoss supports binary outputs with 1 or 2 channels")
        dice_term = self.dice(logits, target)
        return self.ce_weight * ce_term + self.dice_weight * dice_term


class TransUNetCEDiceLoss(nn.Module):
    def __init__(self, n_classes: int = 2, ce_weight: float = 0.5, dice_weight: float = 0.5):
        super().__init__()
        self.n_classes = n_classes
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.ce = nn.CrossEntropyLoss()
        self.dice = TransUNetDiceLoss(n_classes=n_classes)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if logits.ndim != 4:
            raise ValueError("TransUNetCEDiceLoss expects logits with shape [B, C, H, W]")
        if target.ndim == 4 and target.size(1) == 1:
            target_indices = target[:, 0].long()
        elif target.ndim == 3:
            target_indices = target.long()
        else:
            raise ValueError("TransUNetCEDiceLoss expects target with shape [B, 1, H, W] or [B, H, W]")

        if logits.size(1) != self.n_classes:
            raise ValueError(
                f"TransUNetCEDiceLoss expects {self.n_classes} output channels, got {logits.size(1)}"
            )

        ce_term = self.ce(logits, target_indices)
        dice_term = self.dice(logits, target_indices, softmax=True)
        return self.ce_weight * ce_term + self.dice_weight * dice_term


class H2FormerLoss(nn.Module):
    def __init__(self, n_classes: int = 2):
        super().__init__()
        self.n_classes = n_classes
        self.ce = nn.CrossEntropyLoss()
        self.dice = TransUNetDiceLoss(n_classes=n_classes)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if logits.ndim != 4:
            raise ValueError("H2FormerLoss expects logits with shape [B, C, H, W]")
        if logits.size(1) != self.n_classes:
            raise ValueError(f"H2FormerLoss expects {self.n_classes} output channels, got {logits.size(1)}")

        if target.ndim == 4 and target.size(1) == 1:
            target_indices = target[:, 0].long()
        elif target.ndim == 3:
            target_indices = target.long()
        else:
            raise ValueError("H2FormerLoss expects target with shape [B, 1, H, W] or [B, H, W]")

        outputs_soft = torch.softmax(logits, dim=1)
        ce_term = self.ce(logits, target_indices)
        dice_term = self.dice(logits, target_indices, softmax=True)
        loss_margin = torch.mean(
            1 - (outputs_soft[:, 1, :, :] - outputs_soft[:, 0, :, :]) * target_indices.float()
        )
        return ce_term + dice_term + loss_margin


class TransFuseStructureLoss(nn.Module):
    def forward(self, pred: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        weit = 1 + 5 * torch.abs(torch.nn.functional.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
        wbce = torch.nn.functional.binary_cross_entropy_with_logits(pred, mask, reduction="none")
        wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

        pred_sigmoid = torch.sigmoid(pred)
        inter = ((pred_sigmoid * mask) * weit).sum(dim=(2, 3))
        union = ((pred_sigmoid + mask) * weit).sum(dim=(2, 3))
        wiou = 1 - (inter + 1) / (union - inter + 1)
        return (wbce + wiou).mean()


class TransFuseLoss(nn.Module):
    def __init__(self, alpha: float = 0.5, beta: float = 0.3, gamma: float = 0.2):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.structure = TransFuseStructureLoss()

    def forward(self, output, target: torch.Tensor) -> torch.Tensor:
        if not isinstance(output, (tuple, list)) or len(output) != 3:
            raise TypeError("TransFuseLoss expects three outputs: (lateral_map_4, lateral_map_3, lateral_map_2)")
        lateral_map_4, lateral_map_3, lateral_map_2 = output
        loss4 = self.structure(lateral_map_4, target)
        loss3 = self.structure(lateral_map_3, target)
        loss2 = self.structure(lateral_map_2, target)
        return self.alpha * loss2 + self.beta * loss3 + self.gamma * loss4


def build_loss(name: str, wb: float = 1.0, wd: float = 1.0):
    name = name.lower()
    if name == "bce":
        return nn.BCEWithLogitsLoss()
    if name == "dice":
        return dice_loss
    if name in {"bce_dice", "dice_bce"}:
        return BCEDiceLoss()
    if name in {"swin_ce_dice", "ce_dice_swin"}:
        return SwinCEDiceLoss()
    if name in {"transunet_ce_dice", "ce_dice_transunet"}:
        return TransUNetCEDiceLoss(n_classes=2, ce_weight=0.5, dice_weight=0.5)
    if name in {"h2former_loss", "ce_dice_margin", "h2former_ce_dice_margin"}:
        return H2FormerLoss(n_classes=2)
    if name in {"transfuse_loss", "structure_loss_transfuse"}:
        return TransFuseLoss(alpha=0.5, beta=0.3, gamma=0.2)
    if name in {"gt_bce_dice", "gt_bcedice", "ege_gt_bce_dice"}:
        return GTBceDiceLoss(wb=wb, wd=wd)
    raise ValueError(f"Unsupported loss '{name}'")
