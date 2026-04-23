from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F


def _flatten_binary(pred: torch.Tensor, target: torch.Tensor):
    pred = pred.float().flatten(1)
    target = target.float().flatten(1)
    return pred, target


def dice_score(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    pred, target = _flatten_binary(pred, target)
    intersection = (pred * target).sum(dim=1)
    union = pred.sum(dim=1) + target.sum(dim=1)
    return (2.0 * intersection + eps) / (union + eps)


def iou_score(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    pred, target = _flatten_binary(pred, target)
    intersection = (pred * target).sum(dim=1)
    union = pred.sum(dim=1) + target.sum(dim=1) - intersection
    return (intersection + eps) / (union + eps)


def precision_score(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    pred, target = _flatten_binary(pred, target)
    tp = (pred * target).sum(dim=1)
    fp = (pred * (1.0 - target)).sum(dim=1)
    return (tp + eps) / (tp + fp + eps)


def recall_score(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    pred, target = _flatten_binary(pred, target)
    tp = (pred * target).sum(dim=1)
    fn = ((1.0 - pred) * target).sum(dim=1)
    return (tp + eps) / (tp + fn + eps)


def specificity_score(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    pred, target = _flatten_binary(pred, target)
    tn = ((1.0 - pred) * (1.0 - target)).sum(dim=1)
    fp = (pred * (1.0 - target)).sum(dim=1)
    return (tn + eps) / (tn + fp + eps)


def mae_score(prob: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    prob, target = _flatten_binary(prob, target)
    return torch.abs(prob - target).mean(dim=1)


def _morphological_boundary(mask: torch.Tensor, radius: int = 3) -> torch.Tensor:
    kernel = 2 * radius + 1
    dilated = F.max_pool2d(mask.float(), kernel_size=kernel, stride=1, padding=radius)
    eroded = 1.0 - F.max_pool2d(1.0 - mask.float(), kernel_size=kernel, stride=1, padding=radius)
    boundary = (dilated - eroded).clamp(min=0.0, max=1.0)
    return (boundary > 0.0).float()


def boundary_dice_score(pred: torch.Tensor, target: torch.Tensor, radius: int = 3, eps: float = 1e-7) -> torch.Tensor:
    pred_boundary = _morphological_boundary(pred, radius=radius)
    target_boundary = _morphological_boundary(target, radius=radius)
    return dice_score(pred_boundary, target_boundary, eps=eps)


def boundary_iou_score(pred: torch.Tensor, target: torch.Tensor, radius: int = 3, eps: float = 1e-7) -> torch.Tensor:
    pred_boundary = _morphological_boundary(pred, radius=radius)
    target_boundary = _morphological_boundary(target, radius=radius)
    return iou_score(pred_boundary, target_boundary, eps=eps)


def false_positive_rate(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    pred, target = _flatten_binary(pred, target)
    fp = (pred * (1.0 - target)).sum(dim=1)
    tn = ((1.0 - pred) * (1.0 - target)).sum(dim=1)
    return (fp + eps) / (fp + tn + eps)


def false_negative_rate(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    pred, target = _flatten_binary(pred, target)
    fn = ((1.0 - pred) * target).sum(dim=1)
    tp = (pred * target).sum(dim=1)
    return (fn + eps) / (fn + tp + eps)


def over_segmentation_ratio(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    pred, target = _flatten_binary(pred, target)
    fp = (pred * (1.0 - target)).sum(dim=1)
    target_area = target.sum(dim=1)
    return (fp + eps) / (target_area + eps)


def under_segmentation_ratio(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    pred, target = _flatten_binary(pred, target)
    fn = ((1.0 - pred) * target).sum(dim=1)
    target_area = target.sum(dim=1)
    return (fn + eps) / (target_area + eps)


def lesion_area_ratio(target: torch.Tensor) -> torch.Tensor:
    flat = target.float().flatten(1)
    return flat.mean(dim=1)


def _surface_distances(pred_mask: torch.Tensor, target_mask: torch.Tensor, radius: int = 1):
    pred_boundary = _morphological_boundary(pred_mask, radius=radius)
    target_boundary = _morphological_boundary(target_mask, radius=radius)

    distances = []
    for pred_b, target_b in zip(pred_boundary, target_boundary):
        pred_points = torch.nonzero(pred_b.squeeze(0) > 0, as_tuple=False).float()
        target_points = torch.nonzero(target_b.squeeze(0) > 0, as_tuple=False).float()

        # If both masks are empty, there is no boundary mismatch.
        if pred_points.numel() == 0 and target_points.numel() == 0:
            distances.append(torch.tensor([0.0], dtype=torch.float32))
            continue

        # If one side is empty, return a finite image-scale penalty instead of inf.
        if pred_points.numel() == 0 or target_points.numel() == 0:
            height, width = pred_b.shape[-2], pred_b.shape[-1]
            max_distance = float((height**2 + width**2) ** 0.5)
            distances.append(torch.tensor([max_distance], dtype=torch.float32))
            continue

        pairwise = torch.cdist(pred_points, target_points, p=2.0)
        pred_to_target = pairwise.min(dim=1).values
        target_to_pred = pairwise.min(dim=0).values
        all_distances = torch.cat([pred_to_target, target_to_pred], dim=0)
        distances.append(all_distances)

    return distances


def hd95_score(pred: torch.Tensor, target: torch.Tensor, radius: int = 1) -> torch.Tensor:
    samples = _surface_distances(pred, target, radius=radius)
    values = []
    for distances in samples:
        if not isinstance(distances, torch.Tensor):
            raise TypeError("Surface distances must be tensors")
        distances = distances.float().flatten().cpu()
        values.append(torch.quantile(distances, 0.95))
    return torch.stack(values)


def compute_segmentation_metrics(
    pred_mask: torch.Tensor,
    target_mask: torch.Tensor,
    pred_prob: torch.Tensor | None = None,
) -> Dict[str, float]:
    if pred_prob is None:
        pred_prob = pred_mask.float()

    metrics = {
        "dice": dice_score(pred_mask, target_mask).mean().item(),
        "iou": iou_score(pred_mask, target_mask).mean().item(),
        "precision": precision_score(pred_mask, target_mask).mean().item(),
        "recall": recall_score(pred_mask, target_mask).mean().item(),
        "specificity": specificity_score(pred_mask, target_mask).mean().item(),
        "fpr": false_positive_rate(pred_mask, target_mask).mean().item(),
        "fnr": false_negative_rate(pred_mask, target_mask).mean().item(),
        "over_seg_ratio": over_segmentation_ratio(pred_mask, target_mask).mean().item(),
        "under_seg_ratio": under_segmentation_ratio(pred_mask, target_mask).mean().item(),
        "mae": mae_score(pred_prob, target_mask).mean().item(),
        "boundary_dice": boundary_dice_score(pred_mask, target_mask).mean().item(),
        "boundary_iou": boundary_iou_score(pred_mask, target_mask).mean().item(),
        "hd95": hd95_score(pred_mask, target_mask).mean().item(),
        "lesion_area_ratio": lesion_area_ratio(target_mask).mean().item(),
    }
    return metrics
