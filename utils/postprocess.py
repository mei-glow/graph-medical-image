from __future__ import annotations

import torch


def extract_main_output(output):
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, tuple) and len(output) == 2 and isinstance(output[1], torch.Tensor):
        return output[1]
    if isinstance(output, (tuple, list)) and output and isinstance(output[0], torch.Tensor):
        return output[0]
    raise TypeError("Unsupported model output type for post-processing")


def output_to_prob(output) -> torch.Tensor:
    main = extract_main_output(output)
    if main.ndim != 4:
        raise ValueError(f"Expected model output with shape [B, C, H, W], got {tuple(main.shape)}")
    if torch.all(main >= 0) and torch.all(main <= 1):
        if main.size(1) in {1, 2}:
            return main[:, :1] if main.size(1) == 1 else main[:, 1:2]
    return logits_to_prob(main)


def logits_to_prob(logits: torch.Tensor) -> torch.Tensor:
    if logits.ndim != 4:
        raise ValueError(f"Expected logits with shape [B, C, H, W], got {tuple(logits.shape)}")

    if logits.size(1) == 1:
        return torch.sigmoid(logits)
    if logits.size(1) == 2:
        return torch.softmax(logits, dim=1)[:, 1:2]
    raise ValueError("Only binary segmentation outputs with 1 or 2 channels are supported")


def prob_to_mask(prob: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    return (prob >= threshold).float()
