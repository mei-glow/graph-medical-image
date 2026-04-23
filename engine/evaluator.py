from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import torch
from PIL import Image

from semantic_attention.utils.analysis import stratify_by_lesion_size
from semantic_attention.utils.io import ensure_dir, save_csv, save_json
from semantic_attention.utils.logger import SimpleLogger
from semantic_attention.utils.metrics import compute_segmentation_metrics
from semantic_attention.utils.postprocess import output_to_prob, prob_to_mask


def _save_mask_tensor(mask: torch.Tensor, path: Path) -> None:
    array = (mask.squeeze().detach().cpu().numpy() * 255).astype("uint8")
    Image.fromarray(array).save(path)


def evaluate(
    model,
    loader,
    device: str = "cuda",
    threshold: float = 0.5,
    save_predictions: bool = False,
    prediction_dir: str | Path | None = None,
    logger: SimpleLogger | None = None,
) -> Dict[str, float]:
    model.eval()
    metric_history: Dict[str, List[float]] = defaultdict(list)
    per_image_rows = []

    pred_dir = None
    if save_predictions:
        pred_dir = ensure_dir(prediction_dir or "outputs/predictions")
    elif prediction_dir is not None:
        ensure_dir(prediction_dir)

    num_batches = len(loader)
    if logger is not None:
        logger.log(
            f"Starting evaluation | split_samples={len(loader.dataset)} | batches={num_batches} | device={device}"
        )

    with torch.no_grad():
        for batch_idx, (images, masks, meta) in enumerate(loader, start=1):
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            output = model(images)
            probs = output_to_prob(output)
            preds = prob_to_mask(probs, threshold=threshold)

            batch_metrics = compute_segmentation_metrics(preds, masks, pred_prob=probs)
            for key, value in batch_metrics.items():
                metric_history[key].append(value)

            if logger is not None and (batch_idx == 1 or batch_idx % 20 == 0 or batch_idx == num_batches):
                logger.log(
                    f"Eval batch {batch_idx:04d}/{num_batches:04d} | "
                    f"dice={batch_metrics['dice']:.4f} | iou={batch_metrics['iou']:.4f} | "
                    f"boundary_dice={batch_metrics['boundary_dice']:.4f}"
                )

            ids = meta["id"] if isinstance(meta, dict) else [m["id"] for m in meta]
            for idx, sample_id in enumerate(ids):
                sample_metrics = compute_segmentation_metrics(
                    preds[idx : idx + 1],
                    masks[idx : idx + 1],
                    pred_prob=probs[idx : idx + 1],
                )
                sample_metrics["id"] = sample_id
                per_image_rows.append(sample_metrics)
                if pred_dir is not None:
                    _save_mask_tensor(preds[idx], pred_dir / f"{sample_id}.png")

    summary = {key: float(sum(values) / max(len(values), 1)) for key, values in metric_history.items()}
    summary["num_samples"] = len(loader.dataset)
    size_summary = stratify_by_lesion_size(per_image_rows)

    base_dir = Path(prediction_dir) if prediction_dir is not None else pred_dir
    if base_dir is not None:
        base_dir.mkdir(parents=True, exist_ok=True)
        save_json(summary, base_dir / "summary.json")
        save_csv(per_image_rows, base_dir / "per_image_metrics.csv")
        save_json(size_summary, base_dir / "size_stratified_metrics.json")

    if logger is not None:
        logger.log(
            f"Evaluation finished | dice={summary['dice']:.4f} | iou={summary['iou']:.4f} | "
            f"boundary_dice={summary['boundary_dice']:.4f} | hd95={summary['hd95']:.4f}"
        )

    return summary
