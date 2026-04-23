from __future__ import annotations

from pathlib import Path

import torch

from semantic_attention.engine.evaluator import evaluate
from semantic_attention.utils.checkpoint import save_checkpoint
from semantic_attention.utils.io import save_json
from semantic_attention.utils.logger import SimpleLogger


def _compute_loss(loss_fn, logits: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    loss = loss_fn(logits, masks)
    if not isinstance(loss, torch.Tensor):
        raise TypeError("Loss function must return a torch.Tensor")
    return loss


def train(
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    loss_fn,
    epochs: int,
    device: str = "cuda",
    output_dir: str | Path = "outputs/run",
    threshold: float = 0.5,
    early_stopping_patience: int = 20,
    protocol: dict | None = None,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = SimpleLogger(output_dir / "train.log")
    history = []
    best_dice = -1.0
    epochs_without_improvement = 0
    model.to(device)

    if protocol is not None:
        save_json(protocol, output_dir / "protocol.json")
        logger.log(f"Protocol: {protocol}")

    logger.log(
        f"Starting training | epochs={epochs} | device={device} | "
        f"train_samples={len(train_loader.dataset)} | val_samples={len(val_loader.dataset)}"
    )

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        num_batches = len(train_loader)
        logger.log(f"Epoch {epoch:03d}/{epochs:03d} started")

        for batch_idx, (images, masks, _meta) in enumerate(train_loader, start=1):
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = _compute_loss(loss_fn, logits, masks)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

            if batch_idx == 1 or batch_idx % 20 == 0 or batch_idx == num_batches:
                avg_loss_so_far = running_loss / (batch_idx * images.size(0))
                logger.log(
                    f"Epoch {epoch:03d}/{epochs:03d} | "
                    f"batch {batch_idx:04d}/{num_batches:04d} | "
                    f"loss={loss.item():.4f} | approx_avg_loss={avg_loss_so_far:.4f}"
                )

        train_loss = running_loss / len(train_loader.dataset)
        logger.log(f"Epoch {epoch:03d}/{epochs:03d} training finished | train_loss={train_loss:.4f}")

        logger.log(f"Epoch {epoch:03d}/{epochs:03d} validation started")
        val_metrics = evaluate(model, val_loader, device=device, threshold=threshold)
        epoch_record = {
            "epoch": epoch,
            "train_loss": train_loss,
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        history.append(epoch_record)

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_metrics["dice"])
            else:
                scheduler.step()

        latest_path = output_dir / "latest.pt"
        save_checkpoint(
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "history": history,
            },
            latest_path,
        )

        if val_metrics["dice"] > best_dice:
            best_dice = val_metrics["dice"]
            epochs_without_improvement = 0
            save_checkpoint(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "history": history,
                    "best_dice": best_dice,
                    "protocol": protocol,
                },
                output_dir / "best.pt",
            )
        else:
            epochs_without_improvement += 1

        logger.log(
            f"Epoch {epoch:03d} | train_loss={train_loss:.4f} | "
            f"val_dice={val_metrics['dice']:.4f} | val_iou={val_metrics['iou']:.4f} | "
            f"lr={optimizer.param_groups[0]['lr']:.6g} | "
            f"best_dice={best_dice:.4f} | no_improve={epochs_without_improvement}/{early_stopping_patience}"
        )

        if early_stopping_patience and early_stopping_patience > 0 and epochs_without_improvement >= early_stopping_patience:
            logger.log(
                f"Early stopping triggered at epoch {epoch:03d} "
                f"after {early_stopping_patience} epochs without validation Dice improvement."
            )
            break

    logger.log("Training finished")
    return history
