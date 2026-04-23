from __future__ import annotations

import argparse
from pathlib import Path

import torch

from semantic_attention.config import (
    DEFAULT_PROTOCOL,
    build_protocol,
    flatten_config_sections,
    load_config_file,
)
from semantic_attention.datasets import create_dataloader
from semantic_attention.engine import train
from semantic_attention.models import build_model, list_models
from semantic_attention.utils import build_loss, set_seed
from semantic_attention.utils.io import save_json


def parse_args():
    parser = argparse.ArgumentParser(description="Train a segmentation model on ISIC 2018")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--model", type=str, default=None, choices=list_models())
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--optimizer", type=str, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--momentum", type=float, default=None)
    parser.add_argument("--scheduler", type=str, default=None)
    parser.add_argument("--scheduler-t-max", type=int, default=None)
    parser.add_argument("--scheduler-eta-min", type=float, default=None)
    parser.add_argument("--scheduler-patience", type=int, default=None)
    parser.add_argument("--scheduler-power", type=float, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--loss", type=str, default=None)
    parser.add_argument("--loss-wb", type=float, default=None)
    parser.add_argument("--loss-wd", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--base-channels", type=int, default=None)
    parser.add_argument("--pretrained-path", type=str, default=None)
    parser.add_argument("--dataset-name", type=str, default=None)
    parser.add_argument("--train-normalize-mode", type=str, default=None)
    parser.add_argument("--eval-normalize-mode", type=str, default=None)
    parser.add_argument("--train-hflip-prob", type=float, default=None)
    parser.add_argument("--train-vflip-prob", type=float, default=None)
    parser.add_argument("--train-rotation-prob", type=float, default=None)
    parser.add_argument("--train-rotation-min-deg", type=float, default=None)
    parser.add_argument("--train-rotation-max-deg", type=float, default=None)
    parser.add_argument("--train-color-jitter", action="store_true")
    parser.add_argument("--train-brightness-jitter", type=float, default=None)
    parser.add_argument("--train-contrast-jitter", type=float, default=None)
    parser.add_argument("--train-saturation-jitter", type=float, default=None)
    parser.add_argument("--train-hue-jitter", type=float, default=None)
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=None,
    )
    return parser.parse_args()


def resolve_args(args):
    config_values = {}
    if args.config:
        config_values = flatten_config_sections(load_config_file(args.config))

    resolved = {
        "model": args.model or config_values.get("model"),
        "data_root": args.data_root or config_values.get("data_root"),
        "output_dir": args.output_dir or config_values.get("output_dir") or "outputs",
        "image_size": args.image_size if args.image_size is not None else config_values.get("image_size"),
        "batch_size": args.batch_size if args.batch_size is not None else config_values.get("batch_size"),
        "num_workers": args.num_workers if args.num_workers is not None else config_values.get("num_workers"),
        "epochs": args.epochs if args.epochs is not None else config_values.get("epochs"),
        "optimizer": args.optimizer or config_values.get("optimizer"),
        "weight_decay": args.weight_decay if args.weight_decay is not None else config_values.get("weight_decay"),
        "momentum": args.momentum if args.momentum is not None else config_values.get("momentum"),
        "scheduler": args.scheduler or config_values.get("scheduler"),
        "scheduler_t_max": (
            args.scheduler_t_max if args.scheduler_t_max is not None else config_values.get("scheduler_t_max")
        ),
        "scheduler_eta_min": (
            args.scheduler_eta_min
            if args.scheduler_eta_min is not None
            else config_values.get("scheduler_eta_min")
        ),
        "scheduler_patience": (
            args.scheduler_patience
            if args.scheduler_patience is not None
            else config_values.get("scheduler_patience")
        ),
        "scheduler_power": (
            args.scheduler_power if args.scheduler_power is not None else config_values.get("scheduler_power")
        ),
        "lr": args.lr if args.lr is not None else config_values.get("lr"),
        "loss": args.loss or config_values.get("loss"),
        "loss_wb": args.loss_wb if args.loss_wb is not None else config_values.get("loss_wb"),
        "loss_wd": args.loss_wd if args.loss_wd is not None else config_values.get("loss_wd"),
        "seed": args.seed if args.seed is not None else config_values.get("seed"),
        "device": args.device or config_values.get("device"),
        "base_channels": args.base_channels if args.base_channels is not None else config_values.get("base_channels"),
        "pretrained_path": args.pretrained_path or config_values.get("pretrained_path"),
        "dataset_name": args.dataset_name or config_values.get("dataset_name"),
        "train_normalize_mode": args.train_normalize_mode or config_values.get("train_normalize_mode"),
        "eval_normalize_mode": args.eval_normalize_mode or config_values.get("eval_normalize_mode"),
        "train_hflip_prob": (
            args.train_hflip_prob if args.train_hflip_prob is not None else config_values.get("train_hflip_prob")
        ),
        "train_vflip_prob": (
            args.train_vflip_prob if args.train_vflip_prob is not None else config_values.get("train_vflip_prob")
        ),
        "train_rotation_prob": (
            args.train_rotation_prob
            if args.train_rotation_prob is not None
            else config_values.get("train_rotation_prob")
        ),
        "train_rotation_min_deg": (
            args.train_rotation_min_deg
            if args.train_rotation_min_deg is not None
            else config_values.get("train_rotation_min_deg")
        ),
        "train_rotation_max_deg": (
            args.train_rotation_max_deg
            if args.train_rotation_max_deg is not None
            else config_values.get("train_rotation_max_deg")
        ),
        "train_color_jitter": args.train_color_jitter or config_values.get("train_color_jitter"),
        "train_brightness_jitter": (
            args.train_brightness_jitter
            if args.train_brightness_jitter is not None
            else config_values.get("train_brightness_jitter")
        ),
        "train_contrast_jitter": (
            args.train_contrast_jitter
            if args.train_contrast_jitter is not None
            else config_values.get("train_contrast_jitter")
        ),
        "train_saturation_jitter": (
            args.train_saturation_jitter
            if args.train_saturation_jitter is not None
            else config_values.get("train_saturation_jitter")
        ),
        "train_hue_jitter": (
            args.train_hue_jitter if args.train_hue_jitter is not None else config_values.get("train_hue_jitter")
        ),
        "early_stopping_patience": (
            args.early_stopping_patience
            if args.early_stopping_patience is not None
            else config_values.get("early_stopping_patience")
        ),
    }
    if resolved["model"] is None:
        raise ValueError("Model must be provided via --model or config file")
    if resolved["data_root"] is None:
        raise ValueError("Data root must be provided via --data-root or config file")
    return resolved


def build_optimizer(name: str, model, lr: float, weight_decay: float | None = None, momentum: float | None = None):
    name = name.lower()
    if name == "adam":
        kwargs = {"lr": lr}
        if weight_decay is not None:
            kwargs["weight_decay"] = weight_decay
        return torch.optim.Adam(model.parameters(), **kwargs)
    if name == "adamw":
        kwargs = {"lr": lr}
        if weight_decay is not None:
            kwargs["weight_decay"] = weight_decay
        return torch.optim.AdamW(model.parameters(), **kwargs)
    if name == "rmsprop":
        return torch.optim.RMSprop(
            model.parameters(),
            lr=lr,
            weight_decay=1e-8 if weight_decay is None else weight_decay,
            momentum=0.999 if momentum is None else momentum,
            foreach=True,
        )
    if name == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=0.0 if momentum is None else momentum,
            weight_decay=0.0 if weight_decay is None else weight_decay,
        )
    raise ValueError(f"Unsupported optimizer '{name}'")


def build_scheduler(protocol: dict, optimizer):
    name = protocol.get("scheduler")
    if not name:
        return None
    name = name.lower()
    if name == "cosineannealinglr":
        t_max = protocol.get("scheduler_t_max")
        eta_min = protocol.get("scheduler_eta_min")
        if t_max is None or eta_min is None:
            raise ValueError("CosineAnnealingLR requires scheduler_t_max and scheduler_eta_min")
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=t_max,
            eta_min=eta_min,
        )
    if name == "reducelronplateau":
        patience = protocol.get("scheduler_patience", 10)
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            patience=patience,
        )
    if name == "polynomiallr":
        power = protocol.get("scheduler_power")
        if power is None:
            power = 0.9
        return torch.optim.lr_scheduler.PolynomialLR(
            optimizer,
            total_iters=protocol["epochs"],
            power=power,
        )
    raise ValueError(f"Unsupported scheduler '{name}'")


def main():
    args = parse_args()
    resolved = resolve_args(args)
    protocol = build_protocol(
        image_size=resolved["image_size"],
        batch_size=resolved["batch_size"],
        num_workers=resolved["num_workers"],
        epochs=resolved["epochs"],
        optimizer=resolved["optimizer"],
        weight_decay=resolved["weight_decay"],
        momentum=resolved["momentum"],
        scheduler=resolved["scheduler"],
        scheduler_t_max=resolved["scheduler_t_max"],
        scheduler_eta_min=resolved["scheduler_eta_min"],
        scheduler_patience=resolved["scheduler_patience"],
        scheduler_power=resolved["scheduler_power"],
        lr=resolved["lr"],
        loss=resolved["loss"],
        loss_wb=resolved["loss_wb"],
        loss_wd=resolved["loss_wd"],
        seed=resolved["seed"],
        device=resolved["device"],
        base_channels=resolved["base_channels"],
        pretrained_path=resolved["pretrained_path"],
        dataset_name=resolved["dataset_name"],
        train_normalize_mode=resolved["train_normalize_mode"],
        eval_normalize_mode=resolved["eval_normalize_mode"],
        train_hflip_prob=resolved["train_hflip_prob"],
        train_vflip_prob=resolved["train_vflip_prob"],
        train_rotation_prob=resolved["train_rotation_prob"],
        train_rotation_min_deg=resolved["train_rotation_min_deg"],
        train_rotation_max_deg=resolved["train_rotation_max_deg"],
        train_color_jitter=resolved["train_color_jitter"],
        train_brightness_jitter=resolved["train_brightness_jitter"],
        train_contrast_jitter=resolved["train_contrast_jitter"],
        train_saturation_jitter=resolved["train_saturation_jitter"],
        train_hue_jitter=resolved["train_hue_jitter"],
        early_stopping_patience=resolved["early_stopping_patience"],
    )
    set_seed(protocol["seed"])

    train_loader = create_dataloader(
        root=resolved["data_root"],
        split="train",
        image_size=protocol["image_size"],
        batch_size=protocol["batch_size"],
        num_workers=protocol["num_workers"],
        shuffle=True,
        normalize_mode=protocol["train_normalize_mode"],
        dataset_name=protocol["dataset_name"],
        hflip_prob=protocol["train_hflip_prob"],
        vflip_prob=protocol["train_vflip_prob"],
        rotation_prob=protocol["train_rotation_prob"],
        rotation_min_deg=protocol["train_rotation_min_deg"],
        rotation_max_deg=protocol["train_rotation_max_deg"],
        color_jitter=protocol["train_color_jitter"],
        brightness_jitter=protocol["train_brightness_jitter"],
        contrast_jitter=protocol["train_contrast_jitter"],
        saturation_jitter=protocol["train_saturation_jitter"],
        hue_jitter=protocol["train_hue_jitter"],
    )
    val_loader = create_dataloader(
        root=resolved["data_root"],
        split="val",
        image_size=protocol["image_size"],
        batch_size=protocol["batch_size"],
        num_workers=protocol["num_workers"],
        shuffle=False,
        normalize_mode=protocol["eval_normalize_mode"],
        dataset_name=protocol["dataset_name"],
    )

    model = build_model(
        resolved["model"],
        base_channels=protocol["base_channels"],
        image_size=protocol["image_size"],
        pretrained_path=protocol["pretrained_path"],
    )
    model.to(protocol["device"])
    optimizer = build_optimizer(
        protocol["optimizer"],
        model,
        protocol["lr"],
        weight_decay=protocol["weight_decay"],
        momentum=protocol["momentum"],
    )
    scheduler = build_scheduler(protocol, optimizer)
    loss_fn = build_loss(protocol["loss"], wb=protocol["loss_wb"], wd=protocol["loss_wd"])

    output_dir = Path(resolved["output_dir"]) / resolved["model"]
    save_json(protocol, output_dir / "protocol.json")
    train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        epochs=protocol["epochs"],
        device=protocol["device"],
        output_dir=output_dir,
        threshold=protocol["threshold"],
        early_stopping_patience=protocol["early_stopping_patience"],
        protocol=protocol,
    )


if __name__ == "__main__":
    main()
