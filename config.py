from __future__ import annotations

import tomllib
from pathlib import Path

import torch


DEFAULT_PROTOCOL = {
    "image_size": 256,
    "batch_size": 16,
    "num_workers": 0,
    "epochs": 100,
    "optimizer": "adamw",
    "weight_decay": None,
    "momentum": None,
    "scheduler": None,
    "scheduler_t_max": None,
    "scheduler_eta_min": None,
    "scheduler_patience": None,
    "scheduler_power": None,
    "lr": 1e-4,
    "loss": "bce_dice",
    "loss_wb": 1.0,
    "loss_wd": 1.0,
    "threshold": 0.5,
    "seed": 42,
    "base_channels": 32,
    "early_stopping_patience": 20,
    "pretrained_path": None,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "dataset_name": "isic18",
    "train_normalize_mode": "imagenet",
    "eval_normalize_mode": "imagenet",
    "train_hflip_prob": 0.0,
    "train_vflip_prob": 0.0,
    "train_rotation_prob": 0.0,
    "train_rotation_min_deg": 0.0,
    "train_rotation_max_deg": 0.0,
    "train_color_jitter": False,
    "train_brightness_jitter": None,
    "train_contrast_jitter": None,
    "train_saturation_jitter": None,
    "train_hue_jitter": None,
}


def build_protocol(**overrides):
    protocol = dict(DEFAULT_PROTOCOL)
    for key, value in overrides.items():
        if value is not None:
            protocol[key] = value
    return protocol


def load_config_file(path: str | Path) -> dict:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("rb") as handle:
        data = tomllib.load(handle)
    if not isinstance(data, dict):
        raise TypeError("Config file must decode to a dictionary")
    return data


def flatten_config_sections(config: dict) -> dict:
    flat = {}
    for key, value in config.items():
        if isinstance(value, dict):
            flat.update(value)
        else:
            flat[key] = value
    return flat
