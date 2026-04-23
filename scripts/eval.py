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
from semantic_attention.engine import evaluate
from semantic_attention.models import build_model, list_models
from semantic_attention.utils.checkpoint import load_checkpoint
from semantic_attention.utils.io import save_json
from semantic_attention.utils.logger import SimpleLogger


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a segmentation model on ISIC 2018")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--model", type=str, default=None, choices=list_models())
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--split", type=str, default=None, choices=["train", "val", "test"])
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--base-channels", type=int, default=None)
    parser.add_argument("--pretrained-path", type=str, default=None)
    parser.add_argument("--dataset-name", type=str, default=None)
    parser.add_argument("--eval-normalize-mode", type=str, default=None)
    return parser.parse_args()


def resolve_args(args):
    config_values = {}
    if args.config:
        config_values = flatten_config_sections(load_config_file(args.config))

    resolved = {
        "model": args.model or config_values.get("model"),
        "data_root": args.data_root or config_values.get("data_root"),
        "checkpoint": args.checkpoint or config_values.get("checkpoint"),
        "split": args.split or config_values.get("split") or "test",
        "image_size": args.image_size if args.image_size is not None else config_values.get("image_size"),
        "batch_size": args.batch_size if args.batch_size is not None else config_values.get("batch_size"),
        "num_workers": args.num_workers if args.num_workers is not None else config_values.get("num_workers"),
        "threshold": args.threshold if args.threshold is not None else config_values.get("threshold"),
        "device": args.device or config_values.get("device"),
        "output_dir": args.output_dir or config_values.get("output_dir") or "outputs/eval",
        "base_channels": args.base_channels if args.base_channels is not None else config_values.get("base_channels"),
        "pretrained_path": args.pretrained_path or config_values.get("pretrained_path"),
        "dataset_name": args.dataset_name or config_values.get("dataset_name"),
        "eval_normalize_mode": args.eval_normalize_mode or config_values.get("eval_normalize_mode"),
    }
    if resolved["model"] is None:
        raise ValueError("Model must be provided via --model or config file")
    if resolved["data_root"] is None:
        raise ValueError("Data root must be provided via --data-root or config file")
    if resolved["checkpoint"] is None:
        raise ValueError("Checkpoint must be provided via --checkpoint or config file")
    return resolved


def main():
    args = parse_args()
    resolved = resolve_args(args)
    protocol = build_protocol(
        image_size=resolved["image_size"],
        batch_size=resolved["batch_size"],
        num_workers=resolved["num_workers"],
        threshold=resolved["threshold"],
        device=resolved["device"],
        base_channels=resolved["base_channels"],
        pretrained_path=resolved["pretrained_path"],
        dataset_name=resolved["dataset_name"],
        eval_normalize_mode=resolved["eval_normalize_mode"],
    )
    loader = create_dataloader(
        root=resolved["data_root"],
        split=resolved["split"],
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
    ).to(protocol["device"])
    load_checkpoint(model, resolved["checkpoint"], device=protocol["device"])

    run_dir = Path(resolved["output_dir"]) / resolved["model"] / resolved["split"]
    logger = SimpleLogger(run_dir / "eval.log")
    logger.log(f"Loading checkpoint: {resolved['checkpoint']}")
    logger.log(f"Protocol: {protocol}")
    metrics = evaluate(
        model=model,
        loader=loader,
        device=protocol["device"],
        threshold=protocol["threshold"],
        save_predictions=args.save_predictions,
        prediction_dir=run_dir,
        logger=logger,
    )
    save_json(protocol, run_dir / "protocol.json")
    save_json(metrics, run_dir / "metrics.json")
    logger.log(f"Saved metrics to: {run_dir / 'metrics.json'}")
    print(metrics)


if __name__ == "__main__":
    main()
