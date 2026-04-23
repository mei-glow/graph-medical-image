from __future__ import annotations

import argparse
from pathlib import Path

import torch

from semantic_attention.config import DEFAULT_PROTOCOL, build_protocol
from semantic_attention.datasets import create_dataloader
from semantic_attention.engine import evaluate
from semantic_attention.models import build_model, list_models
from semantic_attention.utils.checkpoint import load_checkpoint
from semantic_attention.utils.io import save_csv, save_json


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark multiple model checkpoints on the same split")
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--image-size", type=int, default=DEFAULT_PROTOCOL["image_size"])
    parser.add_argument("--batch-size", type=int, default=DEFAULT_PROTOCOL["batch_size"])
    parser.add_argument("--num-workers", type=int, default=DEFAULT_PROTOCOL["num_workers"])
    parser.add_argument("--threshold", type=float, default=DEFAULT_PROTOCOL["threshold"])
    parser.add_argument("--device", type=str, default=DEFAULT_PROTOCOL["device"])
    parser.add_argument("--output", type=str, default="outputs/benchmark/results.csv")
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="Format: model_name=path/to/checkpoint.pt",
    )
    parser.add_argument("--base-channels", type=int, default=DEFAULT_PROTOCOL["base_channels"])
    parser.add_argument("--pretrained-path", type=str, default=DEFAULT_PROTOCOL["pretrained_path"])
    return parser.parse_args()


def _parse_run(spec: str):
    if "=" not in spec:
        raise ValueError(f"Invalid --run '{spec}'. Expected model=checkpoint")
    model_name, checkpoint = spec.split("=", 1)
    if model_name not in list_models():
        raise ValueError(f"Unknown model '{model_name}'")
    return model_name, checkpoint


def main():
    args = parse_args()
    protocol = build_protocol(
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        threshold=args.threshold,
        device=args.device,
        base_channels=args.base_channels,
        pretrained_path=args.pretrained_path,
    )
    loader = create_dataloader(
        root=args.data_root,
        split=args.split,
        image_size=protocol["image_size"],
        batch_size=protocol["batch_size"],
        num_workers=protocol["num_workers"],
        shuffle=False,
    )

    rows = []
    for spec in args.run:
        model_name, checkpoint_path = _parse_run(spec)
        model = build_model(
            model_name,
            base_channels=protocol["base_channels"],
            image_size=protocol["image_size"],
            pretrained_path=protocol["pretrained_path"],
        ).to(protocol["device"])
        load_checkpoint(model, checkpoint_path, device=protocol["device"])
        metrics = evaluate(model, loader, device=protocol["device"], threshold=protocol["threshold"])
        row = {"model": model_name, "checkpoint": checkpoint_path, **metrics}
        rows.append(row)
        print(row)

    output_path = Path(args.output)
    save_csv(rows, output_path)
    save_json(rows, output_path.with_suffix(".json"))
    save_json(protocol, output_path.with_name(output_path.stem + "_protocol.json"))


if __name__ == "__main__":
    main()
