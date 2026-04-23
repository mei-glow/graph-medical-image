from __future__ import annotations

import copy
from types import SimpleNamespace
from typing import Callable, Dict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from semantic_attention.models.unet import UNet as Unet
from semantic_attention.models.swin_unet import SwinUnet as SwinUnet
MODEL_REGISTRY: Dict[str, Callable[..., nn.Module]] = {}


def register_model(name: str):
    def decorator(factory: Callable[..., nn.Module]):
        if name in MODEL_REGISTRY:
            raise ValueError(f"Model '{name}' is already registered")
        MODEL_REGISTRY[name] = factory
        return factory

    return decorator


def list_models():
    return sorted(MODEL_REGISTRY)


def build_model(name: str, **kwargs) -> nn.Module:
    if name not in MODEL_REGISTRY:
        available = ", ".join(list_models())
        raise KeyError(f"Unknown model '{name}'. Available models: {available}")
    return MODEL_REGISTRY[name](**kwargs)


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = ConvBlock(out_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        diff_y = skip.size(2) - x.size(2)
        diff_x = skip.size(3) - x.size(3)
        if diff_x != 0 or diff_y != 0:
            x = nn.functional.pad(
                x,
                [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2],
            )
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class SimpleUNet(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 1, base_channels: int = 32):
        super().__init__()
        self.enc1 = ConvBlock(in_channels, base_channels)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ConvBlock(base_channels, base_channels * 2)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = ConvBlock(base_channels * 2, base_channels * 4)
        self.pool3 = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(base_channels * 4, base_channels * 8)
        self.up3 = UpBlock(base_channels * 8, base_channels * 4, base_channels * 4)
        self.up2 = UpBlock(base_channels * 4, base_channels * 2, base_channels * 2)
        self.up1 = UpBlock(base_channels * 2, base_channels, base_channels)
        self.head = nn.Conv2d(base_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.enc1(x)
        x2 = self.enc2(self.pool1(x1))
        x3 = self.enc3(self.pool2(x2))
        x4 = self.bottleneck(self.pool3(x3))
        x = self.up3(x4, x3)
        x = self.up2(x, x2)
        x = self.up1(x, x1)
        return self.head(x)


class EGEUNetAdapter(nn.Module):
    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.backbone(x)
        if isinstance(output, tuple):
            output = output[-1]

        if not isinstance(output, torch.Tensor):
            raise TypeError("EGE-UNet must return a tensor or a tuple ending with a tensor")

        output = output.clamp(min=1e-6, max=1.0 - 1e-6)
        return torch.logit(output)


class SigmoidToLogitsAdapter(nn.Module):
    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.backbone(x)
        if not isinstance(output, torch.Tensor):
            raise TypeError("Expected tensor output for sigmoid adapter")
        output = output.clamp(min=1e-6, max=1.0 - 1e-6)
        return torch.logit(output)


class TransFuseAdapter(nn.Module):
    def __init__(self, backbone: nn.Module, model_input_size: tuple[int, int] = (192, 256)):
        super().__init__()
        self.backbone = backbone
        self.model_input_size = model_input_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_hw = x.shape[-2:]
        if original_hw != self.model_input_size:
            x = nn.functional.interpolate(
                x,
                size=self.model_input_size,
                mode="bilinear",
                align_corners=True,
            )

        output = self.backbone(x)
        if isinstance(output, (tuple, list)):
            resized = []
            for item in output:
                if not isinstance(item, torch.Tensor):
                    raise TypeError("TransFuse tuple outputs must be tensors")
                if item.shape[-2:] != original_hw:
                    item = nn.functional.interpolate(
                        item,
                        size=original_hw,
                        mode="bilinear",
                        align_corners=True,
                    )
                resized.append(item)
            return tuple(resized)
        if not isinstance(output, torch.Tensor):
            raise TypeError("TransFuse must return a tensor or a tuple/list beginning with a tensor")

        if output.shape[-2:] != original_hw:
            output = nn.functional.interpolate(
                output,
                size=original_hw,
                mode="bilinear",
                align_corners=True,
            )
        return output


class H2FormerInputAdapter(nn.Module):
    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] != 3:
            raise ValueError("H2Former adapter expects 3-channel RGB input")
        extra = x.mean(dim=1, keepdim=True)
        x4 = torch.cat([x, extra], dim=1)
        return self.backbone(x4)


def _load_partial_state_dict(model: nn.Module, state_dict: dict) -> None:
    model_dict = model.state_dict()
    filtered = {}
    for key, value in state_dict.items():
        if key in model_dict and model_dict[key].shape == value.shape:
            filtered[key] = value
    model.load_state_dict(filtered, strict=False)


def _build_swin_unet_config(
    image_size: int,
    in_channels: int,
    pretrained_path: str | None,
):
    return SimpleNamespace(
        DATA=SimpleNamespace(IMG_SIZE=image_size),
        MODEL=SimpleNamespace(
            PRETRAIN_CKPT=pretrained_path,
            DROP_RATE=0.0,
            DROP_PATH_RATE=0.1,
            SWIN=SimpleNamespace(
                PATCH_SIZE=4,
                IN_CHANS=in_channels,
                EMBED_DIM=96,
                DEPTHS=[2, 2, 2, 2],
                NUM_HEADS=[3, 6, 12, 24],
                WINDOW_SIZE=7,
                MLP_RATIO=4.0,
                QKV_BIAS=True,
                QK_SCALE=None,
                APE=False,
                PATCH_NORM=True,
            ),
        ),
        TRAIN=SimpleNamespace(USE_CHECKPOINT=False),
    )


@register_model("simple_unet")
def build_simple_unet(
    in_channels: int = 3,
    out_channels: int = 1,
    base_channels: int = 32,
    **_: object,
) -> nn.Module:
    return SimpleUNet(
        in_channels=in_channels,
        out_channels=out_channels,
        base_channels=base_channels,
    )


@register_model("unet")
def build_unet(
    in_channels: int = 3,
    out_channels: int = 1,
    bilinear: bool = False,
    **_: object,
):
    return Unet(
        n_channels=in_channels,
        n_classes=out_channels,
        bilinear=bilinear,
    )


@register_model("ege_unet")
def build_ege_unet(
    in_channels: int = 3,
    out_channels: int = 1,
    **_: object,
):
    from semantic_attention.models.egeunet import EGEUNet

    return EGEUNet(
        num_classes=out_channels,
        input_channels=in_channels,
        gt_ds=True,
    )


@register_model("attention_unet")
def build_attention_unet(in_channels: int = 3, out_channels: int = 1, **_: object):
    from semantic_attention.models.attention_unet import AttU_Net
    return AttU_Net(img_ch=in_channels, output_ch=out_channels)

@register_model("r2attention_unet")
def build_r2attention_unet(in_channels: int = 3, out_channels: int = 1, **_: object):
    from semantic_attention.models.attention_unet import R2AttU_Net
    return R2AttU_Net(img_ch=in_channels, output_ch=out_channels)

def _build_swin_unet_impl(
    in_channels: int = 3,
    out_channels: int = 2,
    image_size: int = 256,
    pretrained_path: str | None = None,
    **_: object,
):
    config = _build_swin_unet_config(
        image_size=image_size,
        in_channels=in_channels,
        pretrained_path=pretrained_path,
    )
    model = SwinUnet(config=config, img_size=image_size, num_classes=out_channels)
    if pretrained_path:
        model.load_from(config)
    return model


def _resolve_transunet_pretrained(pretrained_path: str | None) -> str | None:
    if pretrained_path:
        return pretrained_path
    default_path = (
        Path(__file__).resolve().parent
        / "transunet"
        / "vit_checkpoint"
        / "imagenet21k"
        / "R50+ViT-B_16.npz"
    )
    if default_path.exists():
        return str(default_path)
    return None


@register_model("swin_unet")
def build_swin_unet(
    in_channels: int = 3,
    out_channels: int = 2,
    image_size: int = 256,
    pretrained_path: str | None = None,
    **kwargs: object,
):
    return _build_swin_unet_impl(
        in_channels=in_channels,
        out_channels=out_channels,
        image_size=image_size,
        pretrained_path=pretrained_path,
        **kwargs,
    )


@register_model("fat_net")
def build_fat_net(
    in_channels: int = 3,
    out_channels: int = 1,
    image_size: int = 224,
    pretrained_path: str | None = None,
    **_: object,
):
    if image_size != 224:
        raise ValueError("FAT-Net expects image_size=224 because its DeiT branch is hard-coded to 14x14 tokens.")

    from semantic_attention.models.fatnet import FAT_Net

    local_deit_pretrained = pretrained_path
    if local_deit_pretrained is None:
        default_path = (
            Path(__file__).resolve().parent
            / "fatnet"
            / "pretrained"
            / "facebook"
            / "deit-tiny-distilled-patch16-224.bin"
        )
        if default_path.exists():
            local_deit_pretrained = str(default_path)

    local_resnet_pretrained = (
        Path(__file__).resolve().parent
        / "fatnet"
        / "pretrained"
        / "resnet34-b627a593.pth"
    )

    return FAT_Net(
        n_channels=in_channels,
        n_classes=out_channels,
        deit_pretrained_path=local_deit_pretrained,
        resnet_pretrained_path=str(local_resnet_pretrained) if local_resnet_pretrained.exists() else None,
        pretrained_backbone=True,
    )


@register_model("transunet")
def build_transunet(
    in_channels: int = 3,
    out_channels: int = 2,
    image_size: int = 224,
    pretrained_path: str | None = None,
    **_: object,
):
    from semantic_attention.models.transunet import CONFIGS, VisionTransformer

    config = copy.deepcopy(CONFIGS["R50-ViT-B_16"])
    config.n_classes = out_channels
    config.activation = "sigmoid"
    config.patches.grid = (image_size // 16, image_size // 16)

    model = VisionTransformer(config, img_size=image_size, num_classes=out_channels)
    resolved_pretrained = _resolve_transunet_pretrained(pretrained_path)
    if resolved_pretrained is not None:
        checkpoint_path = Path(resolved_pretrained)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"TransUNet pretrained checkpoint not found: {checkpoint_path}")
        model.load_from(np.load(str(checkpoint_path)))
    return model


@register_model("malunet")
def build_malunet(
    in_channels: int = 3,
    out_channels: int = 1,
    **_: object,
):
    from semantic_attention.models.malunet import MALUNet

    backbone = MALUNet(
        num_classes=out_channels,
        input_channels=in_channels,
        bridge=True,
    )
    return SigmoidToLogitsAdapter(backbone)


def _resolve_vm_unet_pretrained(pretrained_path: str | None) -> str | None:
    if pretrained_path:
        return pretrained_path
    weights_dir = Path(__file__).resolve().parent / "vm_unet" / "pre_trained_weights"
    candidates = [
        weights_dir / "vmamba_small_e238_ema.pth",
        weights_dir / "vmamba_tiny_e292.pth",
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return None


@register_model("vm_unet")
def build_vm_unet(
    in_channels: int = 3,
    out_channels: int = 1,
    pretrained_path: str | None = None,
    **_: object,
):
    from semantic_attention.models.vm_unet.vmunet import VMUNet

    resolved_pretrained = _resolve_vm_unet_pretrained(pretrained_path)
    backbone = VMUNet(
        input_channels=in_channels,
        num_classes=out_channels,
        load_ckpt_path=resolved_pretrained,
    )
    if resolved_pretrained is not None:
        backbone.load_from()
    return SigmoidToLogitsAdapter(backbone)


@register_model("ultralightvmunet")
@register_model("ultralight_vm_unet")
def build_ultralightvmunet(
    in_channels: int = 3,
    out_channels: int = 1,
    **_: object,
):
    from semantic_attention.models.ultralightvmunet.UltraLight_VM_UNet import UltraLight_VM_UNet

    backbone = UltraLight_VM_UNet(
        num_classes=out_channels,
        input_channels=in_channels,
        bridge=True,
    )
    return SigmoidToLogitsAdapter(backbone)


@register_model("skinmamba")
def build_skinmamba(
    **_: object,
):
    from semantic_attention.models.skinmamba.SkinMamba import SkinMamba

    backbone = SkinMamba()
    return SigmoidToLogitsAdapter(backbone)


@register_model("h2former")
def build_h2former(
    out_channels: int = 2,
    image_size: int = 512,
    pretrained_path: str | None = None,
    **_: object,
):
    from semantic_attention.models.H2Former import res34_swin_MS

    model = res34_swin_MS(image_size=image_size, num_class=out_channels)

    resolved_pretrained = pretrained_path
    if resolved_pretrained is None:
        default_path = (
            Path(__file__).resolve().parent
            / "H2Former"
            / "pretrained"
            / "resnet34-b627a593.pth"
        )
        if default_path.exists():
            resolved_pretrained = str(default_path)

    if resolved_pretrained is not None:
        checkpoint_path = Path(resolved_pretrained)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"H2Former pretrained checkpoint not found: {checkpoint_path}")
        state = torch.load(checkpoint_path, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        if not isinstance(state, dict):
            raise TypeError("Unsupported H2Former pretrained checkpoint format")
        cleaned_state = {}
        for key, value in state.items():
            new_key = key.replace("module.", "", 1) if key.startswith("module.") else key
            if new_key.startswith("resnet."):
                new_key = new_key.replace("resnet.", "", 1)
            cleaned_state[new_key] = value
        _load_partial_state_dict(model, cleaned_state)

    return H2FormerInputAdapter(model)


@register_model("transfuse")
def build_transfuse(
    out_channels: int = 1,
    image_size: int = 256,
    pretrained_path: str | None = None,
    **_: object,
):
    from semantic_attention.models.TransFuse import TransFuse_S

    pretrained_dir = Path(__file__).resolve().parent / "TransFuse" / "pretrained"
    required_files = [
        pretrained_dir / "deit_small_patch16_224-cd65a155.pth",
        pretrained_dir / "resnet34-333f7ec4.pth",
    ]
    use_pretrained = all(path.exists() for path in required_files)

    if pretrained_path is not None:
        checkpoint_path = Path(pretrained_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"TransFuse checkpoint not found: {checkpoint_path}")

    model = TransFuse_S(
        num_classes=out_channels,
        pretrained=use_pretrained,
    )
    return TransFuseAdapter(model)
