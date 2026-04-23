from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as TF
from torchvision import transforms as T
from torch.utils.data import DataLoader, Dataset
import random
from scipy import ndimage
from scipy.ndimage import zoom


SPLIT_DIRS = {
    "train": (
        "ISIC2018_Task1-2_Training_Input",
        "ISIC2018_Task1_Training_GroundTruth",
    ),
    "val": (
        "ISIC2018_Task1-2_Validation_Input",
        "ISIC2018_Task1_Validation_GroundTruth",
    ),
    "test": (
        "ISIC2018_Task1-2_Test_Input",
        "ISIC2018_Task1_Test_GroundTruth",
    ),
}


@dataclass(frozen=True)
class SampleRecord:
    image_path: Path
    mask_path: Path
    sample_id: str


def _list_images(image_dir: Path) -> Dict[str, Path]:
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    files = {}
    for path in image_dir.iterdir():
        if path.is_file() and path.suffix.lower() in valid_exts:
            files[path.stem] = path
    return files


def _find_mask(mask_dir: Path, sample_id: str) -> Path:
    candidates = [
        mask_dir / f"{sample_id}_segmentation.png",
        mask_dir / f"{sample_id}.png",
        mask_dir / f"{sample_id}_mask.png",
        mask_dir / f"{sample_id}.jpg",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No mask found for sample '{sample_id}' in {mask_dir}")


def _resize_image(image: Image.Image, size: int) -> Image.Image:
    return image.resize((size, size), Image.BILINEAR)


def _resize_mask(mask: Image.Image, size: int) -> Image.Image:
    return mask.resize((size, size), Image.NEAREST)


def _image_to_tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image, dtype=np.float32) / 255.0
    if array.ndim == 2:
        array = np.stack([array, array, array], axis=-1)
    array = (array - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array(
        [0.229, 0.224, 0.225], dtype=np.float32
    )
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def _ege_normalize_array(image: Image.Image, data_name: str, train: bool) -> np.ndarray:
    array = np.asarray(image, dtype=np.float32)
    if data_name == "isic18":
        mean, std = (157.561, 26.706) if train else (149.034, 32.022)
    elif data_name == "isic17":
        mean, std = (159.922, 28.871) if train else (148.429, 25.748)
    elif data_name == "isic18_82":
        mean, std = (156.2899, 26.5457) if train else (149.8485, 35.3346)
    else:
        raise ValueError(f"Unsupported EGE normalize dataset '{data_name}'")

    normalized = (array - mean) / std
    denom = float(normalized.max() - normalized.min())
    if denom <= 1e-8:
        normalized = np.zeros_like(normalized, dtype=np.float32)
    else:
        normalized = ((normalized - normalized.min()) / denom) * 255.0
    return normalized.astype(np.float32)


class _EGETransform:
    def __init__(
        self,
        image_size: int,
        data_name: str,
        train: bool,
        hflip_prob: float = 0.0,
        vflip_prob: float = 0.0,
        rotation_prob: float = 0.0,
        rotation_min_deg: float = 0.0,
        rotation_max_deg: float = 0.0,
    ):
        self.image_size = image_size
        self.data_name = data_name
        self.train = train
        self.hflip_prob = hflip_prob
        self.vflip_prob = vflip_prob
        self.rotation_prob = rotation_prob
        self.rotation_angle = random.uniform(rotation_min_deg, rotation_max_deg)

    def __call__(self, image: Image.Image, mask: Image.Image):
        image_array = _ege_normalize_array(image, data_name=self.data_name, train=self.train)
        mask_array = np.asarray(mask, dtype=np.float32)
        if mask_array.ndim == 2:
            mask_array = mask_array[..., None]

        image_tensor = torch.tensor(image_array).permute(2, 0, 1).float()
        mask_tensor = torch.tensor(mask_array).permute(2, 0, 1).float()

        if self.train and random.random() < self.hflip_prob:
            image_tensor, mask_tensor = TF.hflip(image_tensor), TF.hflip(mask_tensor)
        if self.train and random.random() < self.vflip_prob:
            image_tensor, mask_tensor = TF.vflip(image_tensor), TF.vflip(mask_tensor)
        if self.train and random.random() < self.rotation_prob:
            image_tensor = TF.rotate(image_tensor, self.rotation_angle)
            mask_tensor = TF.rotate(mask_tensor, self.rotation_angle)

        image_tensor = TF.resize(image_tensor, [self.image_size, self.image_size], antialias=True)
        mask_tensor = TF.resize(mask_tensor, [self.image_size, self.image_size])
        mask_tensor = (mask_tensor > 127).float()
        return image_tensor.contiguous(), mask_tensor.contiguous()


def _swin_random_rot_flip(image: np.ndarray, mask: np.ndarray):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k, axes=(0, 1))
    mask = np.rot90(mask, k, axes=(0, 1))
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    mask = np.flip(mask, axis=axis).copy()
    return image, mask


def _swin_random_rotate(image: np.ndarray, mask: np.ndarray, min_deg: float, max_deg: float):
    angle = np.random.randint(int(min_deg), int(max_deg) + 1)
    image = ndimage.rotate(image, angle, axes=(0, 1), order=3, reshape=False)
    mask = ndimage.rotate(mask, angle, axes=(0, 1), order=0, reshape=False)
    return image, mask


class _SwinTransform:
    def __init__(
        self,
        image_size: int,
        train: bool,
        rotation_min_deg: float = -20.0,
        rotation_max_deg: float = 20.0,
    ):
        self.image_size = image_size
        self.train = train
        self.rotation_min_deg = rotation_min_deg
        self.rotation_max_deg = rotation_max_deg

    def __call__(self, image: Image.Image, mask: Image.Image):
        image_array = np.asarray(image, dtype=np.float32) / 255.0
        mask_array = np.asarray(mask, dtype=np.float32)
        if mask_array.ndim == 3:
            mask_array = mask_array[..., 0]
        mask_array = (mask_array > 127).astype(np.float32)

        if self.train:
            if random.random() > 0.5:
                image_array, mask_array = _swin_random_rot_flip(image_array, mask_array)
            elif random.random() > 0.5:
                image_array, mask_array = _swin_random_rotate(
                    image_array,
                    mask_array,
                    self.rotation_min_deg,
                    self.rotation_max_deg,
                )

        height, width = image_array.shape[:2]
        if height != self.image_size or width != self.image_size:
            image_array = zoom(
                image_array,
                (self.image_size / height, self.image_size / width, 1),
                order=3,
            )
            mask_array = zoom(
                mask_array,
                (self.image_size / height, self.image_size / width),
                order=0,
            )

        image_tensor = torch.from_numpy(image_array.astype(np.float32)).permute(2, 0, 1).contiguous()
        mask_tensor = torch.from_numpy((mask_array > 0.5).astype(np.float32)).unsqueeze(0).contiguous()
        return image_tensor, mask_tensor


def _mask_to_tensor(mask: Image.Image) -> torch.Tensor:
    array = np.asarray(mask, dtype=np.float32)
    if array.ndim == 3:
        array = array[..., 0]
    array = (array > 127).astype(np.float32)
    return torch.from_numpy(array).unsqueeze(0).contiguous()


class ISIC2018Dataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        split: str,
        image_size: int = 256,
        normalize_mode: str = "imagenet",
        dataset_name: str = "isic18",
        hflip_prob: float = 0.0,
        vflip_prob: float = 0.0,
        rotation_prob: float = 0.0,
        rotation_min_deg: float = 0.0,
        rotation_max_deg: float = 0.0,
        color_jitter: bool = False,
        brightness_jitter: float | None = None,
        contrast_jitter: float | None = None,
        saturation_jitter: float | None = None,
        hue_jitter: float | None = None,
    ):
        if split not in SPLIT_DIRS:
            raise ValueError(f"Unknown split '{split}'. Expected one of {tuple(SPLIT_DIRS)}")

        self.root = Path(root)
        self.split = split
        self.image_size = image_size
        self.normalize_mode = normalize_mode
        self.dataset_name = dataset_name
        self.color_jitter = None
        if color_jitter and split == "train":
            self.color_jitter = T.ColorJitter(
                brightness=0.0 if brightness_jitter is None else brightness_jitter,
                contrast=0.0 if contrast_jitter is None else contrast_jitter,
                saturation=0.0 if saturation_jitter is None else saturation_jitter,
                hue=0.0 if hue_jitter is None else hue_jitter,
            )
        self.transform = None
        if normalize_mode == "ege":
            self.transform = _EGETransform(
                image_size=image_size,
                data_name=dataset_name,
                train=split == "train",
                hflip_prob=hflip_prob,
                vflip_prob=vflip_prob,
                rotation_prob=rotation_prob,
                rotation_min_deg=rotation_min_deg,
                rotation_max_deg=rotation_max_deg,
            )
        elif normalize_mode == "swin":
            self.transform = _SwinTransform(
                image_size=image_size,
                train=split == "train",
                rotation_min_deg=rotation_min_deg,
                rotation_max_deg=rotation_max_deg,
            )
        self.samples = self._build_index()

    def _build_index(self) -> List[SampleRecord]:
        image_dir_name, mask_dir_name = SPLIT_DIRS[self.split]
        image_dir = self.root / image_dir_name
        mask_dir = self.root / mask_dir_name

        if not image_dir.exists():
            raise FileNotFoundError(f"Missing image directory: {image_dir}")
        if not mask_dir.exists():
            raise FileNotFoundError(f"Missing mask directory: {mask_dir}")

        images = _list_images(image_dir)
        records = []
        for sample_id, image_path in sorted(images.items()):
            mask_path = _find_mask(mask_dir, sample_id)
            records.append(SampleRecord(image_path=image_path, mask_path=mask_path, sample_id=sample_id))

        if not records:
            raise RuntimeError(f"No samples found in {image_dir}")
        return records

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        record = self.samples[index]

        image = Image.open(record.image_path).convert("RGB")
        mask = Image.open(record.mask_path).convert("L")

        original_size = image.size[::-1]

        if self.transform is not None:
            image_tensor, mask_tensor = self.transform(image, mask)
        else:
            if self.color_jitter is not None:
                image = self.color_jitter(image)
            image = _resize_image(image, self.image_size)
            mask = _resize_mask(mask, self.image_size)
            image_tensor = _image_to_tensor(image)
            mask_tensor = _mask_to_tensor(mask)

        meta = {
            "id": record.sample_id,
            "image_path": str(record.image_path),
            "mask_path": str(record.mask_path),
            "original_size": original_size,
        }
        return image_tensor, mask_tensor, meta


def create_dataloader(
    root: str | Path,
    split: str,
    image_size: int = 256,
    batch_size: int = 8,
    num_workers: int = 0,
    shuffle: bool | None = None,
    normalize_mode: str = "imagenet",
    dataset_name: str = "isic18",
    hflip_prob: float = 0.0,
    vflip_prob: float = 0.0,
    rotation_prob: float = 0.0,
    rotation_min_deg: float = 0.0,
    rotation_max_deg: float = 0.0,
    color_jitter: bool = False,
    brightness_jitter: float | None = None,
    contrast_jitter: float | None = None,
    saturation_jitter: float | None = None,
    hue_jitter: float | None = None,
) -> DataLoader:
    dataset = ISIC2018Dataset(
        root=root,
        split=split,
        image_size=image_size,
        normalize_mode=normalize_mode,
        dataset_name=dataset_name,
        hflip_prob=hflip_prob,
        vflip_prob=vflip_prob,
        rotation_prob=rotation_prob,
        rotation_min_deg=rotation_min_deg,
        rotation_max_deg=rotation_max_deg,
        color_jitter=color_jitter,
        brightness_jitter=brightness_jitter,
        contrast_jitter=contrast_jitter,
        saturation_jitter=saturation_jitter,
        hue_jitter=hue_jitter,
    )
    if shuffle is None:
        shuffle = split == "train"

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )
