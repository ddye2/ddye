# -*- coding: utf-8 -*-
"""WHU building image/mask loading and synchronized augmentation."""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from torch.utils.data import Dataset

IMG_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _find_dir(base: Path, candidates: tuple[str, ...]) -> Path:
    for name in candidates:
        candidate = base / name
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(f"Cannot find one of {candidates} under {base}")


def _list_pairs(img_dir: Path, mask_dir: Path) -> list[tuple[Path, Path]]:
    images = {p.stem: p for p in img_dir.iterdir()
              if p.is_file() and p.suffix.lower() in IMG_EXTS}
    masks = {p.stem: p for p in mask_dir.iterdir()
             if p.is_file() and p.suffix.lower() in IMG_EXTS}
    missing_masks = sorted(images.keys() - masks.keys())
    missing_images = sorted(masks.keys() - images.keys())
    if missing_masks or missing_images:
        raise RuntimeError(
            f"Unpaired files: missing masks={len(missing_masks)}, "
            f"missing images={len(missing_images)}")
    pairs = [(images[key], masks[key]) for key in sorted(images)]
    if not pairs:
        raise RuntimeError(f"No image/mask pairs found in {img_dir.parent}")
    return pairs


def _resize(image: Image.Image, mask: Image.Image, size: int):
    image = image.resize((size, size), Image.Resampling.BILINEAR)
    mask = mask.resize((size, size), Image.Resampling.NEAREST)
    return image, mask


def _scale_crop_pad(image: Image.Image, mask: Image.Image, size: int,
                    scale_min: float, scale_max: float):
    """Random scale followed by a fixed-size crop (or padded crop)."""
    scale = random.uniform(scale_min, scale_max)
    new_w = max(1, round(image.width * scale))
    new_h = max(1, round(image.height * scale))
    image = image.resize((new_w, new_h), Image.Resampling.BILINEAR)
    mask = mask.resize((new_w, new_h), Image.Resampling.NEAREST)

    pad_w, pad_h = max(0, size - new_w), max(0, size - new_h)
    if pad_w or pad_h:
        left = random.randint(0, pad_w)
        top = random.randint(0, pad_h)
        right, bottom = pad_w - left, pad_h - top
        image_np = np.pad(
            np.asarray(image), ((top, bottom), (left, right), (0, 0)),
            mode="reflect")
        mask_np = np.pad(
            np.asarray(mask), ((top, bottom), (left, right)),
            mode="constant", constant_values=0)
        image, mask = Image.fromarray(image_np), Image.fromarray(mask_np)

    top = random.randint(0, image.height - size)
    left = random.randint(0, image.width - size)
    box = (left, top, left + size, top + size)
    return image.crop(box), mask.crop(box)


def _photometric(image: Image.Image) -> Image.Image:
    # Mild perturbations preserve roof semantics while improving robustness.
    ops = [
        lambda im: ImageEnhance.Brightness(im).enhance(random.uniform(0.8, 1.2)),
        lambda im: ImageEnhance.Contrast(im).enhance(random.uniform(0.8, 1.2)),
        lambda im: ImageEnhance.Color(im).enhance(random.uniform(0.8, 1.2)),
    ]
    random.shuffle(ops)
    for op in ops:
        image = op(image)
    if random.random() < 0.10:
        image = image.filter(ImageFilter.GaussianBlur(random.uniform(0.1, 1.2)))
    return image


class BuildingDataset(Dataset):
    """Binary building dataset with none/basic/standard augmentation.

    ``basic`` applies synchronized rotations/flips. ``standard`` additionally
    applies random scale/crop and mild image-only perturbations. Boolean values
    remain accepted for compatibility with the old evaluation scripts.
    """

    def __init__(self, root, split="train", img_size=448, augment=None,
                 scale_min=0.75, scale_max=1.25):
        split_dir = Path(root) / split
        img_dir = _find_dir(split_dir, ("images", "Image", "image", "img"))
        mask_dir = _find_dir(split_dir, ("masks", "Mask", "mask", "label", "labels"))
        self.pairs = _list_pairs(img_dir, mask_dir)
        self.img_size = int(img_size)
        if augment is None:
            augment = "basic" if split == "train" else "none"
        elif isinstance(augment, bool):
            augment = "basic" if augment else "none"
        if augment not in {"none", "basic", "standard"}:
            raise ValueError(f"Unknown augmentation preset: {augment}")
        self.augment = augment if split == "train" else "none"
        if not (0 < scale_min <= scale_max):
            raise ValueError("Require 0 < scale_min <= scale_max")
        self.scale_min, self.scale_max = scale_min, scale_max

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]
        with Image.open(img_path) as handle:
            image = handle.convert("RGB")
        with Image.open(mask_path) as handle:
            mask = handle.convert("L")
        if image.size != mask.size:
            raise RuntimeError(
                f"Image/mask size mismatch: {img_path} {image.size} vs "
                f"{mask_path} {mask.size}")

        if self.augment == "standard":
            image, mask = _scale_crop_pad(
                image, mask, self.img_size, self.scale_min, self.scale_max)
        elif image.size != (self.img_size, self.img_size):
            image, mask = _resize(image, mask, self.img_size)

        if self.augment in {"basic", "standard"}:
            k = random.randint(0, 3)
            if k:
                image = image.rotate(90 * k, resample=Image.Resampling.BILINEAR)
                mask = mask.rotate(90 * k, resample=Image.Resampling.NEAREST)
            if random.random() < 0.5:
                image, mask = ImageOps.mirror(image), ImageOps.mirror(mask)
            if random.random() < 0.5:
                image, mask = ImageOps.flip(image), ImageOps.flip(mask)

        if self.augment == "standard":
            image = _photometric(image)

        image_np = np.asarray(image, dtype=np.float32) / 255.0
        if self.augment == "standard" and random.random() < 0.10:
            noise = np.random.normal(0.0, 0.015, image_np.shape).astype(np.float32)
            image_np = np.clip(image_np + noise, 0.0, 1.0)
        mask_np = (np.asarray(mask) > 127).astype(np.float32)

        image_np = (image_np - MEAN) / STD
        image_tensor = torch.from_numpy(image_np.transpose(2, 0, 1).copy())
        mask_tensor = torch.from_numpy(mask_np[None].copy())
        return image_tensor, mask_tensor


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)
