# -*- coding: utf-8 -*-
"""Train a DINOv3 building segmenter with a reproducible U-Net-style loop."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import time
from functools import partial
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from dataset import BuildingDataset, seed_worker
from losses import BCEDiceBoundaryLoss, BCEDiceLoss
from segmenter import BuildingSegmenter
from models.decoders import DECODERS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dinov3-repo",
                        default=os.environ.get("DINO_DD_DINO_REPO"),
                        help="Local DINOv3 repository path; can also be set "
                             "with DINO_DD_DINO_REPO")
    parser.add_argument("--pretrained-dir", default=None,
                        help="Directory containing DINOv3 .pth weights; "
                             "defaults to dinov3/pretrained in the DINOv3 repository")
    parser.add_argument("--data-root", required=True,
                        help="Dataset root containing train/ and val/ (or test/)")
    parser.add_argument("--backbone", default="vits16", choices=["vits16", "vitl16_sat"])
    parser.add_argument("--decoder", default="dpt",
                        choices=list(DECODERS))
    parser.add_argument("--img-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=3e-5,
                        help="Decoder peak learning rate")
    parser.add_argument("--backbone-lr", type=float, default=3e-6,
                        help="Peak LR for unfrozen backbone blocks")
    parser.add_argument("--unfreeze-blocks", type=int, default=0,
                        help="Fine-tune the final N transformer blocks; 0 freezes DINOv3")
    parser.add_argument("--optimizer", choices=["adamw", "sgd"], default="adamw")
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-epochs", type=float, default=5.0)
    parser.add_argument("--min-lr-ratio", type=float, default=0.01)
    parser.add_argument("--loss", choices=["bce_dice", "bce_dice_boundary"],
                        default="bce_dice")
    parser.add_argument("--boundary-weight", type=float, default=0.4,
                        help="Used only by bce_dice_boundary")
    parser.add_argument("--augment", choices=["none", "basic", "standard"],
                        default="standard")
    parser.add_argument("--scale-min", type=float, default=0.75)
    parser.add_argument("--scale-max", type=float, default=1.25)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--persistent-workers", action=argparse.BooleanOptionalAction,
                        default=False,
                        help="Faster epoch startup, but worker augmentation RNG cannot be exactly resumed")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction,
                        default=False)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--amp-dtype", choices=["auto", "float16", "bfloat16"],
                        default="auto")
    parser.add_argument("--clip-grad", type=float, default=1.0)
    parser.add_argument("--early-stop", type=int, default=20,
                        help="Stop after N epochs without val-IoU improvement; 0 disables")
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--resume", default=None,
                        help="Full resume from a new last.pth; old checkpoints are weight-only")
    parser.add_argument("--init", default=None,
                        help="Initialize model weights only (for a new fine-tuning stage)")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


class BinaryMeter:
    def __init__(self, threshold: float):
        self.threshold = threshold
        self.tp = self.fp = self.fn = self.tn = 0.0

    @torch.no_grad()
    def update(self, logits: torch.Tensor, target: torch.Tensor) -> None:
        pred = torch.sigmoid(logits.float()) > self.threshold
        truth = target > 0.5
        self.tp += (pred & truth).sum().item()
        self.fp += (pred & ~truth).sum().item()
        self.fn += (~pred & truth).sum().item()
        self.tn += (~pred & ~truth).sum().item()

    def compute(self) -> dict[str, float]:
        eps = 1e-7
        precision = self.tp / (self.tp + self.fp + eps)
        recall = self.tp / (self.tp + self.fn + eps)
        iou = self.tp / (self.tp + self.fp + self.fn + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)
        accuracy = (self.tp + self.tn) / (self.tp + self.fp + self.fn + self.tn + eps)
        return {"iou": iou, "f1": f1, "precision": precision,
                "recall": recall, "accuracy": accuracy}


def set_seed(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        torch.backends.cudnn.benchmark = True


def rng_state(generator: torch.Generator) -> dict:
    state = {"python": random.getstate(), "numpy": np.random.get_state(),
             "torch": torch.get_rng_state(), "loader": generator.get_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _cpu_byte_tensor(value) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return torch.from_numpy(value.detach().cpu().numpy().copy()).to(torch.uint8)
    return torch.tensor(value, dtype=torch.uint8, device="cpu")


def restore_rng_state(state: dict | None, generator: torch.Generator) -> None:
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(_cpu_byte_tensor(state["torch"]))
    generator.set_state(_cpu_byte_tensor(state["loader"]))
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all([_cpu_byte_tensor(v) for v in state["cuda"]])


def cosine_warmup_lambda(step: int, warmup_steps: int, total_steps: int,
                         min_ratio: float) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return max((step + 1) / warmup_steps, min_ratio)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    progress = min(max(progress, 0.0), 1.0)
    return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))


def atomic_torch_save(payload: dict, path: Path) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temp)
    os.replace(temp, path)


def append_csv(path: Path, row: dict) -> None:
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def select_amp_dtype(args, device: torch.device) -> torch.dtype:
    if args.amp_dtype == "float16":
        return torch.float16
    if args.amp_dtype == "bfloat16":
        return torch.bfloat16
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def make_optimizer(args, model: BuildingSegmenter):
    groups = [{"params": model.decoder_params(), "lr": args.lr,
               "name": "decoder"}]
    backbone_params = model.backbone_params()
    if backbone_params:
        groups.append({"params": backbone_params, "lr": args.backbone_lr,
                       "name": "backbone"})
    if args.optimizer == "adamw":
        return torch.optim.AdamW(groups, weight_decay=args.weight_decay,
                                 betas=(0.9, 0.999))
    return torch.optim.SGD(groups, weight_decay=args.weight_decay, momentum=0.9,
                           nesterov=True)


def run_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module,
              device: torch.device, threshold: float, amp: bool, amp_dtype: torch.dtype,
              optimizer=None, scaler=None, scheduler=None, clip_grad: float = 0.0):
    training = optimizer is not None
    model.train(training)
    meter = BinaryMeter(threshold)
    loss_sum, seen = 0.0, 0
    context = torch.enable_grad if training else torch.no_grad
    with context():
        for images, masks in loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype,
                                enabled=amp and device.type == "cuda"):
                logits = model(images)
                loss = criterion(logits, masks)
            if training:
                scaler.scale(loss).backward()
                if clip_grad > 0:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad], clip_grad)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
            loss_sum += loss.item() * images.shape[0]
            seen += images.shape[0]
            meter.update(logits.detach(), masks)
    result = meter.compute()
    result["loss"] = loss_sum / max(seen, 1)
    return result


def main() -> None:
    args = parse_args()
    if not Path(args.data_root).is_dir():
        raise FileNotFoundError(
            f"Dataset root not found: {args.data_root}. "
            "Pass --data-root /path/to/whu on Linux.")
    if args.img_size % 16:
        raise ValueError("--img-size must be divisible by DINOv3 patch size 16")
    if not (0.0 < args.threshold < 1.0):
        raise ValueError("--threshold must be between 0 and 1")
    if not (0.0 <= args.min_lr_ratio <= 1.0):
        raise ValueError("--min-lr-ratio must be in [0, 1]")
    if args.loss == "bce_dice_boundary" and args.boundary_weight <= 0:
        raise ValueError("Boundary loss requires --boundary-weight > 0")
    if args.resume and args.init:
        raise ValueError("Use only one of --resume and --init")

    default_name = f"{args.backbone}_{args.decoder}_{args.img_size}_bs{args.batch_size}"
    out_dir = Path(args.out_dir or Path("outputs") / default_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but is not available")
    set_seed(args.seed, args.deterministic)
    use_amp = args.amp and device.type == "cuda"
    amp_dtype = select_amp_dtype(args, device)

    train_set = BuildingDataset(
        args.data_root, "train", args.img_size, args.augment,
        args.scale_min, args.scale_max)
    val_set = BuildingDataset(args.data_root, "val", args.img_size, "none")
    loader_generator = torch.Generator().manual_seed(args.seed)
    loader_args = dict(
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
        worker_init_fn=seed_worker,
        persistent_workers=args.persistent_workers and args.num_workers > 0)
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, drop_last=True,
        generator=loader_generator, **loader_args)
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False, drop_last=False,
        **loader_args)

    model = BuildingSegmenter(
        args.backbone, args.decoder, device=str(device),
        unfreeze_blocks=args.unfreeze_blocks,
        pretrained_dir=args.pretrained_dir,
        dinov3_repo=args.dinov3_repo)
    criterion = (BCEDiceBoundaryLoss(boundary_weight=args.boundary_weight)
                 if args.loss == "bce_dice_boundary" else BCEDiceLoss())
    optimizer = make_optimizer(args, model)
    total_steps = args.epochs * len(train_loader)
    warmup_steps = round(args.warmup_epochs * len(train_loader))
    lr_lambda = partial(
        cosine_warmup_lambda, warmup_steps=warmup_steps,
        total_steps=total_steps, min_ratio=args.min_lr_ratio)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    # BF16 has ample exponent range and does not require gradient scaling.
    scaler_enabled = use_amp and amp_dtype == torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=scaler_enabled)

    start_epoch, best_iou, stale_epochs, global_step = 1, -1.0, 0, 0
    if args.init:
        checkpoint = torch.load(args.init, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        print(f"[init] model weights from {args.init}; optimizer/schedule start fresh")
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        old_config = checkpoint.get("config", {})
        structural_keys = ("backbone", "decoder", "img_size", "unfreeze_blocks")
        mismatches = [
            f"{key}: checkpoint={old_config.get(key)!r}, command={getattr(args, key)!r}"
            for key in structural_keys
            if key in old_config and old_config[key] != getattr(args, key)
        ]
        if mismatches:
            raise ValueError("Resume configuration mismatch: " + "; ".join(mismatches))
        model.load_state_dict(checkpoint["model"])
        full_state = all(key in checkpoint for key in ("optimizer", "scaler", "global_step"))
        if full_state:
            optimizer.load_state_dict(checkpoint["optimizer"])
            scaler.load_state_dict(checkpoint["scaler"])
            start_epoch = checkpoint["epoch"] + 1
            best_iou = checkpoint.get("best_iou", checkpoint.get("val_iou", -1.0))
            stale_epochs = checkpoint.get("stale_epochs", 0)
            global_step = checkpoint["global_step"]
            # Rebuild for the requested final epoch, allowing an extended run.
            scheduler.last_epoch = global_step - 1
            ratio = lr_lambda(max(global_step - 1, 0))
            for group, base_lr in zip(optimizer.param_groups, scheduler.base_lrs):
                group["lr"] = base_lr * ratio
            scheduler._last_lr = [group["lr"] for group in optimizer.param_groups]
            restore_rng_state(checkpoint.get("rng_state"), loader_generator)
            print(f"[resume] full state from epoch={start_epoch - 1}, best={best_iou:.4f}")
        else:
            # A legacy file is a warm start, not an exact continuation. Start a
            # fresh model-selection run so the new output always gets best.pth.
            best_iou = -1.0
            print("[resume] legacy checkpoint: loaded model weights only; optimizer/schedule restart")

    (out_dir / "config.json").write_text(
        json.dumps(vars(args), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[config] backbone={args.backbone} decoder={args.decoder} image={args.img_size} "
          f"batch={args.batch_size} epochs={args.epochs}")
    print(f"[config] loss={args.loss} augment={args.augment} optimizer={args.optimizer} "
          f"decoder_lr={args.lr:g} backbone_lr={args.backbone_lr:g} warmup={args.warmup_epochs}")
    print(f"[config] device={device} amp={use_amp} dtype={amp_dtype} "
          f"unfreeze_blocks={args.unfreeze_blocks}")
    print(f"[data] train={len(train_set)} val={len(val_set)}")
    print(f"[model] trainable={model.trainable_params()/1e6:.2f}M "
          f"total={model.total_params()/1e6:.2f}M")

    history_path = out_dir / "history.csv"
    for epoch in range(start_epoch, args.epochs + 1):
        started = time.time()
        train_result = run_epoch(
            model, train_loader, criterion, device, args.threshold, use_amp, amp_dtype,
            optimizer, scaler, scheduler, args.clip_grad)
        global_step += len(train_loader)
        val_result = run_epoch(
            model, val_loader, criterion, device, args.threshold, use_amp, amp_dtype)
        elapsed = time.time() - started
        improved = val_result["iou"] > best_iou + args.min_delta
        if improved:
            best_iou, stale_epochs = val_result["iou"], 0
        else:
            stale_epochs += 1

        payload = {
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
            "epoch": epoch, "global_step": global_step, "best_iou": best_iou,
            "val_iou": val_result["iou"], "stale_epochs": stale_epochs,
            "config": vars(args), "rng_state": rng_state(loader_generator),
        }
        atomic_torch_save(payload, out_dir / "last.pth")
        if improved:
            atomic_torch_save(payload, out_dir / "best.pth")

        row = {
            "epoch": epoch, "decoder_lr": optimizer.param_groups[0]["lr"],
            "backbone_lr": (optimizer.param_groups[1]["lr"]
                              if len(optimizer.param_groups) > 1 else 0.0),
            **{f"train_{key}": value for key, value in train_result.items()},
            **{f"val_{key}": value for key, value in val_result.items()},
            "seconds": elapsed,
        }
        append_csv(history_path, row)
        tag = " <- best" if improved else ""
        print(
            f"epoch {epoch:03d}/{args.epochs} lr={row['decoder_lr']:.2e} "
            f"train loss={train_result['loss']:.4f} iou={train_result['iou']:.4f} | "
            f"val loss={val_result['loss']:.4f} iou={val_result['iou']:.4f} "
            f"f1={val_result['f1']:.4f} p={val_result['precision']:.4f} "
            f"r={val_result['recall']:.4f} ({elapsed:.0f}s){tag}")

        if args.early_stop > 0 and stale_epochs >= args.early_stop:
            print(f"[early-stop] no val-IoU improvement for {stale_epochs} epochs")
            break
    print(f"[done] best val IoU={best_iou:.4f}; outputs={out_dir.resolve()}")


if __name__ == "__main__":
    main()
