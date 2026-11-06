# -*- coding: utf-8 -*-
"""Sliding-window inference with overlapping-tile probability averaging."""
import argparse
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from segmenter import BuildingSegmenter


@torch.no_grad()
def predict_full(model, img: np.ndarray, device, window=448, stride=384, threshold=0.5):
    """img: HxWx3 (0~255). 返回概率图与二值mask (H,W) float32"""
    H, W = img.shape[:2]
    prob = np.zeros((H, W), dtype=np.float32)
    count = np.zeros((H, W), dtype=np.float32)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    model.eval()
    ys = list(range(0, max(H - window, 0) + 1, stride)) or [0]
    xs = list(range(0, max(W - window, 0) + 1, stride)) or [0]
    if ys[-1] + window < H: ys.append(H - window)
    if xs[-1] + window < W: xs.append(W - window)
    for y in ys:
        for x in xs:
            tile = img[y:y + window, x:x + window]
            t = torch.from_numpy((tile.astype(np.float32) / 255.0 - mean) / std)
            t = t.permute(2, 0, 1).unsqueeze(0).to(device)
            logits = model(t)
            p = torch.sigmoid(logits)[0, 0].cpu().numpy()
            ph, pw = p.shape[:2]
            prob[y:y + ph, x:x + pw] += p
            count[y:y + ph, x:x + pw] += 1
    prob /= np.maximum(count, 1)
    return prob, (prob > threshold).astype(np.float32)


def load_model(ckpt_path, device, pretrained_dir=None, dinov3_repo=None):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    # A saved absolute path may belong to another machine. Let the local
    # repository default resolve weights unless an override was provided.
    local_pretrained = pretrained_dir
    if local_pretrained is None and cfg.get("pretrained_dir"):
        saved = Path(cfg["pretrained_dir"])
        local_pretrained = str(saved) if saved.is_dir() else None
    local_repo = dinov3_repo
    if local_repo is None and cfg.get("dinov3_repo"):
        saved_repo = Path(cfg["dinov3_repo"])
        local_repo = str(saved_repo) if saved_repo.is_dir() else None
    model = BuildingSegmenter(cfg["backbone"], cfg["decoder"], device=device,
                              pretrained_dir=local_pretrained,
                              dinov3_repo=local_repo)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default=None, help="Path to one input image")
    ap.add_argument("--image-dir", default=None, help="Directory of input images")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dinov3-repo", default=os.environ.get("DINO_DD_DINO_REPO"))
    ap.add_argument("--pretrained-dir", default=None)
    ap.add_argument("--out", default="pred.png")
    ap.add_argument("--out-dir", default="preds")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--window", type=int, default=448)
    ap.add_argument("--stride", type=int, default=384)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(args.ckpt, device, args.pretrained_dir,
                       args.dinov3_repo)

    def run(img_path, out_path):
        img = np.asarray(Image.open(img_path).convert("RGB"))
        prob, mask = predict_full(model, img, device, args.window,
                                  args.stride, args.threshold)
        Image.fromarray((mask * 255).astype(np.uint8)).save(out_path)
        Image.fromarray((prob * 255).astype(np.uint8)).save(
            os.path.splitext(out_path)[0] + "_prob.png")
        print(f"saved: {out_path} (building ratio={mask.mean():.3f})")

    if args.image:
        run(args.image, args.out)
    elif args.image_dir:
        os.makedirs(args.out_dir, exist_ok=True)
        for f in sorted(os.listdir(args.image_dir)):
            if f.lower().endswith((".png", ".jpg", ".tif", ".tiff")):
                run(os.path.join(args.image_dir, f),
                    os.path.join(args.out_dir, f))
    else:
        ap.error("Either --image or --image-dir is required")


if __name__ == "__main__":
    main()
