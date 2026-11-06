# dino_dd

Clean DINOv3 building-extraction codebase containing only the training,
inference, data, loss, backbone, and decoder components. Experimental scripts,
temporary checks, report-generation utilities, logs, and caches were not copied.

## Layout

```text
train.py                 Training and validation loop
infer.py                 Single-image or directory sliding-window inference
dataset.py               WHU-style train/val image-mask dataset
losses.py                BCE+Dice and BCE+Dice+Boundary losses
segmenter.py             Backbone + decoder model wrapper
models/backbone.py       DINOv3 loading and intermediate feature extraction
models/decoders/         Linear, CNN, DPT, DPT-GLA, U-Net, and HR-DPT heads
```

## DINOv3 path

This project is standalone and does not assume that it lives inside the DINOv3
repository. Set the local source path once:

```bash
export DINO_DD_DINO_REPO=/path/to/dinov3-main
```

Or pass `--dinov3-repo /path/to/dinov3-main` explicitly. Pretrained weights are
searched in `DINOV3_REPO/dinov3/pretrained` unless `--pretrained-dir` is given.
On the current Windows workspace, the source path is
`C:\ddye-work\B-working\dinov3\dinov3-main`.

## Training

Frozen-backbone HR-DPT baseline:

```bash
python train.py \
  --dinov3-repo /path/to/dinov3-main \
  --pretrained-dir /path/to/pretrained \
  --data-root /path/to/whu \
  --backbone vitl16_sat \
  --decoder unibuild_hlrdpt \
  --img-size 512 --batch-size 6 --epochs 100 \
  --lr 3e-5 --backbone-lr 1e-6 --unfreeze-blocks 0 \
  --optimizer adamw --weight-decay 1e-4 \
  --warmup-epochs 5 --min-lr-ratio 0.01 \
  --loss bce_dice_boundary --boundary-weight 0.4 \
  --augment standard --amp --amp-dtype auto --clip-grad 1.0 \
  --early-stop 20 --seed 42 --device cuda:0 \
  --out-dir outputs/hlrdpt_frozen
```

Fine-tune all ViT-L transformer blocks, warm-started from HR-DPT:

```bash
python train.py \
  --dinov3-repo /path/to/dinov3-main \
  --pretrained-dir /path/to/pretrained \
  --data-root /path/to/whu \
  --backbone vitl16_sat --decoder unibuild_hlrdpt \
  --img-size 512 --batch-size 4 --epochs 50 \
  --lr 2e-5 --backbone-lr 2e-6 --unfreeze-blocks 24 \
  --optimizer adamw --weight-decay 0.01 \
  --warmup-epochs 5 --min-lr-ratio 0.01 \
  --loss bce_dice_boundary --boundary-weight 0.4 \
  --augment standard --amp --amp-dtype auto --clip-grad 1.0 \
  --early-stop 20 --seed 42 --device cuda:0 \
  --init outputs/hlrdpt_frozen/best.pth \
  --out-dir outputs/hlrdpt_fullft
```

`--unfreeze-blocks 24` unfreezes all ViT-L transformer blocks and the final
backbone norm. A few non-block parameters such as patch embedding remain
frozen, matching the previous experiment code.

## Inference

```bash
python infer.py \
  --dinov3-repo /path/to/dinov3-main \
  --ckpt outputs/hlrdpt_fullft/best.pth \
  --image-dir /path/to/images \
  --out-dir outputs/preds \
  --window 512 --stride 384 --threshold 0.5
```

The script writes both binary masks and probability images.
