# -*- coding: utf-8 -*-
"""Building segmentation wrapper for DINOv3 and selectable decoders."""
import torch
import torch.nn as nn

from models.backbone import load_backbone, extract_multiscale, BACKBONES
from models.decoders import DECODERS


class BuildingSegmenter(nn.Module):
    def __init__(self, backbone_name: str = "vits16", decoder_name: str = "dpt",
                 num_classes: int = 1, device: str = "cuda",
                 unfreeze_blocks: int = 0,
                 pretrained_dir: str | None = None,
                 dinov3_repo: str | None = None):
        super().__init__()
        entry, _, in_ch, layers = BACKBONES[backbone_name]
        self.backbone_name = backbone_name
        self.layers = layers
        self.backbone = load_backbone(
            backbone_name, device, pretrained_dir, dinov3_repo)
        self.decoder = DECODERS[decoder_name](in_ch=in_ch, num_classes=num_classes).to(device)
        self.unfreeze_blocks = 0
        self.set_unfreeze_blocks(unfreeze_blocks)

    def set_unfreeze_blocks(self, count: int = 0):
        """Freeze the backbone except for its final ``count`` transformer blocks."""
        blocks = getattr(self.backbone, "blocks", None)
        if count < 0 or (blocks is not None and count > len(blocks)):
            raise ValueError(f"Invalid unfreeze_blocks={count}")
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
        if count:
            if blocks is None:
                raise TypeError("Backbone has no .blocks attribute for partial fine-tuning")
            for block in blocks[-count:]:
                for parameter in block.parameters():
                    parameter.requires_grad = True
            # The final norm participates in all requested intermediate outputs.
            norm = getattr(self.backbone, "norm", None)
            if norm is not None:
                for parameter in norm.parameters():
                    parameter.requires_grad = True
        self.unfreeze_blocks = count

    def train(self, mode: bool = True):
        """Keep a frozen backbone in eval; enable train mode for fine-tuned blocks."""
        super().train(mode)
        # Frozen DINOv3 blocks must not enable stochastic depth merely because
        # a later block is fine-tuned. Only the explicitly unfrozen tail trains.
        self.backbone.eval()
        if mode and self.unfreeze_blocks:
            for block in self.backbone.blocks[-self.unfreeze_blocks:]:
                block.train(True)
            norm = getattr(self.backbone, "norm", None)
            if norm is not None:
                norm.train(True)
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out_size = x.shape[-2:]
        feats = extract_multiscale(
            self.backbone, x, self.layers, no_grad=self.unfreeze_blocks == 0)
        if getattr(self.decoder, "requires_image", False):
            return self.decoder(feats, out_size, x)
        return self.decoder(feats, out_size)

    def trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def decoder_params(self):
        return [p for p in self.decoder.parameters() if p.requires_grad]

    def backbone_params(self):
        return [p for p in self.backbone.parameters() if p.requires_grad]

    def total_params(self):
        return sum(p.numel() for p in self.parameters())
