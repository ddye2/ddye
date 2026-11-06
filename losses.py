# -*- coding: utf-8 -*-
"""损失: BCE + Dice 组合 (对抗建筑物像素占比小的类别不平衡)"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def _dice_loss(probs: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """共享 Dice 实现(单一来源, 修复P3-5重复); fp32防护(P1-1, 实测autocast下sigmoid
    已在fp32白名单故非必需, 保留为防御性)。
    probs/target: 展平后 (B, HW), 取值[0,1]"""
    probs = probs.float()
    target = target.float()
    inter = (probs * target).sum(1)
    union = probs.sum(1) + target.sum(1)
    return (1 - (2 * inter + 1e-6) / (union + 1e-6)).mean()


class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight: float = 1.0, dice_weight: float = 1.0):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """logits (B,1,H,W); target (B,1,H,W) 取值0/1"""
        bce = self.bce(logits, target)
        probs = torch.sigmoid(logits)
        dice = _dice_loss(probs.flatten(1), target.flatten(1))
        return self.bce_weight * bce + self.dice_weight * dice


class BoundaryF1Loss(nn.Module):
    """Boundary F1 Loss (Bokhovkin et al., arXiv:1905.07852)

    按 yiskw713/boundary_loss_for_remote_sensing 官方复现移植为二值版:
      1) 边界带: b(x) = x - erode(x)   (max_pool(1-x) - (1-x) 的内边界)
      2) 容忍带: ext(b) = max_pool(b, theta)  (允许预测边界偏移 ±(theta-1)/2 px)
      3) 软匹配 P/R:
           P = sum(pred_b * gt_b_ext) / sum(pred_b)   # 预测边界落在真值容忍带内
           R = sum(pred_b_ext * gt_b) / sum(gt_b)     # 真值边界被预测容忍带覆盖
         loss = 1 - 2PR/(P+R)

    与带BCE的本质区别: 对 ±2px 内的边界偏移不惩罚(与BF1评估指标对齐),
    不会扰乱概率校准。空mask样本(GT无边)不参与惩罚。
    """

    def __init__(self, theta0: int = 3, theta: int = 5):
        super().__init__()
        self.theta0 = theta0   # 边界带宽度
        self.theta = theta     # 容忍带宽度(论文默认5 -> ±2px)

    @staticmethod
    def _bnd(x: torch.Tensor, k: int) -> torch.Tensor:
        """边界带 b(x) = x - erode(x), 对0/1和连续概率均成立且可导"""
        return x - (-F.max_pool2d(-x, k, stride=1, padding=k // 2))

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        p = torch.sigmoid(logits.float())
        t = target.float()
        # 内边界带
        pred_b = self._bnd(p, self.theta0)
        gt_b = self._bnd(t, self.theta0)
        # 容忍带(膨胀)
        pred_b_ext = F.max_pool2d(pred_b, self.theta, stride=1, padding=self.theta // 2)
        gt_b_ext = F.max_pool2d(gt_b, self.theta, stride=1, padding=self.theta // 2)
        # 展平为 (B, HW)
        pred_b = pred_b.flatten(1); gt_b = gt_b.flatten(1)
        pred_b_ext = pred_b_ext.flatten(1); gt_b_ext = gt_b_ext.flatten(1)
        eps = 1e-7
        # Precision / Recall, 空边界样本不惩罚(P或R分母为0时置1)
        ps = pred_b.sum(1)
        gs = gt_b.sum(1)
        P = torch.where(ps > 0,
                        (pred_b * gt_b_ext).sum(1) / (ps + eps),
                        torch.ones_like(ps))
        R = torch.where(gs > 0,
                        (pred_b_ext * gt_b).sum(1) / (gs + eps),
                        torch.ones_like(gs))
        bf1 = 2 * P * R / (P + R + eps)
        return (1 - bf1).mean()


class BCEDiceBoundaryLoss(nn.Module):
    """BCE + Dice + Boundary-F1 软匹配边界损失 (Bokhovkin式)

    total = BCE + Dice + boundary_weight * (1 - BF1)
    论文推荐辅助权重 0.4; 不需要额外边界头, 直接作用于主分割logits。
    """

    def __init__(self, bce_weight: float = 1.0, dice_weight: float = 1.0,
                 boundary_weight: float = 0.4):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.boundary_weight = boundary_weight
        self.boundary_f1 = BoundaryF1Loss()

    def forward(self, logits: torch.Tensor, target: torch.Tensor):
        # 1) 全图BCE
        bce = self.bce(logits, target)
        # 2) Dice
        probs = torch.sigmoid(logits)
        dice = _dice_loss(probs.flatten(1), target.flatten(1))
        # 3) Boundary-F1 软匹配 (±2px 容忍)
        if self.boundary_weight > 0:
            bnd = self.boundary_f1(logits, target)
        else:
            bnd = torch.zeros((), device=logits.device)
        return (self.bce_weight * bce + self.dice_weight * dice
                + self.boundary_weight * bnd)


@torch.no_grad()
def confusion_counts(logits, target, threshold: float = 0.5):
    """返回 (TP, FP, FN) 像素计数, 用于累计计算 IoU/F1"""
    pred = (torch.sigmoid(logits) > threshold).float()
    tp = (pred * target).sum().item()
    fp = (pred * (1 - target)).sum().item()
    fn = ((1 - pred) * target).sum().item()
    return tp, fp, fn


def prf_from_counts(tp, fp, fn):
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    iou = tp / (tp + fp + fn + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    return precision, recall, f1, iou
