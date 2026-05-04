from __future__ import annotations

import torch
import torch.nn.functional as F


def hinted_l1(pred_ab: torch.Tensor, gt_ab: torch.Tensor, mask: torch.Tensor, *, lambda_hint: float = 20.0) -> torch.Tensor:
    """全图 L1 + hint 区域加权 L1。"""
    l_all = F.l1_loss(pred_ab, gt_ab)
    denom = mask.sum().clamp(min=1.0)
    l_hint = (torch.abs(pred_ab - gt_ab) * mask).sum() / denom
    return l_all + float(lambda_hint) * l_hint


def total_variation(x: torch.Tensor) -> torch.Tensor:
    dh = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]).mean()
    dw = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]).mean()
    return dh + dw
