from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Union

import numpy as np
import cv2
import torch

from .unet import UNetColorizer
from .color_utils import bgr_to_lab_norm, lab_norm_to_bgr

Point = tuple[int, int]
PointWithColor = Union[tuple[int,int], tuple[int,int,tuple[int,int,int]], tuple[int,int,int,int,int]]


def _pad_to_multiple(x: torch.Tensor, multiple: int = 16) -> tuple[torch.Tensor, tuple[int,int,int,int]]:
    _, _, h, w = x.shape
    pad_h = (multiple - (h % multiple)) % multiple
    pad_w = (multiple - (w % multiple)) % multiple
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left
    x_pad = torch.nn.functional.pad(x, (left, right, top, bottom), mode="reflect")
    return x_pad, (left, right, top, bottom)


def _unpad(x: torch.Tensor, pad: tuple[int,int,int,int]) -> torch.Tensor:
    left, right, top, bottom = pad
    if top or bottom:
        x = x[:, :, top:x.size(2)-bottom, :]
    if left or right:
        x = x[:, :, :, left:x.size(3)-right]
    return x


def _parse_points(points: list[PointWithColor], img_bgr: np.ndarray) -> tuple[list[Point], list[tuple[int,int,int]]]:
    h, w = img_bgr.shape[:2]
    pts_xy: list[Point] = []
    pts_bgr: list[tuple[int,int,int]] = []
    for p in points:
        if len(p) == 2:
            x, y = int(p[0]), int(p[1])
            if 0 <= x < w and 0 <= y < h:
                pts_xy.append((x, y))
                pts_bgr.append(tuple(int(v) for v in img_bgr[y, x]))
        elif len(p) == 3 and isinstance(p[2], (tuple, list)) and len(p[2]) == 3:
            x, y = int(p[0]), int(p[1])
            if 0 <= x < w and 0 <= y < h:
                pts_xy.append((x, y))
                pts_bgr.append(tuple(int(v) for v in p[2]))
        elif len(p) == 5:
            x, y = int(p[0]), int(p[1])
            if 0 <= x < w and 0 <= y < h:
                pts_xy.append((x, y))
                pts_bgr.append((int(p[2]), int(p[3]), int(p[4])))
    return pts_xy, pts_bgr


class Colorizer:
    """用于 GUI 的推理封装。"""

    def __init__(self, model: UNetColorizer, device: torch.device):
        self.model = model.to(device).eval()
        self.device = device

    @staticmethod
    def choose_device() -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    @classmethod
    def from_checkpoint(cls, ckpt_path: str, device: Optional[torch.device] = None) -> "Colorizer":
        device = device or cls.choose_device()
        payload = torch.load(ckpt_path, map_location="cpu")
        mcfg = payload.get("model_cfg", {"in_channels": 4, "out_channels": 2, "base_channels": 64})
        model = UNetColorizer(
            in_channels=int(mcfg.get("in_channels", 4)),
            out_channels=int(mcfg.get("out_channels", 2)),
            base_channels=int(mcfg.get("base_channels", 64)),
        )
        model.load_state_dict(payload["state_dict"], strict=True)
        return cls(model=model, device=device)

    @torch.no_grad()
    def colorize_bgr(
        self,
        img_bgr: np.ndarray,
        points: list[PointWithColor],
        *,
        radius: int = 4,
        hard_overwrite: bool = True,
        max_side: Optional[int] = None,
    ) -> np.ndarray:
        if img_bgr.dtype != np.uint8:
            img_bgr = np.clip(img_bgr, 0, 255).astype(np.uint8)

        orig_h, orig_w = img_bgr.shape[:2]
        scale = 1.0
        if max_side is not None:
            m = max(orig_h, orig_w)
            if m > max_side:
                scale = max_side / float(m)
                new_w = int(round(orig_w * scale))
                new_h = int(round(orig_h * scale))
                img_bgr = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

                # scale points
                pts2: list[PointWithColor] = []
                for p in points:
                    x = int(round(p[0] * scale))
                    y = int(round(p[1] * scale))
                    if len(p) == 2:
                        pts2.append((x, y))
                    elif len(p) == 3:
                        pts2.append((x, y, p[2]))
                    elif len(p) == 5:
                        pts2.append((x, y, p[2], p[3], p[4]))
                points = pts2

        h, w = img_bgr.shape[:2]
        L, _ab = bgr_to_lab_norm(img_bgr)

        pts_xy, pts_bgr = _parse_points(points, img_bgr)

        hint_ab = np.zeros((2, h, w), dtype=np.float32)
        mask = np.zeros((1, h, w), dtype=np.float32)
        mask_u8 = np.ascontiguousarray(np.zeros((h, w), dtype=np.uint8))

        r = max(0, int(radius))
        for (x, y), bgr in zip(pts_xy, pts_bgr):
            col = np.uint8([[[bgr[0], bgr[1], bgr[2]]]])
            lab = cv2.cvtColor(col, cv2.COLOR_BGR2LAB)[0, 0]
            a = (float(lab[1]) - 128.0) / 127.0
            b = (float(lab[2]) - 128.0) / 127.0

            if r <= 0:
                mask_u8[y, x] = 255
                hint_ab[0, y, x] = a
                hint_ab[1, y, x] = b
            else:
                tmp = np.ascontiguousarray(np.zeros((h, w), dtype=np.uint8))
                cv2.circle(tmp, (int(x), int(y)), int(r), 255, thickness=-1)
                m = tmp > 0
                mask_u8[m] = 255
                hint_ab[0, m] = a
                hint_ab[1, m] = b

        m = mask_u8 > 0
        mask[0, m] = 1.0

        x_in = np.concatenate([L, hint_ab, mask], axis=0).astype(np.float32)
        x = torch.from_numpy(x_in)[None, ...].to(self.device)
        x_pad, pad = _pad_to_multiple(x, 16)
        pred = self.model(x_pad)
        pred = _unpad(pred, pad)[0].detach().cpu().numpy()

        if hard_overwrite:
            pred[:, m] = hint_ab[:, m]

        out = lab_norm_to_bgr(L, pred)

        if max_side is not None and scale != 1.0:
            out = cv2.resize(out, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)

        return out
