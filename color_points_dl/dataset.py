from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .scan import read_manifest
from .color_utils import bgr_to_lab_norm
from .points import PointSamplerConfig, sample_points, make_hint_maps_from_ab


def _resize_square(img_bgr: np.ndarray, size: int) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    if h == size and w == size:
        return img_bgr
    inter = cv2.INTER_AREA if (h > size or w > size) else cv2.INTER_CUBIC
    return cv2.resize(img_bgr, (size, size), interpolation=inter)


def _square_crop_resize(
    img_bgr: np.ndarray,
    size: int,
    rng: np.random.Generator,
    *,
    random_crop: bool,
    scale_min: float = 0.65,
) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    side_max = min(h, w)

    scale_min = max(0.01, min(1.0, float(scale_min)))

    if random_crop and side_max > 1:
        side_min = max(16, int(side_max * scale_min))
        side_min = min(side_min, side_max)

        side = int(rng.integers(side_min, side_max + 1))
        x0 = int(rng.integers(0, w - side + 1))
        y0 = int(rng.integers(0, h - side + 1))
    else:
        side = side_max
        x0 = (w - side) // 2
        y0 = (h - side) // 2

    img_bgr = img_bgr[y0:y0 + side, x0:x0 + side]
    inter = cv2.INTER_AREA if side > size else cv2.INTER_CUBIC
    return cv2.resize(img_bgr, (size, size), interpolation=inter)


def _color_jitter_bgr(
    img_bgr: np.ndarray,
    rng: np.random.Generator,
    *,
    brightness: float,
    contrast: float,
    saturation: float,
) -> np.ndarray:
    img = img_bgr.astype(np.float32)

    if contrast > 0:
        alpha = 1.0 + float(rng.uniform(-contrast, contrast))
        img = (img - 127.5) * alpha + 127.5

    if brightness > 0:
        beta = float(rng.uniform(-brightness, brightness)) * 255.0
        img = img + beta

    img = np.clip(img, 0, 255).astype(np.uint8)

    if saturation > 0:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] *= 1.0 + float(rng.uniform(-saturation, saturation))
        hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
        img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    return img


@dataclass(slots=True)
class DataConfig:
    crop_size: int = 256

    # 原来的增强
    augment_hflip: bool = True

    # 新增：random crop，只在 train 用
    augment_random_crop: bool = True
    crop_scale_min: float = 0.65

    # 新增：可选几何增强，只在 train 用
    augment_vflip: bool = False
    augment_rot90: bool = False

    # 新增：可选颜色增强，只在 train 用
    augment_color_jitter: bool = False
    jitter_brightness: float = 0.05
    jitter_contrast: float = 0.05
    jitter_saturation: float = 0.10

    # 原来的 point sampler
    point_cfg: PointSamplerConfig = field(default_factory=PointSamplerConfig)

    # 原来的 max_images
    max_images: Optional[int] = None


class ColorPointsDataset(Dataset):
    """从 manifest 读取图片，在线生成训练样本。

    Return dict:
      x: (4,H,W) float32
      y: (2,H,W) float32
      mask: (1,H,W) float32
      L: (1,H,W) float32
      meta: 仅包含可 collate 的字段（避免 batch 拼接报错）
    """

    def __init__(
        self,
        manifest_path: str,
        data_cfg: DataConfig,
        seed: int = 0,
        is_train: bool = True,
    ):
        self.paths = read_manifest(manifest_path)

        if data_cfg.max_images is not None:
            self.paths = self.paths[: int(data_cfg.max_images)]

        self.cfg = data_cfg
        self.is_train = bool(is_train)

        # 按照原来的逻辑：train 和 val 都用这个 rng。
        # 所以 val 每次 __getitem__ 仍然会随机生成 hint points / radius。
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        path = self.paths[idx]

        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"Failed to read image: {path}")

        rng = self.rng
        size = int(self.cfg.crop_size)

        # train: 可选 random crop
        # val: 按照原来的方式 resize_square，不做固定 crop，不做增强
        if self.is_train and self.cfg.augment_random_crop:
            img = _square_crop_resize(
                img,
                size,
                rng,
                random_crop=True,
                scale_min=float(self.cfg.crop_scale_min),
            )
        else:
            img = _resize_square(img, size)

        # train only: horizontal flip
        if self.is_train and self.cfg.augment_hflip:
            if float(rng.random()) < 0.5:
                img = cv2.flip(img, 1)

        # train only: vertical flip
        if self.is_train and self.cfg.augment_vflip:
            if float(rng.random()) < 0.2:
                img = cv2.flip(img, 0)

        # train only: 90-degree rotation
        if self.is_train and self.cfg.augment_rot90:
            k = int(rng.integers(0, 4))
            if k:
                img = np.ascontiguousarray(np.rot90(img, k))

        # train only: colour jitter
        if self.is_train and self.cfg.augment_color_jitter:
            img = _color_jitter_bgr(
                img,
                rng,
                brightness=float(self.cfg.jitter_brightness),
                contrast=float(self.cfg.jitter_contrast),
                saturation=float(self.cfg.jitter_saturation),
            )

        img = np.ascontiguousarray(img)

        # BGR -> LAB normalized
        L, ab = bgr_to_lab_norm(img)  # L: (1,H,W), ab: (2,H,W)
        _, H, W = L.shape

        # 按照原来的逻辑：
        # train 和 val 都随机采样 points/radius。
        points, radius = sample_points(W, H, self.cfg.point_cfg, rng=rng)
        hint_ab, mask = make_hint_maps_from_ab(ab, points, radius)

        # x = L + sparse ab hints + mask
        x = np.concatenate([L, hint_ab, mask], axis=0).astype(np.float32)

        return {
            "x": torch.from_numpy(x),
            "y": torch.from_numpy(ab.astype(np.float32)),
            "mask": torch.from_numpy(mask.astype(np.float32)),
            "L": torch.from_numpy(L.astype(np.float32)),
            "meta": {
                "path": path,
                "radius": int(radius),
                "points_count": int(len(points)),
            },
        }
