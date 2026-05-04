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


@dataclass(slots=True)
class DataConfig:
    crop_size: int = 256
    augment_hflip: bool = True
    point_cfg: PointSamplerConfig = field(default_factory=PointSamplerConfig)
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

    def __init__(self, manifest_path: str, data_cfg: DataConfig, seed: int = 0, is_train: bool = True):
        self.paths = read_manifest(manifest_path)
        if data_cfg.max_images is not None:
            self.paths = self.paths[: int(data_cfg.max_images)]
        self.cfg = data_cfg
        self.is_train = bool(is_train)
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        path = self.paths[idx]
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"Failed to read image: {path}")

        size = int(self.cfg.crop_size)
        img = _resize_square(img, size)

        if self.is_train and self.cfg.augment_hflip:
            if float(self.rng.random()) < 0.5:
                img = cv2.flip(img, 1)

        L, ab = bgr_to_lab_norm(img)  # (1,H,W), (2,H,W)
        _, H, W = L.shape

        points, radius = sample_points(W, H, self.cfg.point_cfg, rng=self.rng)
        hint_ab, mask = make_hint_maps_from_ab(ab, points, radius)

        x = np.concatenate([L, hint_ab, mask], axis=0).astype(np.float32)

        return {
            "x": torch.from_numpy(x),
            "y": torch.from_numpy(ab.astype(np.float32)),
            "mask": torch.from_numpy(mask.astype(np.float32)),
            "L": torch.from_numpy(L.astype(np.float32)),
            "meta": {"path": path, "radius": int(radius), "points_count": int(len(points))},
        }
