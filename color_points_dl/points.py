from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import cv2

Point = tuple[int, int]


def _cv_safe(arr: Any, dtype=np.uint8) -> np.ndarray:
    """确保 OpenCV 写入数组是 uint8 且内存连续。"""
    return np.ascontiguousarray(np.asarray(arr, dtype=dtype))


def _randint_inclusive(rng: Any, low: int, high: int) -> int:
    """兼容 numpy.random.Generator 与 random.Random，返回 [low, high]。"""
    if low > high:
        low, high = high, low
    if hasattr(rng, "integers"):
        return int(rng.integers(low, high + 1))
    if hasattr(rng, "randint"):
        return int(rng.randint(low, high))
    raise TypeError(f"Unsupported RNG type: {type(rng)}")


def _choice_no_replace(rng: Any, n: int, k: int) -> list[int]:
    k = max(0, min(int(k), int(n)))
    if k == 0:
        return []
    if hasattr(rng, "choice"):
        return [int(x) for x in rng.choice(n, size=k, replace=False).tolist()]
    if hasattr(rng, "sample"):
        return [int(x) for x in rng.sample(range(n), k)]
    # fallback
    import random
    return random.sample(range(n), k)


@dataclass(slots=True)
class PointSamplerConfig:
    """点采样配置（支持旧字段名）。"""

    mode: str = "mixed"  # random | uniform | mixed

    # new names
    k_min: int = 20
    k_max: int = 200
    step_min: int = 16
    step_max: int = 64
    radius_min: int = 1
    radius_max: int = 5

    # legacy names (optional)
    min_points: Optional[int] = None
    max_points: Optional[int] = None
    uniform_step: Optional[int] = None
    radius: Optional[int] = None

    def normalize(self) -> "PointSamplerConfig":
        """把旧字段映射到新字段，并做基本约束。"""
        if self.min_points is not None:
            self.k_min = int(self.min_points)
        if self.max_points is not None:
            self.k_max = int(self.max_points)
        if self.uniform_step is not None:
            s = int(self.uniform_step)
            self.step_min = s
            self.step_max = s
        if self.radius is not None:
            r = int(self.radius)
            self.radius_min = r
            self.radius_max = r

        self.k_min = max(0, int(self.k_min))
        self.k_max = max(0, int(self.k_max))
        self.step_min = max(1, int(self.step_min))
        self.step_max = max(1, int(self.step_max))
        self.radius_min = max(0, int(self.radius_min))
        self.radius_max = max(0, int(self.radius_max))
        return self


def sample_points(w: int, h: int, cfg: PointSamplerConfig, rng: Any) -> tuple[list[Point], int]:
    cfg = cfg.normalize()
    mode = (cfg.mode or "mixed").lower()
    if mode not in {"random", "uniform", "mixed"}:
        mode = "mixed"

    if mode == "random":
        k = _randint_inclusive(rng, cfg.k_min, cfg.k_max)
        idxs = _choice_no_replace(rng, w * h, k)
        pts = [(i % w, i // w) for i in idxs]
    elif mode == "uniform":
        sx = _randint_inclusive(rng, cfg.step_min, cfg.step_max)
        sy = _randint_inclusive(rng, cfg.step_min, cfg.step_max)
        pts = [(x, y) for y in range(0, h, sy) for x in range(0, w, sx)]
    else:
        if _randint_inclusive(rng, 0, 1) == 0:
            k = _randint_inclusive(rng, cfg.k_min, cfg.k_max)
            idxs = _choice_no_replace(rng, w * h, k)
            pts = [(i % w, i // w) for i in idxs]
        else:
            sx = _randint_inclusive(rng, cfg.step_min, cfg.step_max)
            sy = _randint_inclusive(rng, cfg.step_min, cfg.step_max)
            pts = [(x, y) for y in range(0, h, sy) for x in range(0, w, sx)]

    radius = _randint_inclusive(rng, cfg.radius_min, cfg.radius_max)
    return pts, int(radius)


def make_hint_maps_from_ab(ab: np.ndarray, points: list[Point], radius: int) -> tuple[np.ndarray, np.ndarray]:
    """从 GT ab 与 points 构建 hint_ab 与 mask。"""
    assert ab.ndim == 3 and ab.shape[0] == 2, "ab must be (2,H,W)"
    _, H, W = ab.shape

    hint_ab = np.zeros((2, H, W), dtype=np.float32)
    mask = np.zeros((1, H, W), dtype=np.float32)

    if not points:
        return hint_ab, mask

    r = max(0, int(radius))
    mask_u8 = _cv_safe(np.zeros((H, W), dtype=np.uint8), dtype=np.uint8)

    for x, y in points:
        if 0 <= x < W and 0 <= y < H:
            if r <= 0:
                mask_u8[y, x] = 255
            else:
                cv2.circle(mask_u8, (int(x), int(y)), int(r), 255, thickness=-1)

    m = mask_u8 > 0
    mask[0, m] = 1.0
    hint_ab[0, m] = ab[0, m]
    hint_ab[1, m] = ab[1, m]
    return hint_ab, mask
