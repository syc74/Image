from __future__ import annotations

from typing import Tuple
import numpy as np
import cv2


def bgr_to_lab_norm(img_bgr_u8: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """BGR uint8 -> (L, ab) normalized.

    Returns:
      L:  (1,H,W) float32 in [0,1]
      ab: (2,H,W) float32 in [-1,1]
    """
    if img_bgr_u8.dtype != np.uint8:
        img_bgr_u8 = np.clip(img_bgr_u8, 0, 255).astype(np.uint8)

    lab = cv2.cvtColor(img_bgr_u8, cv2.COLOR_BGR2LAB)
    L = lab[:, :, 0].astype(np.float32) / 255.0
    a = (lab[:, :, 1].astype(np.float32) - 128.0) / 127.0
    b = (lab[:, :, 2].astype(np.float32) - 128.0) / 127.0
    return L[None, :, :], np.stack([a, b], axis=0).astype(np.float32)


def lab_norm_to_bgr(L: np.ndarray, ab: np.ndarray) -> np.ndarray:
    """(L,ab) normalized -> BGR uint8."""
    L = np.clip(L, 0.0, 1.0)
    ab = np.clip(ab, -1.0, 1.0)
    H, W = L.shape[1], L.shape[2]
    lab = np.zeros((H, W, 3), dtype=np.uint8)
    lab[:, :, 0] = np.clip(L[0] * 255.0, 0, 255).astype(np.uint8)
    lab[:, :, 1] = np.clip(ab[0] * 127.0 + 128.0, 0, 255).astype(np.uint8)
    lab[:, :, 2] = np.clip(ab[1] * 127.0 + 128.0, 0, 255).astype(np.uint8)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
