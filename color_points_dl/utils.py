from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional
import random

import numpy as np
import torch


def seed_everything(seed: int = 42) -> None:
    """尽量让训练可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(prefer: Optional[Literal["cuda", "mps", "cpu"]] = None) -> torch.device:
    """自动选择设备：cuda > mps > cpu。"""
    if prefer == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if prefer == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if prefer == "cpu":
        return torch.device("cpu")

    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
