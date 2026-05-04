from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable, Sequence

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def scan_images(root: str | os.PathLike) -> list[str]:
    """递归扫描 root 目录下所有图片文件，返回排序后的路径列表。"""
    root = os.fspath(root)
    paths: list[str] = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            ext = os.path.splitext(fn)[1].lower()
            if ext in IMAGE_EXTS:
                paths.append(os.path.join(dirpath, fn))
    paths.sort()
    return paths


def split_train_val(paths: Sequence[str], val_ratio: float = 0.1, seed: int = 42) -> tuple[list[str], list[str]]:
    import random
    rng = random.Random(seed)
    idx = list(range(len(paths)))
    rng.shuffle(idx)
    n_val = int(round(len(paths) * val_ratio))
    val = [paths[i] for i in idx[:n_val]]
    train = [paths[i] for i in idx[n_val:]]
    return train, val


def write_manifest(paths: Sequence[str], out_path: str | os.PathLike) -> None:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(list(paths), ensure_ascii=False, indent=2), encoding="utf-8")


def read_manifest(manifest_path: str | os.PathLike) -> list[str]:
    data = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Manifest must be a list[str], got: {type(data)}")
    return [str(x) for x in data]
