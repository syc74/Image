from __future__ import annotations

from dataclasses import dataclass, asdict
import os
from typing import Any

import torch
from tqdm import tqdm

from .losses import hinted_l1, total_variation


@dataclass(slots=True)
class TrainConfig:
    epochs: int = 20
    lr: float = 2e-4
    weight_decay: float = 0.0
    log_every: int = 50
    grad_clip: float = 0.0

    save_dir: str = "./checkpoints"
    best_name: str = "best.pt"
    last_name: str = "last.pt"

    # 继续训练相关：
    # - None / ""：从头训练
    # - 传入 checkpoint 路径：从该 checkpoint 恢复
    resume_from: str | None = None
    resume_strict: bool = True
    reset_optimizer: bool = False


@dataclass(slots=True)
class LossConfig:
    lambda_hint: float = 20.0
    tv_weight: float = 0.0


def _torch_load(path: str, map_location: str | torch.device = "cpu") -> Any:
    """兼容不同 PyTorch 版本，且显式声明当前使用完整 checkpoint。"""
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _optimizer_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        if isinstance(state, dict):
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to(device)
        elif torch.is_tensor(state):
            state.data = state.data.to(device)
            if state._grad is not None:
                state._grad.data = state._grad.data.to(device)


def _normalize_history(history: Any) -> dict[str, list[float]]:
    default = {"train_loss": [], "val_loss": []}
    if not isinstance(history, dict):
        return default

    out: dict[str, list[float]] = {}
    for key in ["train_loss", "val_loss"]:
        values = history.get(key, [])
        if isinstance(values, (list, tuple)):
            out[key] = [float(v) for v in values]
        else:
            out[key] = []
    return out


def _save_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    train_cfg: TrainConfig,
    loss_cfg: LossConfig,
    history: dict[str, list[float]],
    epoch: int,
    best_val: float,
) -> None:
    save_folder = os.path.dirname(path)
    if save_folder:
        os.makedirs(save_folder, exist_ok=True)

    payload = {
        "epoch": int(epoch),
        "best_val": float(best_val),
        "train_cfg": asdict(train_cfg),
        "loss_cfg": asdict(loss_cfg),
        "model_cfg": {
            "in_channels": getattr(model, "in_channels", 4),
            "out_channels": getattr(model, "out_channels", 2),
            "base_channels": getattr(model, "base_channels", 64),
        },
        "history": _normalize_history(history),
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(payload, path)


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    device: torch.device | str | None = None,
    *,
    strict: bool = True,
    load_optimizer: bool = True,
) -> dict[str, Any]:
    """加载 checkpoint，并返回恢复训练需要的元信息。"""
    if not path:
        raise ValueError("checkpoint 路径不能为空。")
    if not os.path.exists(path):
        raise FileNotFoundError(f"checkpoint 不存在: {path}")

    checkpoint = _torch_load(path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise ValueError("checkpoint 格式不正确：期望 dict。")

    state_dict = checkpoint.get("state_dict", checkpoint)
    incompatible = model.load_state_dict(state_dict, strict=strict)

    optimizer_loaded = False
    if optimizer is not None and load_optimizer:
        optimizer_state = checkpoint.get("optimizer")
        if optimizer_state is not None:
            optimizer.load_state_dict(optimizer_state)
            if device is not None:
                _optimizer_to_device(optimizer, torch.device(device))
            optimizer_loaded = True

    return {
        "epoch": int(checkpoint.get("epoch", 0)),
        "start_epoch": int(checkpoint.get("epoch", 0)) + 1,
        "best_val": float(checkpoint.get("best_val", float("inf"))),
        "history": _normalize_history(checkpoint.get("history")),
        "model_cfg": checkpoint.get("model_cfg", {}),
        "train_cfg": checkpoint.get("train_cfg", {}),
        "loss_cfg": checkpoint.get("loss_cfg", {}),
        "optimizer_loaded": optimizer_loaded,
        "missing_keys": list(getattr(incompatible, "missing_keys", [])),
        "unexpected_keys": list(getattr(incompatible, "unexpected_keys", [])),
        "path": path,
    }


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader, device: torch.device, loss_cfg: LossConfig) -> float:
    model.eval()
    total = 0.0
    n = 0
    for batch in loader:
        x = batch["x"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True)
        m = batch["mask"].to(device, non_blocking=True)

        pred = model(x)
        loss = hinted_l1(pred, y, m, lambda_hint=loss_cfg.lambda_hint)
        if loss_cfg.tv_weight > 0:
            loss = loss + float(loss_cfg.tv_weight) * total_variation(pred)

        total += float(loss.item()) * x.size(0)
        n += int(x.size(0))
    return total / max(1, n)


def fit(model: torch.nn.Module, train_loader, val_loader, device: torch.device, train_cfg: TrainConfig, loss_cfg: LossConfig) -> dict:
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)

    best_val = float("inf")
    history = {"train_loss": [], "val_loss": []}
    start_epoch = 1

    best_path = os.path.join(train_cfg.save_dir, train_cfg.best_name)
    last_path = os.path.join(train_cfg.save_dir, train_cfg.last_name)

    if train_cfg.resume_from:
        resume_info = load_checkpoint(
            train_cfg.resume_from,
            model,
            optimizer=optimizer,
            device=device,
            strict=train_cfg.resume_strict,
            load_optimizer=not train_cfg.reset_optimizer,
        )
        start_epoch = max(1, int(resume_info["start_epoch"]))
        best_val = float(resume_info["best_val"])
        history = _normalize_history(resume_info["history"])

        print(
            f"Resume from {resume_info['path']} | "
            f"last epoch {resume_info['epoch']} | "
            f"next epoch {start_epoch} | "
            f"best {best_val:.4f} | "
            f"optimizer {'loaded' if resume_info['optimizer_loaded'] else 'reset'}"
        )
        if not train_cfg.resume_strict:
            if resume_info["missing_keys"]:
                print("Missing keys:", resume_info["missing_keys"])
            if resume_info["unexpected_keys"]:
                print("Unexpected keys:", resume_info["unexpected_keys"])

    total_epochs = int(train_cfg.epochs)
    if start_epoch > total_epochs:
        print(
            f"checkpoint 已训练到 epoch {start_epoch - 1}，"
            f"而当前 train_cfg.epochs={total_epochs}，因此没有新的 epoch 需要继续训练。"
        )
        return history

    for epoch in range(start_epoch, total_epochs + 1):
        model.train()
        running = 0.0
        n = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{train_cfg.epochs}", leave=False)
        for it, batch in enumerate(pbar, start=1):
            x = batch["x"].to(device, non_blocking=True)
            y = batch["y"].to(device, non_blocking=True)
            m = batch["mask"].to(device, non_blocking=True)

            pred = model(x)
            loss = hinted_l1(pred, y, m, lambda_hint=loss_cfg.lambda_hint)
            if loss_cfg.tv_weight > 0:
                loss = loss + float(loss_cfg.tv_weight) * total_variation(pred)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if train_cfg.grad_clip and train_cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
            optimizer.step()

            running += float(loss.item()) * x.size(0)
            n += int(x.size(0))

            if it % max(1, train_cfg.log_every) == 0:
                pbar.set_postfix(loss=f"{running/max(1,n):.4f}")

        train_loss = running / max(1, n)
        val_loss = evaluate(model, val_loader, device, loss_cfg)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            _save_checkpoint(best_path, model, optimizer, train_cfg, loss_cfg, history, epoch, best_val)
        _save_checkpoint(last_path, model, optimizer, train_cfg, loss_cfg, history, epoch, best_val)

        print(f"Epoch {epoch:03d} | train {train_loss:.4f} | val {val_loss:.4f} | best {best_val:.4f}")

    return history
