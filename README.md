# color_points_dl (Python 3.12+)

这是一套 **可训练 + 可推理 + 可接入你前面 GUI** 的深度学习交互式上色代码，目标是：

输入（4 通道）：
- L：灰度/亮度（CIELAB 的 L）
- hint_a / hint_b：稀疏提示点的色度
- mask：提示点位置

输出（2 通道）：
- 预测的 ab（CIELAB 色度）

模型：
- Conditional U-Net（条件 U-Net）

## 目录结构
- `color_points_dl/`：核心 Python 模块
- `01_preprocess_existing_data.ipynb`：扫描你现有数据目录生成 manifest
- `02_train_and_export.ipynb`：训练并导出 checkpoint（供 GUI 使用）

## 安装依赖（示例）
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> 注意：PyTorch 需要安装支持 Python 3.12 的版本（不同平台可能安装命令略有差异）。

## 训练流程
1. 运行 Notebook 01 生成：
   - `./manifests/train.json`
   - `./manifests/val.json`

2. 运行 Notebook 02 训练并得到：
   - `./checkpoints/best.pt`
   - `./checkpoints/last.pt`

### 从 checkpoint 继续训练
在 Notebook 02 里把：

```python
RESUME_FROM = None  # 例如 "./checkpoints/last.pt"
RESET_OPTIMIZER = False
```

改成：

```python
RESUME_FROM = "./checkpoints/last.pt"
RESET_OPTIMIZER = False  # True 表示只加载模型参数，不恢复优化器状态
```

然后在 `TrainConfig(...)` 中传入：

```python
resume_from=RESUME_FROM,
reset_optimizer=RESET_OPTIMIZER,
```

说明：
- `epochs` 表示**总训练轮数**。例如 checkpoint 已训练到第 5 轮，而你把 `epochs=20`，那就会从第 6 轮继续训练到第 20 轮。
- 老的 checkpoint（只有 `state_dict`、没有 `optimizer`）也能加载，只是优化器会重新开始。
- 当前这个无模型版本默认不附带 `best.pt / last.pt`，第一次训练会从头开始；训练完成后会自动生成新的 checkpoint，之后就可以继续训练。

## GUI 调用（最关键）
在你的 GUI 代码里：

```python
from color_points_dl.infer import Colorizer

colorizer = Colorizer.from_checkpoint("./checkpoints/best.pt")
out_bgr = colorizer.colorize_bgr(img_bgr, points=[(x,y), ...], radius=4, hard_overwrite=True)
```

`out_bgr` 就是重建后的彩色 BGR 图（uint8），直接显示/保存即可。
