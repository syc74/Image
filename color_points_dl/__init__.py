"""color_points_dl (Python 3.12+)

公共入口：
- Dataset: ColorPointsDataset, DataConfig
- Model: UNetColorizer
- Train: fit, TrainConfig, LossConfig
- Inference (GUI): Colorizer
"""

from .dataset import ColorPointsDataset, DataConfig
from .points import PointSamplerConfig
from .unet import UNetColorizer
from .train_loop import TrainConfig, LossConfig, fit
from .infer import Colorizer
