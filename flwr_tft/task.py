# task.py

from __future__ import annotations
from collections import OrderedDict
from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from typing import Tuple, Dict, Any, List
import numpy as np
import os
import yaml
import torch
import pytorch_lightning as pl

# ====== 你的项目内模块：根据仓库结构导入 ======
# 若本地类/函数名不同，只需改这里两行 import 或下面构造/调用处的参数。
from model.tft_module import MyTFTModule          # 你的 LightningModule
from data.load_dataset import get_dataloaders     # 你的 DataLoader 工厂


# ------- Flower <-> Torch 权重互转 -------
def get_weights(module: torch.nn.Module) -> List["np.ndarray"]:
    import numpy as np
    return [v.detach().cpu().numpy() for _, v in module.state_dict().items()]

def set_weights(module: torch.nn.Module, ndarrays: List["np.ndarray"]) -> None:
    sd = module.state_dict()
    new_sd = OrderedDict((k, torch.tensor(v)) for k, v in zip(sd.keys(), ndarrays))
    module.load_state_dict(new_sd, strict=False)  # strict=False 以兼容缓冲区/少量不对齐


# ------- 简单 YAML 加载（找不到就给最小默认配置） -------
def load_yaml(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    # 默认兜底（尽量与项目常见键保持一致）
    return {
        "batch_size": 256,
        "num_workers": 4,
    }


# ------- 构建分区列表：partition-id → (symbol, period) -------
# 与 pyproject 中 num-supernodes=4 对齐；要更多客户端，直接新增条目即可。
PARTITIONS = [
    {"symbol": "BTC/USDT", "period": "1h"},
    {"symbol": "ETH/USDT", "period": "1h"},
    {"symbol": "BTC/USDT", "period": "4h"},
    {"symbol": "ETH/USDT", "period": "4h"},
    # 例如再加：
    # {"symbol": "BNB/USDT", "period": "1h"},
    # {"symbol": "BNB/USDT", "period": "4h"},
]


def resolve_partition(partition_id: int) -> Dict[str, str]:
    if partition_id < 0 or partition_id >= len(PARTITIONS):
        raise IndexError(f"partition-id={partition_id} 超出范围(0..{len(PARTITIONS)-1})")
    return PARTITIONS[partition_id]


# ------- 构建 模型 + 训练/验证 DataLoader -------
def build_model_and_data(
    partition_id: int,
    config_path: str | Path = "configs/model_config.yaml",
    batch_size: int | None = None,
    num_workers: int | None = None,
):
    cfg = load_yaml(config_path)

    part = resolve_partition(partition_id)
    symbol, period = part["symbol"], part["period"]

    bs = int(batch_size or cfg.get("batch_size", 256))
    nw = int(num_workers or cfg.get("num_workers", 4))

    # 优先按“聚焦某 symbol/period”的方式取数据；如果签名不匹配则回退
    try:
        train_loader, val_loader = get_dataloaders(
            config=cfg,
            focus_symbol=symbol,
            focus_period=period,
            batch_size=bs,
            num_workers=nw,
        )
    except TypeError:
        # 你本地若是其它签名/参数名，这里可按需改造
        train_loader, val_loader = get_dataloaders(config=cfg)

    # 构造 LightningModule（若需要更多初始化参数，改这里）
    try:
        model = MyTFTModule(cfg)
    except TypeError:
        model = MyTFTModule()

    return model, train_loader, val_loader


# ------- 本地训练与评估（Lightning 驱动）-------
def local_train_validate(
    model: pl.LightningModule,
    train_loader,
    val_loader,
    local_epochs: int = 1,
) -> Dict[str, float]:
    trainer = pl.Trainer(
        accelerator="auto",
        devices=1,
        max_epochs=local_epochs,
        enable_checkpointing=False,
        logger=False,
        log_every_n_steps=50,
    )
    trainer.fit(model, train_loader, val_loader)

    # 从 callback_metrics 抓一些常见键；没有也不影响 Flower 正常流程
    metrics = trainer.callback_metrics
    out: Dict[str, float] = {}
    for k in ["val_loss", "val_loss_weighted_epoch", "val_loss_for_ckpt", "val_composite_score"]:
        if k in metrics:
            v = metrics[k]
            try:
                out[k] = float(v.item())  # Tensor
            except Exception:
                try:
                    out[k] = float(v)      # 标量
                except Exception:
                    pass
    return out


def local_validate(
    model: pl.LightningModule,
    val_loader,
) -> Dict[str, float]:
    trainer = pl.Trainer(
        accelerator="auto",
        devices=1,
        enable_checkpointing=False,
        logger=False,
    )
    results = trainer.validate(model, dataloaders=val_loader, verbose=False)
    return {k: float(v) for k, v in (results[0] if results else {}).items()}
