# task.py

from __future__ import annotations
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Any, List
import sys, os, yaml, torch, numpy as np
import lightning.pytorch as pl
from lightning.pytorch.callbacks import TQDMProgressBar  
import os
os.environ.setdefault("RICH_FORCE_TERMINAL", "1")  # 强制按终端渲染
os.environ.setdefault("RICH_NO_COLOR", "1")       # 可选：去掉颜色，日志更干净
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.loss_factory import get_losses_by_targets
from utils.metric_factory import get_metrics_by_targets

# ==== 路径注入：确保能 import 到 D:\tft_module ====
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
print(f"[FL] task.py loaded from: {Path(__file__).resolve()}")
print("[BOOT] CVD=", os.environ.get("CUDA_VISIBLE_DEVICES"))
# ==== 导入你项目内模块 ====
from model.tft_module import MyTFTModule
from data.load_dataset import get_dataloaders

# 自检：确保 get_dataloaders 接受必填 data_path
if "data_path" not in get_dataloaders.__code__.co_varnames:
    raise RuntimeError("当前导入的 get_dataloaders 不含必填参数 data_path，请检查导入路径是否正确")

# ==== Flower ↔ Torch 权重互转 ====
def get_weights(module: torch.nn.Module) -> List["np.ndarray"]:
    return [v.detach().cpu().numpy() for _, v in module.state_dict().items()]

def set_weights(module: torch.nn.Module, ndarrays: List["np.ndarray"]) -> None:
    sd = module.state_dict()
    new_sd = OrderedDict((k, torch.tensor(v)) for k, v in zip(sd.keys(), ndarrays))
    module.load_state_dict(new_sd, strict=False)

# ==== 配置 ====
DEFAULT_CFG = PROJECT_ROOT / "configs" / "model_config.yaml"
WEIGHTS_CFG = PROJECT_ROOT / "configs" / "weights_config.yaml"
print("[CLIENT] CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"))
import torch
print("[CLIENT] cuda.is_available =", torch.cuda.is_available(),
      "| count =", torch.cuda.device_count())
def load_yaml(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    if p.exists():
        return yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    return {"batch_size": 256, "num_workers": 4}

def _resolve_data_path(cfg: Dict[str, Any]) -> str:
    # 优先环境变量；其次配置；最后给个默认
    p = os.environ.get("TFM_DATA_PATH") or cfg.get("data_path") or "data/pkl_merged/full_merged.pkl"
    P = Path(p)
    if not P.is_absolute():
        P = PROJECT_ROOT / P   # ★ 用项目根作为相对路径基准
    P = P.resolve()
    print(f"[FL] using data_path: {P}")  # 方便日志排查
    return str(P)

# ==== 分片定义 ====
# task.py 中
SYMBOLS = ["BTC/USDT", "ETH/USDT", "BNB/USDT", "ADA/USDT", "SOL/USDT"]
PERIODS = ["1h", "4h"]  # 需要 1d 的话就加 "1d"

PARTITIONS = [{"symbol": s, "period": p} for s in SYMBOLS for p in PERIODS]

def resolve_partition(partition_id: int) -> Dict[str, str]:
    if partition_id < 0 or partition_id >= len(PARTITIONS):
        raise IndexError(f"partition-id={partition_id} 超出范围(0..{len(PARTITIONS)-1})")
    return PARTITIONS[partition_id]

# ==== 仅取前两个 DataLoader ====
def _take_two_loaders(ret) -> tuple:
    if isinstance(ret, (list, tuple)) and len(ret) >= 2:
        return ret[0], ret[1]
    raise RuntimeError("get_dataloaders 返回值不含前两个 DataLoader")

# ==== 构建 模型 + 数据 ====
def build_model_and_data(
    partition_id: int,
    config_path: str | Path = DEFAULT_CFG,
    batch_size: int | None = None,
    num_workers: int | None = None,
):
    cfg = load_yaml(config_path)
    data_path = _resolve_data_path(cfg)
    if not data_path:
        raise RuntimeError("未找到数据目录：请在 configs/model_config.yaml 写 data_path，或设置环境变量 TFM_DATA_PATH")

    bs = int(batch_size or cfg.get("batch_size", 256))
    nw = int(num_workers or cfg.get("num_workers", 4))
    val_mode = cfg.get("val_mode", "days")
    val_days = cfg.get("val_days", 252)
    val_ratio = cfg.get("val_ratio", 0.2)

    part = resolve_partition(partition_id)
    symbol, period = part["symbol"], part["period"]

    # ★ 只用带 data_path 的真实签名（绝不兜底到 config=cfg）
    ret = get_dataloaders(
        data_path=data_path,
        batch_size=bs,
        num_workers=nw,
        val_mode=val_mode,
        val_days=val_days,
        val_ratio=val_ratio,
        focus_symbol=symbol,     # 若实现未使用，会被 **kwargs 忽略
        focus_period=period,
    )
    if not isinstance(ret, (list, tuple)) or len(ret) < 6:
        raise RuntimeError("get_dataloaders 需要返回 (train_loader, val_loader, targets, train_ds, periods, norm_pack)")
    train_loader, val_loader, target_names, train_ds, periods, norm_pack = ret[:6]

    weight_cfg = load_yaml(WEIGHTS_CFG)
    weights = weight_cfg.get("custom_weights", [1.0] * len(target_names))

    enc = train_ds.categorical_encoders.get("period", None)
    classes_ = getattr(enc, "classes_", None)
    period_map = {i: c for i, c in enumerate(classes_)} if classes_ is not None else {i: p for i, p in enumerate(periods)}

    steps_per_epoch = len(train_loader)
    accum = int(cfg.get("accumulate", 1)) or 1
    steps_per_epoch_eff = max(1, steps_per_epoch // accum)

    model = MyTFTModule(
        dataset=train_ds,
        loss_list=get_losses_by_targets(target_names),
        weights=weights,
        output_size=[1] * len(target_names),
        metrics_list=get_metrics_by_targets(target_names, periods),
        target_names=target_names,
        period_map=period_map,
        learning_rate=cfg.get("learning_rate", 1e-3),
        loss_schedule=cfg.get("loss_schedule", {}),
        norm_pack=norm_pack,
        steps_per_epoch=steps_per_epoch_eff,
        hidden_size=cfg.get("hidden_size"),
        lstm_layers=cfg.get("lstm_layers"),
        attention_head_size=cfg.get("attention_head_size"),
        dropout=cfg.get("dropout"),
    )

    print(f"[FL] partition={partition_id} -> ({symbol},{period}) | data_path={data_path} | bs={bs} nw={nw}")
    return model, train_loader, val_loader

# ==== 本地训练/验证 ====
def local_train_validate(model: pl.LightningModule, train_loader, val_loader, local_epochs: int = 1) -> Dict[str, float]:
    use_bar = os.environ.get("FLWR_PROGRESS", "tqdm").lower()  # 可选: tqdm / none
    callbacks = []
    enable_bar = True
    if use_bar == "tqdm":
        callbacks = [TQDMProgressBar(refresh_rate=200)]   # 覆盖刷新，不刷屏
    elif use_bar == "none":
        enable_bar = False
        callbacks = []

    trainer = pl.Trainer(
    accelerator="gpu" if torch.cuda.is_available() else "cpu",
    devices=1,
    max_epochs=local_epochs,
    logger=False,
    enable_checkpointing=False,
    enable_progress_bar=enable_bar,
    callbacks=callbacks,
    num_sanity_val_steps=0,   # 关键：跳过 sanity check，规避 rich 的钩子
    log_every_n_steps=200   
)
    trainer.fit(model, train_loader, val_loader)
    metrics = trainer.callback_metrics
    out: Dict[str, float] = {}
    for k in ["val_loss", "val_loss_weighted_epoch", "val_loss_for_ckpt", "val_composite_score"]:
        if k in metrics:
            v = metrics[k]
            try: out[k] = float(v.item())
            except Exception:
                try: out[k] = float(v)
                except Exception: pass
    return out

def local_validate(model: pl.LightningModule, val_loader) -> Dict[str, float]:
    use_bar = os.environ.get("FLWR_PROGRESS", "tqdm").lower()  # 可选: tqdm / none
    callbacks = []
    enable_bar = True
    if use_bar == "tqdm":
        callbacks = [TQDMProgressBar(refresh_rate=200)]   # 覆盖刷新，不刷屏
    elif use_bar == "none":
        enable_bar = False
        callbacks = []
    trainer =pl.Trainer(
    accelerator="gpu" if torch.cuda.is_available() else "cpu",
    devices=1,
    logger=False,
    enable_checkpointing=False,
    enable_progress_bar=enable_bar,
    callbacks=callbacks,
    num_sanity_val_steps=0,   # 关键：跳过 sanity check，规避 rich 的钩子
    log_every_n_steps=200
)
    results = trainer.validate(model, dataloaders=val_loader, verbose=False)
    return {k: float(v) for k, v in (results[0] if results else {}).items()}
#              flwr run . local-simulation --run-config "num-server-rounds=3 local-epochs=1" --stream