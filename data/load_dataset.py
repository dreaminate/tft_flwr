# data/load_dataset.py

import pandas as pd
import pickle
from pathlib import Path
from typing import List, Tuple, Dict, Any
import numpy as np
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data import NaNLabelEncoder, MultiNormalizer, TorchNormalizer
from torch.utils.data import DataLoader

# 项目根：.../tft_flwr
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# ---- 兼容老版本 pickle 中的 numpy._core 路径 ----
class _NumpyCoreAliasUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith("numpy._core"):
            module = module.replace("numpy._core", "numpy.core")
        return super().find_class(module, name)

def _resolve_path(p) -> Path:
    P = Path(p)
    if not P.is_absolute():
        P = PROJECT_ROOT / P            # ★ 相对路径基于项目根
    return P.resolve()

def read_pickle_compat(path: str):
    P = _resolve_path(path)             # ★ 统一解析
    print(f"[FL] read_pickle_compat -> {P}")
    try:
        return pd.read_pickle(P)
    except ModuleNotFoundError as e:
        # 兼容旧对象里引用 numpy._core 的情况
        if "numpy._core" not in str(e):
            raise
        with open(P, "rb") as fh:
            return _NumpyCoreAliasUnpickler(fh).load()


def get_dataloaders(
    data_path: str,
    batch_size: int = 64,
    num_workers: int = 4,
    val_mode: str = "days",
    val_days: int = 30,
    val_ratio: float = 0.2,
    **kwargs,
) -> Tuple[DataLoader, DataLoader, List[str], TimeSeriesDataSet, List[str], Dict[str, Any]]:
    """
    返回：
      train_loader, val_loader, targets, train_ds, period_classes, norm_pack
    """

    # === 加载 pkl 数据（保持训练逻辑不变） ===
    df = read_pickle_compat(data_path)

    # === 类型转换（可选）===
    df = df.astype({
        **{col: "float32" for col in df.select_dtypes(include="float64").columns},
        **{col: "int32" for col in df.select_dtypes(include="int64").columns},
    })

    # === 时间戳处理、排序、构建时间索引 ===
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values(["symbol", "period", "datetime"]).copy()
    df["time_idx"] = df.groupby(["symbol", "period"]).cumcount()

    df["symbol"] = df["symbol"].astype(str)
    df["period"] = df["period"].astype(str)

    
    targets = [
        "target_binarytrend",
        "target_logreturn",
        "target_logsharpe_ratio",
        "target_breakout_count",
        "target_drawdown_prob",
        "target_max_drawdown",
        "target_pullback_prob",
        "target_return_outlier_lower_10",
        "target_return_outlier_upper_90",
        "target_sideway_detect",
        "target_trend_persistence",
    ]
    regression_targets = [
        "target_logreturn",
        "target_logsharpe_ratio",
        "target_breakout_count",
        "target_max_drawdown",
        "target_trend_persistence",
    ]

    known_reals = ["time_idx"]
    df.replace({None: np.nan}, inplace=True)

    # === 切分训练/验证 ===
    train_parts = []
    val_parts = []
    for (sym, per), group_df in df.groupby(["symbol", "period"]):
        group_df = group_df.sort_values("datetime")
        if val_mode == "days":
            val_cutoff = group_df["datetime"].max() - pd.Timedelta(days=val_days)
            train_parts.append(group_df[group_df["datetime"] <= val_cutoff])
            val_parts.append(group_df[group_df["datetime"] > val_cutoff])
        elif val_mode == "ratio":
            val_idx = int(len(group_df) * (1 - val_ratio))
            train_parts.append(group_df.iloc[:val_idx])
            val_parts.append(group_df.iloc[val_idx:])

    df_train = pd.concat(train_parts, ignore_index=True)
    df_val = pd.concat(val_parts, ignore_index=True)

    print(f"\n总样本数: {len(df)}")
    print(f"训练集样本数: {len(df_train)}, 验证集样本数: {len(df_val)}")

    # === 动态未知实值特征 ===
    unknown_reals = [
        c for c in df_train.columns
        if c not in known_reals
        and c not in ["timestamp", "symbol", "time_idx", "datetime", "period"]
        and not c.startswith("future")
        and c not in targets
        and not df_train[c].isna().all()
    ]

    # === 类别空间与标准化包 ===
    symbol_classes = sorted(df_train["symbol"].dropna().unique().tolist())
    period_classes = sorted(df_train["period"].dropna().unique().tolist())
    sym2idx = {s: i for i, s in enumerate(symbol_classes)}
    per2idx = {p: i for i, p in enumerate(period_classes)}

    S, P, T = len(symbol_classes), len(period_classes), len(regression_targets)
    means = np.zeros((S, P, T), dtype="float32")
    stds  = np.ones((S, P, T), dtype="float32")  # 避免除零，缺省为 1
    for (sym, per), g in df_train.groupby(["symbol", "period"]):
        si, pi = sym2idx[sym], per2idx[per]
        for ti, col in enumerate(regression_targets):
            if col in g.columns and len(g[col].dropna()):
                m = float(g[col].mean())
                s = float(g[col].std() or 0.0)
                std_safe = max(s, 1e-8)
                means[si, pi, ti] = m
                stds[si, pi, ti]  = std_safe

    norm_pack: Dict[str, Any] = {
        "regression_targets": regression_targets,
        "symbol_classes": symbol_classes,
        "period_classes": period_classes,
        "sym2idx": sym2idx,
        "per2idx": per2idx,
        "means": means,   # np.float32 [S,P,T]
        "stds": stds,     # np.float32 [S,P,T]
    }

    symbol_encoder = NaNLabelEncoder(add_nan=True)
    symbol_encoder.fit(pd.Series(symbol_classes))
    period_encoder = NaNLabelEncoder(add_nan=True)
    period_encoder.fit(pd.Series(period_classes))

    target_normalizer = MultiNormalizer([TorchNormalizer(method="identity", center=False)] * len(targets))
    ts_cfg = dict(
        time_idx="time_idx",
        target=targets,
        group_ids=["symbol", "period"],
        max_encoder_length=36,
        max_prediction_length=1,
        static_categoricals=["symbol", "period"],
        static_reals=[],
        categorical_encoders={
            "symbol": symbol_encoder,
            "period": period_encoder,
        },
        time_varying_known_reals=known_reals,
        time_varying_unknown_reals=unknown_reals,
        add_relative_time_idx=True,
        add_target_scales=False,
        allow_missing_timesteps=True,
        target_normalizer=target_normalizer,
    )

    try:
        train_ds = TimeSeriesDataSet(df_train, **ts_cfg)
        val_ds = TimeSeriesDataSet.from_dataset(train_ds, df_val, stop_randomization=True)
        print("✅ train_ds 构建成功")
        print("✅ val_ds 构建成功")
    except Exception as e:
        print("❌ TimeSeriesDataSet 构建失败！")
        print("原因：", str(e))
        import traceback
        traceback.print_exc()
        raise ValueError("构建 TimeSeriesDataSet 失败")


    train_loader = train_ds.to_dataloader(
        train=True,
        batch_size=batch_size,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        shuffle=True,
        pin_memory=True,
        prefetch_factor=4,
        drop_last=False,
    )

    val_loader = val_ds.to_dataloader(
        train=False,
        batch_size=batch_size,
        num_workers=num_workers,
        persistent_workers=False,
        pin_memory=True,
        prefetch_factor=2,
        drop_last=True,
    )

    return train_loader, val_loader, targets, train_ds, period_classes, norm_pack
