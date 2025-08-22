
# tft\_flwr

> **多周期、多目标金融时间序列的 Temporal Fusion Transformer (TFT)** 训练与评估，并可一键切换到 **Flower 联邦学习**（仿真/多进程/分布式）。
> 特点：**可配置特征工程与数据合并**、**从零训练/断点恢复/热启动微调**、**复合指标选模**、**运行目录自动管理**、**安全聚合（成对掩码）与详细日志**。

---

## 目录（Table of Contents）

* [特性概览](#特性概览)
* [目录结构](#目录结构)
* [环境与依赖](#环境与依赖)
* [数据准备](#数据准备)

  * [特征与目标构建](#特征与目标构建)
  * [数据加载](#数据加载)
* [快速上手（TL;DR）](#快速上手tldr)
* [训练流程](#训练流程)

  * [1. 从零开始](#1-从零开始)
  * [2. 从断点恢复](#2-从断点恢复)
  * [3. 热启动微调（仅加载权重）](#3-热启动微调仅加载权重)
* [配置说明](#配置说明)

  * [model\_config.yaml](#model_configyaml)
  * [weights\_config.yaml](#weights_configyaml)
  * [composite\_score.yaml](#composite_scoreyaml)
* [联邦学习（Flower）](#联邦学习flower)

  * [安装与本地模拟](#安装与本地模拟)
  * [核心文件与职责](#核心文件与职责)
  * [常见运行方式](#常见运行方式)
  * [可调参数与示例](#可调参数与示例)
  * [安全聚合与隐私](#安全聚合与隐私)
* [日志与运行目录](#日志与运行目录)
* [最佳实践与建议](#最佳实践与建议)
* [常见问题（FAQ）](#常见问题faq)
* [其他实用脚本](#其他实用脚本)
* [贡献与许可证](#贡献与许可证)
* [联系方式](#联系方式)

---

## 特性概览

* ✅ **多任务预测**：在 PyTorch Lightning + pytorch-forecasting 上封装多目标（分类/回归）TFT。
* ✅ **数据流水线**：可配置的**特征工程**（指标、异常提示、局部归一化）与**多周期合并**（1h/4h/1d 下放）。
* ✅ **训练全流程**：新训练 / 断点恢复 / 热启动微调（只载权重）。
* ✅ **实验可复现**：运行目录自动化、TensorBoard 日志、配置快照、复合指标选模（Top-k）。
* ✅ **联邦学习接入**：提供 Flower **客户端/服务器**脚本，支持**安全聚合（成对掩码）**与**详细日志**。
* ✅ **可扩展**：支持策略拓展、数据分片、（可选）差分隐私等增强模块。

---

## 目录结构

```
tft_flwr/
├─ train_multi_tft.py          # 从零开始训练
├─ train_resume.py             # 从断点恢复训练
├─ warm_start_train.py         # 热启动微调（仅加载权重）
├─ configs/
│  ├─ model_config.yaml        # 模型与训练超参
│  ├─ weights_config.yaml      # 各目标损失权重
│  └─ composite_score.yaml     # 复合指标权重矩阵
├─ data/
│  └─ load_dataset.py          # 构造 TimeSeriesDataSet 与 DataLoader
├─ model/
│  └─ tft_module.py            # LightningModule 封装的 TFT
├─ utils/
│  ├─ run_helper.py            # 自动生成运行目录
│  ├─ loss_factory.py          # 损失工厂
│  ├─ metric_factory.py        # 指标工厂
│  ├─ checkpoint_utils.py      # 选择性加载权重（热启动）
│  ├─ composite.py             # 复合指标工具
│  └─ weighted_bce.py          # 带正样本权重的 BCE
├─ callbacks/
│  └─ custom_checkpoint.py     # 以复合指标保存 top-k
├─ src/                        # 数据抓取/特征工程脚本
│  ├─ indicating.py / indicators.py
│  ├─ data_fusion.py
│  ├─ target_config.py
│  └─ ...
└─ flwr_tft/                   # Flower 联邦学习组件
   ├─ client_app.py            # 客户端逻辑（带安全聚合）
   ├─ server_app.py            # 服务器策略与日志
   └─ task.py                  # 数据分片与模型构建
```

---

## 环境与依赖

* Python **≥ 3.9**
* `torch`, `pytorch-lightning`
* `pytorch-forecasting`
* `pyyaml`, `omegaconf`
* **联邦学习**：`flwr >= 1.19.0`

> 推荐创建独立虚拟环境（Conda/Mamba）。如需针对 GPU，请安装匹配 CUDA 的 PyTorch 版本。

---

## 数据准备

### 特征与目标构建

```bash
# 生成技术指标（RSI/MACD/ATR/...）、异常提示（可选 LOF）、局部滑动归一化
python src/indicating.py

# 构建多目标：如 target_binarytrend / target_logreturn / target_drawdown_prob / ...
python src/target_config.py

# 多周期对齐合并（1d/4h 下放到 1h），产出合并后的训练数据（parquet/pkl）
python src/data_fusion.py
```

### 数据加载

* `data/load_dataset.py` 根据 `configs/model_config.yaml` 的 `data_path` 加载合并后的 **parquet/pkl**：

  * 构建 `pytorch_forecasting.TimeSeriesDataSet` + `DataLoader`
  * 自动生成目标名列表、(symbol, period) 粒度的回归目标标准化统计
* 可通过环境变量覆盖路径：

  ```bash
  export TFM_DATA_PATH=/abs/path/to/full_merged.pkl
  ```

---

## 快速上手（TL;DR）

```bash
# 1) 准备环境（示例）
conda create -n tft_flwr python=3.10 -y
conda activate tft_flwr
pip install -r requirements.txt   # 若提供；否则请逐项安装上文依赖

# 2) 生成数据
python src/indicating.py
python src/target_config.py
python src/data_fusion.py

# 3) 训练（从零）
python train_multi_tft.py --config configs/model_config.yaml

# 4) 可视化
tensorboard --logdir runs
```

---

## 训练流程

### 1. 从零开始

```bash
python train_multi_tft.py --config configs/model_config.yaml
```

> 自动在 `runs/<脚本名-YYYYMMDD_HHMM>/` 下创建：
>
> * `checkpoints/`（包含 best/top-k/last）
> * `lightning_logs/`（TensorBoard）
> * `configs/`（运行时配置快照）

### 2. 从断点恢复

```bash
python train_resume.py \
  --config configs/model_config.yaml \
  --ckpt runs/.../checkpoints/last.ckpt
```

> 恢复优化器、学习率调度器、当前 epoch 等，**无缝续训**。

### 3. 热启动微调（仅加载权重）

```bash
python warm_start_train.py \
  --config configs/model_config.yaml \
  --ckpt runs/.../checkpoints/best-epoch=XX.ckpt
```

> 不恢复优化器/epoch；可更换目标/损失/学习率，对**新任务**快速适配。

---

## 配置说明

### `model_config.yaml`

* **数据**：`data_path`, `targets`, `categoricals`, `reals`, `group_ids`, `time_idx`
* **窗口**：`max_encoder_length`（默认 96），`max_prediction_length`（默认 1）
* **模型**（TFT）：`hidden_size`, `lstm_layers`, `dropout`, `attention_head_size` 等
* **训练**：`batch_size`, `max_epochs`, `optimizer`, `lr`, `precision`, `num_workers`, `gradient_clip_val`
* **调度**：`onecycle` 参数（`pct_start/div_factor/final_div_factor`）
* **保存**：`monitor=val_composite_score`, `monitor_mode=max`, `top_k`, `save_last=true`
* **数据划分**：`val_mode`（`days` / `ratio`）、`focus_period`、`early_stop_patience`（如启用）
* **可选**：`resume_ckpt`（续训）、`warm_start_ckpt`（热启动）

### `weights_config.yaml`

* 各 `target_*` 的**基础损失权重**；可与（若实现）`loss_scheduler` 联动分阶段切换，例如：

  ```yaml
  stage_1:
    target_binarytrend: 1.0
    target_logreturn: 1.0
    target_drawdown_prob: 1.0

  stage_2:
    target_binarytrend: 1.5
    target_logreturn: 0.8
    target_drawdown_prob: 1.0
  ```

### `composite_score.yaml`

* **复合指标权重矩阵**，用于自定义选模策略（`val_composite_score`）：

  * 建议**显式 period** 标注（`@1h/@4h/@1d`），并确保与日志键名一致（如 `val_target_binarytrend_f1@1h`）。
  * 对“越小越好”的指标（如 RMSE/MAE），可**负权**或在 normalize 时反向。

示例：

```yaml
val_target_binarytrend_f1@1h: 0.25
val_target_binarytrend_f1@4h: 0.15
val_target_logreturn_rmse@1h: -0.20
val_target_drawdown_prob_auc@1d: 0.10
val_loss_for_ckpt: -0.30
```

---

## 联邦学习（Flower）

### 安装与本地模拟

```bash
cd flwr_tft
pip install -e .    # 以可编辑模式安装 Flower 应用（含入口）
```

**一键本地模拟（单机多进程）**：

```bash
cd flwr_tft
flwr run . local-simulation \
  --run-config "num-server-rounds=3 local-epochs=1 batch-size=16"
```

> 说明：`task.py` 负责**数据分片**（如 symbol × period 组合），`client_app.py` 负责**构造模型与数据**并执行本地训练，`server_app.py` 负责**聚合策略与日志**。

### 核心文件与职责

* `task.py`

  * 定义客户端数、每个客户端的数据**切分方式**（如按 `symbol/period` 分桶）
  * 提供 `build_model_and_data(config)`：返回 **LightningModule + DataLoader**
* `client_app.py`

  * Flower 客户端入口：封装 `fit/evaluate/get_parameters`
  * 兼容 **安全聚合（成对掩码）**：对出参梯度/参数进行掩码，再由服务器端聚合取消
* `server_app.py`

  * Flower 服务器入口：策略（加权平均/自定义）与**详细日志**（每轮客户端参与度、样本数、模型大小等）

> 如需真实多机，可改用**独立 server + 多 client 进程**，参考 Flower 官方“gRPC server/client”模式（把本地读取的数据路径切到各机节点）。

### 常见运行方式

1）**最小化仿真**（单机模拟多客户端）

```bash
flwr run . local-simulation --run-config "num-server-rounds=5 local-epochs=1 batch-size=64"
```

2）**控制参与率与评估频率**

```bash
flwr run . local-simulation --run-config "num-server-rounds=10 fraction-fit=0.5 fraction-evaluate=0.5 local-epochs=1"
```

3）**模拟网络特性**（丢包/延迟/带宽）

```bash
flwr run . local-simulation --run-config "num-server-rounds=5 p-drop=0.1 max-latency=200ms bandwidth-limit=50mbps"
```

> 注：具体参数名称与 Flower 版本有关，请以项目内 `pyproject.toml` / 代码为准。

### 可调参数与示例

可在 `pyproject.toml` 的 `[tool.flwr.app.config]` 或 `--run-config` 中设置：

* 训练轮次与采样：`num-server-rounds`, `fraction-fit`, `fraction-evaluate`
* 客户端本地训练：`local-epochs`, `batch-size`, `num-workers`
* 数据切分与任务：`num-clients`, `partitions`, `symbols`, `periods`（若在 `task.py` 暴露）
* 网络仿真：`p-drop`, `max-latency`, `bandwidth-limit`
* （可选）隐私：`dp-max-norm`, `dp-sigma`（若已实现 DP 钩子）

示例：

```bash
flwr run . local-simulation \
  --run-config "num-server-rounds=8 fraction-fit=0.75 local-epochs=1 batch-size=128 num-workers=4"
```

### 安全聚合与隐私

* **安全聚合**：本项目客户端内置**成对掩码**（pairwise mask）方案（同轮两两交换随机子掩码，使得单客户端更新不可被单独还原，服务器聚合后掩码相抵消）。
* **差分隐私（可选）**：若需要：

  * **客户端 DP**：在上传前执行裁剪 + 高斯噪声；
  * **服务器 DP**：在聚合后对全局更新再加噪；
  * 建议显式记录：`dp_max_norm / dp_sigma` 与**隐私会计**（ε, δ）日志（如已实现）。

> ⚠️ 提示：安全聚合 ≠ 差分隐私。若有强隐私需求，建议二者**同时**使用。

---

## 日志与运行目录

* 由 `utils/run_helper.prepare_run_dirs()` 自动生成：

```
runs/<脚本名-YYYYMMDD_HHMM>/
├─ checkpoints/        # best-epoch=...ckpt, top-k, last.ckpt
├─ lightning_logs/     # TensorBoard
└─ configs/            # 运行时配置快照（用于完全复现）
```

* 启动 TensorBoard：

  ```bash
  tensorboard --logdir runs
  ```

---

## 最佳实践与建议

* **显存/批大小**：数据量（\~30万行、100+特征）下，建议从 `batch_size=128~256` 起步；OOM 时优先减 `batch_size`，并启用 `precision="16-mixed"`。
* **OneCycleLR**：`pct_start` 选 0.2\~0.4；热启动无需重复长 warmup，直接小 LR 微调。
* **损失设计**：重尾/尖峰型回归用 `SmoothL1Loss`；极不平衡分类用 `WeightedBCE(auto_pos_weight=True)`。
* **复合指标**：为避免“量纲/方向”混乱，回归指标使用负权或在归一化中做反向；建议给 `val_loss_for_ckpt` 一定权重（0.2\~0.4）增强稳定性。
* **数据一致性**：长表结构（特征不带后缀，`symbol/period` 两列标识）；慢频向快频**下放**；宏观/全市场变量**同一时间对所有 symbol 赋同值**。
* **联邦切分**：尽量按 **(symbol, period)** 或 **交易所/市场** 切分，确保各客户端样本**长度 ≥ max\_encoder\_length + max\_prediction\_length**（默认 97）。

---

## 常见问题（FAQ）

**Q1：验证时出现 `default_collate` 报错？**
A：通常是某些 `(symbol, period)` 组长度 < 97，无法形成有效序列。请在加载前**筛除短组**或缩小 `max_encoder_length`。

**Q2：验证集中出现未知类别（symbol/period）导致 NaN？**
A：验证集中出现了训练未见过的类别。请确保**划分前**全局类别一致，或在编码器中注册完整类别集合。

**Q3：`composite_score.yaml` 不生效/日志键不匹配？**
A：检查日志键名（如 `val_target_logreturn_rmse@1h`），确保与权重矩阵中的键一致；`utils/composite.py` 会跳过不存在的条目并警告一次。

**Q4：Flower 仿真时客户端 0 参与/样本为 0？**
A：检查 `task.py` 的分片是否为空（比如某个 `symbol×period` 实际没数据或不足 97 步）。可临时打印每分片的样本统计。

**Q5：热启动时某些层名不匹配？**
A：使用 `utils/checkpoint_utils.py` 的**选择性加载**，忽略 shape 不一致的 head 层或增删的目标头。

---

## 其他实用脚本

* `utils/stage_summary.py`：记录训练阶段最优指标（`val_composite_score`、`val_loss`）
* `utils/composite.py`：复合指标构建与键名过滤/补齐
* `src/csv2Parquet.py` / `src/csv2Pkl.py`：格式转换
* `src/feature_healthcheck.py`：特征健康检查与可视化（缺失/分布/相关性等）

---

## 贡献与许可证

* 欢迎提交 **Issue / PR** 共建项目（新特征脚本、新联邦策略、隐私增强模块、评估可视化等）。
* 许可证：请根据你的实际需求指定（如 MIT / Apache-2.0）。如未指定，默认遵循原仓库约定。




