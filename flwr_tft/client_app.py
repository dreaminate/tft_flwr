# client_app.py
from __future__ import annotations
from typing import Dict, Any,List
import re
import random
import time
import hashlib
import numpy as np
import torch
from flwr.client import NumPyClient, ClientApp
from flwr.common import Context
import os
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
print("[BOOT] CVD=", os.environ.get("CUDA_VISIBLE_DEVICES"))
from task import (
    build_model_and_data,
    get_weights, set_weights,
    local_train_validate, local_validate,
    PARTITIONS,
)
def _pairwise_masks_with_seed(shapes: List[tuple], self_pid: int, peer_pids: List[int], round_seed: int) -> List[np.ndarray]:
    """为每个张量形状生成两两抵消的加性掩码（仅模拟 SecAgg）。
    同一 (round_seed, pid_pair) 在两端生成相同随机向量；小 pid 加、大 pid 减，求和后刚好为 0。
    """
    masks = [np.zeros(s, dtype=np.float32) for s in shapes]
    for peer in peer_pids:
        if peer == self_pid:
            continue
        lo, hi = (self_pid, peer) if self_pid < peer else (peer, self_pid)
        # 以 (round_seed, lo, hi) 派生确定性种子
        h = hashlib.sha256(f"{round_seed}:{lo}:{hi}".encode()).digest()
        seed = int.from_bytes(h[:8], "little", signed=False)
        rng = np.random.default_rng(seed)
        for i, s in enumerate(shapes):
            m = rng.standard_normal(size=s).astype(np.float32, copy=False)
            if self_pid == lo:
                masks[i] += m
            else:
                masks[i] -= m
    return masks


def _infer_partition_id(context: Context) -> int:
    # 优先从 node_config 取；否则从 node_id 末尾数字推断；最后兜底 0
    if "partition-id" in context.node_config:
        pid = int(context.node_config["partition-id"])
    else:
        m = re.search(r"(\d+)$", str(context.node_id))
        pid = int(m.group(1)) if m else 0
    # 映射到有效范围
    return pid % len(PARTITIONS)


class TFTClient(NumPyClient):
    def __init__(self, partition_id: int, run_cfg: Dict[str, Any]):
        self.partition_id = partition_id
        self.local_epochs = int(run_cfg.get("local-epochs", 1))
        self.batch_size = int(run_cfg.get("batch-size", 16))
        self.num_workers = int(run_cfg.get("num-workers", 1))

        self.model, self.train_loader, self.val_loader = build_model_and_data(
            partition_id=self.partition_id,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
        # Simulation knobs
        self.drop_rate = float(run_cfg.get("p-drop", 0.0))
        self.latency = float(run_cfg.get("max-latency", 0.0))
        self.jitter = float(run_cfg.get("latency-jitter", 0.0))
        self.bandwidth = float(run_cfg.get("bandwidth-limit", 0.0))
        self.version = str(run_cfg.get("client-version", "1"))
        self.expected_version = str(run_cfg.get("expected-version", self.version))

    # --- Flower NumPyClient API ---
    def get_parameters(self, config):
        return get_weights(self.model)

    def fit(self, parameters, config):
        # 是否打印仅模拟用的统计（生产不会传）
        debug_priv = bool(int(config.get("debug-priv", 0)))

        # 0) 获取本轮参与者与回合种子（服务端需在 configure_fit 注入）
        pids_str = str(config.get("participant_ids", ""))
        round_seed = int(config.get("round_seed", 0))
        peer_pids = [int(x) for x in pids_str.split(",") if x.strip()]

        # 1) 取到下发的全局参数 W_g（保持 float32）
        Wg = [w.astype(np.float32, copy=True) for w in parameters]
        set_weights(self.model, Wg)

        # 2) 本地训练，得到本地权重 W_i
        if debug_priv:
            print("[SIM-ONLY][生产不会传] FIT_START", flush=True)
        metrics = local_train_validate(
            self.model, self.train_loader, self.val_loader, local_epochs=self.local_epochs
        )
        Wi = [w.astype(np.float32, copy=False) for w in get_weights(self.model)]

        # 3) 计算 ΔW_i
        dW = [wi - wg for wi, wg in zip(Wi, Wg)]

        # 4) 乘以样本数 n_i → S_i = n_i * ΔW_i
        n_i = int(len(getattr(self.train_loader, "dataset", [])))
        S = [dw * float(n_i) for dw in dW]

        # 5) 成对掩码（仅当 server 传了 participant_ids 和 round_seed）
        if peer_pids and round_seed:
            shapes = [a.shape for a in S]
            masks = _pairwise_masks_with_seed(shapes, self.partition_id, peer_pids, round_seed)
            S = [s + m for s, m in zip(S, masks)]

        # 仅模拟打印：统计信息（不含原始值）
        if debug_priv:
            def _stats(arrs):
                return [{"shape": a.shape, "l2": float(np.linalg.norm(a)),
                        "mean": float(np.mean(a)), "std": float(np.std(a))} for a in arrs[:5]]
            try:
                print("[SIM-ONLY][生产不会传] delta_stats(sample)=", _stats(dW), flush=True)
                print("[SIM-ONLY][生产不会传] masked_S_stats(sample)=", _stats(S), flush=True)
            except Exception:
                pass

        # 6) 网络模拟 & 返回 “被掩码的 nᵢ·ΔWᵢ”
        payload_bytes = sum(a.nbytes for a in S)
        self._simulate_network(payload_bytes)

        if debug_priv:
            print(f"[SIM-ONLY][生产不会传] FIT_DONE payloadMB={payload_bytes/1e6:.3f}", flush=True)

        # 关键：返回 S（不是 W_i），并将 num_examples 设为 n_i
        # 服务器端需要用“sum(S) / sum(n_i)”来更新全局。
        return S, n_i, {"report_mode": "delta_sum"}


    def evaluate(self, parameters, config):
        set_weights(self.model, parameters)
        self._simulate_network()
        results = local_validate(self.model, self.val_loader)
        loss = float(results.get("val_loss", 0.0))
        num_examples = len(getattr(self.val_loader, "dataset", []))
        return loss, num_examples, results
    # --- Simulation helpers -------------------------------------------------
    def _simulate_network(self, payload: int = 0) -> None:
        if random.random() < self.drop_rate:
            raise RuntimeError("simulated random disconnect")
        delay = self.latency + random.uniform(0, self.jitter)
        if delay > 0:
            time.sleep(delay)
        if self.bandwidth > 0 and payload > 0:
            time.sleep(payload / self.bandwidth)

def client_fn(context: Context):
    partition_id = _infer_partition_id(context)
    run_cfg = context.run_config
    return TFTClient(partition_id=partition_id, run_cfg=run_cfg).to_client()


app = ClientApp(client_fn)
