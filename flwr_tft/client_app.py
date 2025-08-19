# client_app.py
from __future__ import annotations
from typing import Dict, Any
import re
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
        

    # --- Flower NumPyClient API ---
    def get_parameters(self, config):
        return get_weights(self.model)

    def fit(self, parameters, config):
        set_weights(self.model, parameters)
        metrics = local_train_validate(
            self.model, self.train_loader, self.val_loader, local_epochs=self.local_epochs
        )
        num_examples = len(getattr(self.train_loader, "dataset", []))
        return get_weights(self.model), num_examples, metrics

    def evaluate(self, parameters, config):
        set_weights(self.model, parameters)
        results = local_validate(self.model, self.val_loader)
        loss = float(results.get("val_loss", 0.0))
        num_examples = len(getattr(self.val_loader, "dataset", []))
        return loss, num_examples, results


def client_fn(context: Context):
    partition_id = _infer_partition_id(context)
    run_cfg = context.run_config
    return TFTClient(partition_id=partition_id, run_cfg=run_cfg).to_client()


app = ClientApp(client_fn)
