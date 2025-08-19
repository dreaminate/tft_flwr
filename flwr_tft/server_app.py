# server_app.py
from __future__ import annotations
from typing import Dict, Any

from flwr.server import ServerApp, ServerAppComponents
from flwr.common import Context, ndarrays_to_parameters
from flwr.server.strategy import FedAvg

# 兼容不同 Flower 版本的 ServerConfig 位置
try:
    from flwr.server import ServerConfig  # 新版
except Exception:  # pragma: no cover
    from flwr.server.server import ServerConfig  # 旧版

from task import build_model_and_data, get_weights


def server_fn(context: Context) -> ServerAppComponents:
    run_cfg: Dict[str, Any] = context.run_config
    num_rounds = int(run_cfg.get("num-server-rounds", 3))
    fraction_fit = float(run_cfg.get("fraction-fit", 1.0))
    min_available = int(run_cfg.get("min-available-clients", 2))

    # 用任意一个分区的模型来初始化全局参数（结构一致即可）
    init_model, _, _ = build_model_and_data(partition_id=0)
    init_params = ndarrays_to_parameters(get_weights(init_model))

    strategy = FedAvg(
        fraction_fit=fraction_fit,
        fraction_evaluate=1.0,
        min_available_clients=min_available,
        initial_parameters=init_params,
    )
    cfg = ServerConfig(num_rounds=num_rounds)
    return ServerAppComponents(strategy=strategy, config=cfg)


app = ServerApp(server_fn=server_fn)
