# server_app.py
from __future__ import annotations
from typing import Dict, Any, List, Tuple
import re
from flwr.server import ServerApp, ServerAppComponents
from flwr.common import Context, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.strategy import FedAvg

# 兼容不同 Flower 版本的 ServerConfig 位置
try:
    from flwr.server import ServerConfig  # 新版
except Exception:  # pragma: no cover
    from flwr.server.server import ServerConfig  # 旧版

from task import build_model_and_data, get_weights
class SecureAggregationStrategy(FedAvg):
    """FedAvg strategy which informs clients of participants for masking."""

    @staticmethod
    def _infer_pid(cid: str) -> int:
        m = re.search(r"(\d+)$", str(cid))
        return int(m.group(1)) if m else 0

    def configure_fit(self, server_round, parameters, client_manager):
        cfg = super().configure_fit(server_round, parameters, client_manager)
        pids = [self._infer_pid(client_proxy.cid) for client_proxy, _ in cfg]
        pids_str = ",".join(str(pid) for pid in pids)
        for _, fit_ins in cfg:
            fit_ins.config["participant_ids"] = pids_str
        return cfg
class DetailedLoggingStrategy(SecureAggregationStrategy):
    """Strategy which prints rich information for each round."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._current_clients: List[str] = []

    def configure_fit(self, server_round, parameters, client_manager):  # type: ignore[override]
        cfg = super().configure_fit(server_round, parameters, client_manager)
        self._current_clients = [client.cid for client, _ in cfg]
        print(f"[ROUND {server_round}] selected clients: {self._current_clients}")
        return cfg

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[Any, Any]],
        failures: List[Tuple[Any, BaseException]],
    ):  # type: ignore[override]
        sample_info = []
        responded = []
        for client, fit_res in results:
            params = parameters_to_ndarrays(fit_res.parameters)
            payload = int(sum(p.nbytes for p in params))
            sample_info.append(
                {
                    "cid": client.cid,
                    "examples": fit_res.num_examples,
                    "payload": payload,
                }
            )
            responded.append(client.cid)
        # ``failures`` can contain bare exceptions or ``(client, exc)`` tuples
        failed_ids = []
        timeout_cnt = 0
        for failure in failures:
            if isinstance(failure, tuple):
                client, err = failure
                failed_ids.append(client.cid)
            else:
                err = failure
            if isinstance(err, TimeoutError):
                timeout_cnt += 1
        dropped = list(
            set(self._current_clients) - set(responded) - set(failed_ids)
        )
        print(
            f"[ROUND {server_round}] samples: {sample_info}, failures: {failed_ids}, ",
            f"dropped: {dropped}, timeouts: {timeout_cnt}",
        )
        return super().aggregate_fit(server_round, results, failures)

def server_fn(context: Context) -> ServerAppComponents:
    run_cfg: Dict[str, Any] = context.run_config
    num_rounds = int(run_cfg.get("num-server-rounds", 2))
    fraction_fit = float(run_cfg.get("fraction-fit", 0.5))
    min_available = int(run_cfg.get("min-available-clients", 1))
    min_evaluate_clients = int(run_cfg.get("min-evaluate-clients", 1))
    fraction_evaluate = float(run_cfg.get("fraction-evaluate", 0.5))
    min_fit_clients = int(run_cfg.get("min-fit-clients", 1))  
    # 用任意一个分区的模型来初始化全局参数（结构一致即可）
    init_model, _, _ = build_model_and_data(partition_id=0)
    init_params = ndarrays_to_parameters(get_weights(init_model))

    strategy = DetailedLoggingStrategy(
        fraction_fit=fraction_fit,
        fraction_evaluate=fraction_evaluate,
        min_available_clients=min_available,
        min_fit_clients=min_fit_clients,
        initial_parameters=init_params,
     min_evaluate_clients=min_evaluate_clients,
    )
    cfg = ServerConfig(num_rounds=num_rounds)
    return ServerAppComponents(strategy=strategy, config=cfg)


app = ServerApp(server_fn=server_fn)
