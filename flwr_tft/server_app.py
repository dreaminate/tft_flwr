# server_app.py
from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

from flwr.server import ServerApp, ServerAppComponents
from flwr.common import Context, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.strategy import FedAvg

# 兼容不同 Flower 版本的 ServerConfig 位置
try:
    from flwr.server import ServerConfig  # 新版
except Exception:  # pragma: no cover
    from flwr.server.server import ServerConfig  # 旧版

from task import build_model_and_data, get_weights

print(f"[server_app] module loaded from: {__file__}", flush=True)

# ------------------------ 策略 ------------------------
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
        print("[strategy] DetailedLoggingStrategy constructed", flush=True)

    def configure_fit(self, server_round, parameters, client_manager):  # type: ignore[override]
        cfg = super().configure_fit(server_round, parameters, client_manager)
        self._current_clients = [client.cid for client, _ in cfg]
        avail = client_manager.num_available()
        print(
            f"[R{server_round}] available={avail} picked={len(self._current_clients)} "
            f"cids={self._current_clients}",
            flush=True,
        )
        return cfg

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[Any, Any]],
        failures: List[Any],
    ):  # type: ignore[override]
        sample_info: List[Dict[str, Any]] = []
        responded: List[str] = []

        total_examples = 0
        total_payload_mb = 0.0

        for client, fit_res in results:
            params = parameters_to_ndarrays(fit_res.parameters)
            payload_bytes = int(sum(getattr(p, "nbytes", 0) for p in params))
            payload_mb = payload_bytes / 1e6
            sample_info.append(
                {
                    "cid": client.cid,
                    "examples": int(getattr(fit_res, "num_examples", 0)),
                    "payloadMB": round(payload_mb, 3),
                }
            )
            responded.append(client.cid)
            total_examples += int(getattr(fit_res, "num_examples", 0))
            total_payload_mb += payload_mb

        # 兼容多种 failures 形态：异常对象 或 (client, exc)
        failed_ids: List[str] = []
        timeout_cnt = 0
        failure_msgs: List[str] = []
        for failure in failures:
            if isinstance(failure, tuple) and len(failure) == 2:
                client, exc = failure
                failed_ids.append(getattr(client, "cid", "unknown"))
            else:
                exc = failure
            failure_msgs.append(f"{type(exc).__name__}: {exc}")
            if isinstance(exc, TimeoutError):
                timeout_cnt += 1

        dropped = list(set(self._current_clients) - set(responded) - set(failed_ids))

        print(
            f"[R{server_round}] reports={len(results)}, examples={total_examples}, "
            f"payload≈{total_payload_mb:.2f}MB",
            flush=True,
        )
        print(
            f"[R{server_round}] samples={sample_info} | failures={failed_ids} "
            f"| dropped={dropped} | timeouts={timeout_cnt}",
            flush=True,
        )
        if failure_msgs:
            print(f"[R{server_round}] failure_msgs={failure_msgs}", flush=True)

        # 打印一个回包的张量形状示例，便于确认模型结构（不打印数值）
        if results:
            try:
                arrs = parameters_to_ndarrays(results[0][1].parameters)
                shapes = [getattr(a, "shape", None) for a in arrs[:5]]
                print(f"[R{server_round}] example tensor shapes: {shapes} ...", flush=True)
            except Exception as e:
                print(f"[R{server_round}] shape_inspect_error: {e}", flush=True)

        return super().aggregate_fit(server_round, results, failures)

    def aggregate_evaluate(self, server_round, results, failures):  # type: ignore[override]
        print(
            f"[R{server_round}] eval_reports={len(results)}, eval_failures={len(failures)}",
            flush=True,
        )
        return super().aggregate_evaluate(server_round, results, failures)


# ------------------------ ServerApp ------------------------
def server_fn(context: Context) -> ServerAppComponents:
    run_cfg: Dict[str, Any] = context.run_config
    num_rounds = int(run_cfg.get("num-server-rounds", 2))
    fraction_fit = float(run_cfg.get("fraction-fit", 0.5))
    fraction_evaluate = float(run_cfg.get("fraction-evaluate", 0.5))
    min_available = int(run_cfg.get("min-available-clients", 1))
    min_fit_clients = int(run_cfg.get("min-fit-clients", 1))
    min_evaluate_clients = int(run_cfg.get("min-evaluate-clients", 1))

    # 用任意一个分区的模型来初始化全局参数（结构一致即可）
    init_model, _, _ = build_model_and_data(partition_id=0)
    init_params = ndarrays_to_parameters(get_weights(init_model))
    print("[server_fn] built init model and params", flush=True)

    strategy = DetailedLoggingStrategy(
        fraction_fit=fraction_fit,
        fraction_evaluate=fraction_evaluate,
        min_available_clients=min_available,
        min_fit_clients=min_fit_clients,
        min_evaluate_clients=min_evaluate_clients,
        initial_parameters=init_params,
    )
    print("[server_fn] using DetailedLoggingStrategy", flush=True)

    cfg = ServerConfig(num_rounds=num_rounds)
    return ServerAppComponents(strategy=strategy, config=cfg)


app = ServerApp(server_fn=server_fn)
print("[server_app] app object created", flush=True)
