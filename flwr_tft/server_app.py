# server_app.py
from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

import numpy as np
from flwr.server import ServerApp, ServerAppComponents
from flwr.common import (
    Context,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
    Scalar,
)
from flwr.server.strategy import FedAvg

# 兼容不同 Flower 版本的 ServerConfig 位置
try:
    from flwr.server import ServerConfig  # 新版
except Exception:  # pragma: no cover
    from flwr.server.server import ServerConfig  # 旧版

from task import build_model_and_data, get_weights

print(f"[server_app] module loaded from: {__file__}", flush=True)

# ------------------------ 小工具 ------------------------
def _sample_stats(arrs: List[np.ndarray], k: int = 5) -> List[Dict[str, Any]]:
    """仅打印统计而非原始值：shape / l2 / mean / std。"""
    out: List[Dict[str, Any]] = []
    for a in arrs[:k]:
        try:
            out.append(
                {
                    "shape": tuple(getattr(a, "shape", ())),
                    "l2": float(np.linalg.norm(a)),
                    "mean": float(np.mean(a)),
                    "std": float(np.std(a)),
                }
            )
        except Exception:
            out.append({"shape": tuple(getattr(a, "shape", ())), "error": True})
    return out


def _weighted_avg(metrics_list) -> Dict[str, float]:
    """按样本数加权平均：
    - 接受 [(metrics_dict, n)], [(n, metrics_dict)], 或包含多余元素的元组
    - 自动忽略 NaN/Inf
    - 找不到 n 时按 1 计
    """
    import math

    def _extract(item):
        md = None
        n = None
        # 元组/列表：从中“找一个 dict”和“找一个 int/float”
        if isinstance(item, (list, tuple)):
            for elem in item:
                if md is None and isinstance(elem, dict):
                    md = elem
                elif n is None and isinstance(elem, (int, float)):
                    n = int(elem)
        elif isinstance(item, dict):
            md = item

        # 兜底：从字典里尝试拿 'num_examples'/'n' 之类；再兜底 1
        if n is None:
            if isinstance(md, dict):
                for key in ("num_examples", "n", "N"):
                    if key in md:
                        try:
                            n = int(md[key])
                            break
                        except Exception:
                            pass
        if n is None:
            n = 1
        return md or {}, int(n)

    total = 0
    acc: Dict[str, float] = {}

    for item in metrics_list or []:
        md, n = _extract(item)
        if n <= 0:
            continue
        for k, v in md.items():
            try:
                val = float(v)
            except Exception:
                continue
            if not math.isfinite(val):
                continue  # 跳过 nan/inf
            acc[k] = acc.get(k, 0.0) + val * n
        total += n

    if total <= 0:
        return {}
    return {k: acc[k] / float(total) for k in acc}


def _sum_ndarrays(list_of_list: List[List[np.ndarray]]) -> List[np.ndarray]:
    """按位置求和一组张量列表。"""
    acc: List[np.ndarray] | None = None
    for arrs in list_of_list:
        if acc is None:
            acc = [a.astype(np.float32, copy=True) for a in arrs]
        else:
            for i, a in enumerate(arrs):
                acc[i] += a.astype(np.float32, copy=False)
    assert acc is not None
    return acc


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
            # 将 debug-priv 透传给客户端（便于客户端也打印“仅模拟”的统计；客户端可选使用）
            fit_ins.config["debug-priv"] = int(getattr(self, "_debug_priv", 0))
        return cfg


class DetailedLoggingStrategy(SecureAggregationStrategy):
    """Strategy which prints rich information for each round."""

    def __init__(self, *args, debug_priv: bool = False, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._current_clients: List[str] = []
        self._debug_priv = bool(debug_priv)
        self._last_global = None  # 缓存本轮下发前的全局参数
        print("[strategy] DetailedLoggingStrategy constructed", flush=True)

    def configure_fit(self, server_round, parameters, client_manager):  # type: ignore[override]
        # 保存“本轮下发前”的全局，用于之后把平均ΔW加回去
        self._last_global = parameters

        cfg = super().configure_fit(server_round, parameters, client_manager)

        # 本轮选中客户端列表与可用数
        self._current_clients = [client.cid for client, _ in cfg]
        avail = client_manager.num_available()

        # 参与者ID（从 cid 尾号推断）
        pids = [self._infer_pid(cid) for cid in self._current_clients]
        pids_str = ",".join(str(pid) for pid in pids)

        # 本轮随机种子（用于客户端成对掩码）
        import secrets
        round_seed = int(secrets.randbits(32))

        # 向每个客户端下发本轮配置
        for _, fit_ins in cfg:
            fit_ins.config["participant_ids"] = pids_str
            fit_ins.config["round_seed"] = round_seed
            fit_ins.config["report_mode"] = "delta_sum"  # 客户端上传 n_i * ΔW_i
            fit_ins.config["debug-priv"] = int(getattr(self, "_debug_priv", 0))

        # 日志
        print(
            f"[R{server_round}] available={avail} picked={len(self._current_clients)} "
            f"cids={self._current_clients}",
            flush=True,
        )
        print(
            f"[SIM-ONLY][生产不会传] [R{server_round}] injected "
            f"report_mode=delta_sum, round_seed={round_seed}, "
            f"participant_ids=[{pids_str}]",
            flush=True,
        )
        return cfg

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[Any, Any]],
        failures: List[Any],
    ):  # type: ignore[override]
        # ---- 汇总日志（与你原来一致）----
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

        # 打印一个回包的张量形状示例（不打印数值）
        if results:
            try:
                arrs = parameters_to_ndarrays(results[0][1].parameters)
                shapes = [getattr(a, "shape", None) for a in arrs[:5]]
                print(f"[R{server_round}] example tensor shapes: {shapes} ...", flush=True)
            except Exception as e:
                print(f"[R{server_round}] shape_inspect_error: {e}", flush=True)

        # ===== 仅模拟用的敏感明细统计（生产不会传输/存储）=====
        if self._debug_priv and results:
            for client, fr in results:
                arrs = parameters_to_ndarrays(fr.parameters)
                stats = _sample_stats(arrs, k=5)
                print(
                    f"[SIM-ONLY][生产不会传] [R{server_round}] cid={client.cid} "
                    f"update_stats(sample)={stats}",
                    flush=True,
                )

        # ---- 关键：把 Sᵢ = nᵢ·ΔWᵢ 聚合成新全局 ----
        if not results or self._last_global is None:
            return None  # 交给上层处理空结果

        # 上轮全局 W_g
        Wg = parameters_to_ndarrays(self._last_global)

        # S = Σ (nᵢ·ΔWᵢ)，N = Σ nᵢ
        sum_S_list: List[List[np.ndarray]] = []
        N = 0
        for _, fr in results:
            Si = parameters_to_ndarrays(fr.parameters)  # 客户端上传的 nᵢ·ΔWᵢ（若做了成对掩码，求和会抵消）
            sum_S_list.append(Si)
            N += int(getattr(fr, "num_examples", 0))
        S = _sum_ndarrays(sum_S_list)
        N = max(1, N)

        # 平均 ΔW = S / N，新全局 = W_g + 平均 ΔW
        avg_dW = [s / float(N) for s in S]
        new_global = [wg.astype(np.float32) + d for wg, d in zip(Wg, avg_dW)]
        new_params = ndarrays_to_parameters(new_global)

        # 只返回聚合后的全局与简要指标（不暴露单点）
        metrics: Dict[str, Scalar] = {"num_clients": len(results), "num_examples": int(N)}
        return new_params, metrics

    def aggregate_evaluate(self, server_round, results, failures):  # type: ignore[override]
        detail: List[Dict[str, Any]] = []
        for client, eval_res in results:
            detail.append(
                {
                    "cid": client.cid,
                    "n": int(getattr(eval_res, "num_examples", 0)),
                    "loss": float(getattr(eval_res, "loss", 0.0)),
                    "metrics": getattr(eval_res, "metrics", {}),
                }
            )
        print(
            f"[R{server_round}] eval_reports={len(results)}, eval_failures={len(failures)}",
            flush=True,
        )
        if detail:
            print(f"[R{server_round}] eval_detail={detail}", flush=True)
        # 评估阶段仍走父类默认聚合（或你也可自定义）
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
    debug_priv = bool(int(run_cfg.get("debug-priv", 0)))  # ← 开关（默认关）

    # 用任意一个分区的模型来初始化全局参数（结构一致即可）
    init_model, _, _ = build_model_and_data(partition_id=0)
    init_params = ndarrays_to_parameters(get_weights(init_model))
    print("[server_fn] built init model and params", flush=True)

    # 可选：打印初始化参数总体积与形状示例（非敏感）
    try:
        arrs0 = parameters_to_ndarrays(init_params)
        init_mb = sum(getattr(a, "nbytes", 0) for a in arrs0) / 1e6
        print(
            f"[INIT] global param payload≈{init_mb:.2f}MB, "
            f"shapes_sample={[getattr(a,'shape',None) for a in arrs0[:5]]}",
            flush=True,
        )
    except Exception as e:
        print(f"[INIT] inspect error: {e}", flush=True)

    strategy = DetailedLoggingStrategy(
        fraction_fit=fraction_fit,
        fraction_evaluate=fraction_evaluate,
        min_available_clients=min_available,
        min_fit_clients=min_fit_clients,
        min_evaluate_clients=min_evaluate_clients,
        initial_parameters=init_params,
        evaluate_metrics_aggregation_fn=_weighted_avg,  # ✅ 消除 WARNING
        fit_metrics_aggregation_fn=_weighted_avg,        # ✅ 训练阶段也聚合
        debug_priv=debug_priv,                           # ✅ 仅模拟下打印明细
    )
    print("[server_fn] using DetailedLoggingStrategy", flush=True)

    cfg = ServerConfig(num_rounds=num_rounds)
    return ServerAppComponents(strategy=strategy, config=cfg)


app = ServerApp(server_fn=server_fn)
print("[server_app] app object created", flush=True)
