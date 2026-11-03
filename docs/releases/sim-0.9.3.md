# mindie-sim 0.9.3

本版本修复 **Ascend A5 DT（`a5-pod-dt` / `a5-server-dt` / `a5`）下 MoE `dispatch` 与 `combine` 集合通信耗时翻倍与量化策略失配问题**，对齐 `vllm-ascend` 的专家路由量化传输契约，并优化大规模 EP（如 `DP64 TP1 EP64`）下的算子分析与队列流量绑定耗时。

- **修复 `per_port` 双向全双工链路字节在效率分析中的 Tx+Rx 双计问题（`mindie_sim/prediction/efficiency.py`）**：
  - 在 `analyze_operator_row` 中，当集合通信走 `per_port` 物理端口资源（`:tx:port` 与 `:rx:port`）时，仅统计单向发送端口（`:tx:port`）并按参与 Rank 数折算单卡逻辑 `link_bytes` 与 `t_link_ideal_ms`，修复了原先将全群組 Tx 与 Rx 端口字节相加导致 `link_bytes` 与 `t_link_ideal_ms` 翻倍（2x）的问题，同时保留 `semantics="shared"` 共享总线下的完整链路负载语义。
- **修复 `alltoall` / `alltoallv` 集合通信固定启动时延按 $(EP - 1)$ 线性累加的问题（`mindie_sim/prediction/communication.py`, `topology.py`, `execution.py`）**：
  - 在 `CommunicationWork` 中显式记录单跳路径时延 `path_latency_ms`，并在 `Execution._context` 中对 `alltoall` / `alltoallv` 采用单步并发交换启动时延，消除 `EP=64` 下将单次启动开销误乘 63 倍的问题。
- **对齐 `vllm-ascend` 的 MoE `dispatch` / `combine` 量化通信策略（`mindie_sim/prediction/moe_transport.py`, `feedforward.py`, `deepep.py`, `compose.py`）**：
  - 新增 `default_dispatch_spec` 与 `default_exchange_contract`：`compose_case` 默认采用 `moe_dispatch_quantization="routed_expert"`，在 `dispatch` 前插入 `pack.quantize`（`dynamic_quantize`），使 `expert_dispatch` 跟随 `routed_expert` 权重格式传输量化激活（如 `fp8` + `block-128` 缩放因子或 `int8` + 逐 Token 缩放因子），`expert_combine` 保持 `bf16` 激活规约，并在 `dispatch` 已完成激活量化时自动省略冗余的 `expert_up.quantize` 算子。
- **扩展 `GLM-5.3` 模型目录别名与大规模 EP 仿真加速（`mindie_sim/model_catalog.py`, `prediction/operator_analysis.py`, `prediction/cohort_traffic.py`）**：
  - 在模型目录中新增 `glm5.3`、`GLM5.3`、`glm-5`、`GLM-5`、`glm5.3-flash` 别名映射。
  - 对非 `balanced` 均匀副本复用 `_target_ops` 图构建并缓存同通信组 `PeerTraffic`，将 64 卡 `GLM-5.3` 算子级网络效率分析耗时从 `~25s` 降至 `1.6s`。
