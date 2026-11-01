# mindie-sim 0.9.0

本版本严格遵循“封闭世界公理底座（`mindie-sim`）vs. 开放世界经验决策（`Agent + Knowledge + Profiler`）”的架构边界，补齐确定性 Case 合成器、CLI/MCP 全能力暴露、Wave 5 模型目录与 HuggingFace 细粒度量化排除解析、多维并行拓扑与 P/D 分离精确 Pareto 求解器、多 DP 上下文偏斜/MoE 负载不均衡/Chunked-Prefill 干扰屏障物理力学，以及显存（HBM）容量与异步调度（`async_scheduling`）延迟释放 KV 块的公理化建模。

- **显存（HBM）与异步调度（`async_scheduling`）延迟释放公理化建模（`mindie_sim/prediction/capacity.py`, `scheduler.py`, `efficiency.py`）**：
  - `Memory`（`capacity.py`）：显式建模 `async_scheduling=True` 下的三大物理显存开销——（1）跨步重叠准备的 Ping-Pong 双缓冲前向输入/残差/输出工作区（`async_workspace_overhead_bytes_per_stage_rank`）；（2）前序步未完成采样/EOS 校验导致的完成请求 KV Block 延迟 1 步释放（`async_delayed_release_steps` 与 `peak_delayed_release_cache_bytes_per_replica_stage`）；（3）在飞步未退役时的前瞻槽位预留（`async_in_flight_lookahead_steps`）。
  - `Scheduler`（`scheduler.py`）：在离散事件仿真中精确执行 `async_scheduling=True` 下的 1 步延迟 KV 释放与前瞻槽位预留，并在仅剩单槽位等待延迟释放时自动等待前序步释放而非误判永久拒绝。
  - `solve_hardware_sla_throughput` & `_vram_max_sequences_per_dp`（`efficiency.py`）：联合求解同步模式（`b_vram_max_sync`）与异步模式（`b_vram_max_async`）的最大显存并发上限，输出 `async_scheduling_memory` 账本及关闭异步调度可释放的额外并发度 `delta_b_vram_from_disabling_async` 与可行吞吐收益。
- **确定性 Case 合成器与 CLI 全能力暴露（`mindie_sim/compose.py`, `mindie_sim/__main__.py`）**：
  - 新增 `compose_case` 与 `resolve_hardware_sku`（支持 `a2`/`a3`/`a5` 等硬件别名、目录/模型名/自定义路径、并行维度校验、量化与合成层数/专家数覆盖及 `--async-scheduling` 开关）。
  - `mindie-sim` CLI 新增 `compose`、`efficiency`、`sla`、`sweep`、`skew` 五大子命令，实现与 Python API 的全对称暴露。
- **Wave 5 模型目录与 HuggingFace 角色级量化排除解析（`mindie_sim/model_catalog.py`, `mindie_sim/catalog/hf_config.py`）**：
  - 内置 `deepseek-v2-lite`、`deepseek-v3.2-prune-3l`、`qwen3-30b-a3b`、`qwen3.5-27b` 四个模型配置与别名映射，并新增 `customize_model_spec` 支持任意层数/专家数/Top-K 的确定性结构派生。
  - `hf_config.py` 新增 `resolve_role_exclusions`，自动解析 HuggingFace `quantization_config` 中的 `modules_to_not_convert` / `ignore` 列表并映射为 `unquantized_roles`（如 `shared_expert`、`attention`、`dense_ffn`、`router`）。
- **多维并行切分（DP/TP/EP/PP/PCP）与 P/D 分离配比精确 Pareto 求解器（`mindie_sim/sweep.py`）**：
  - 新增 `enumerate_parallel_candidates`、`sweep_topology_grid`、`sweep_pd_disaggregation_ratios` 与 `compute_pareto_frontier`，在给定卡数、头数/专家数整除约束、HBM 容量与 TTFT/TPOT SLA 约束下精确求解无支配 Pareto 前沿与 P/D 最小卡数配比。
- **多 DP 上下文偏斜、MoE 专家路由失衡（含 EPLB）与 Chunked-Prefill 干扰屏障物理力学（`mindie_sim/prediction/skew_analysis.py`）**：
  - 新增 `analyze_multi_dp_barrier_skew`（量化跨 DP Local Attention 时延差导致的共享 MoE AllToAllV 同步屏障等待损耗）、`analyze_moe_routing_imbalance`（量化专家热点与 EPLB 冗余专家平滑效应）以及 `analyze_chunked_prefill_decode_interference`（求解满足 TPOT SLA 的最大安全 Chunk 大小 $C_{\max}^*$）。
