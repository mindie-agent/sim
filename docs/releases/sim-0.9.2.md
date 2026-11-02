# mindie-sim 0.9.2

本版本新增 **可配置网络带宽/时延/利用率与多级拓扑建模**、**A5 EP MC2 与 DeepEP 两阶段集合通信与计算掩盖建模**、**Attention-FFN 分离（AFD）与多步微批次流水线 Pareto 求解**、**MoE 专家与 KV Cache 分层 DDR/SSD 卸载建模**，以及 **公开硬件 SKU 目录扩展与私有/保密自定义硬件 SKU 模板引擎**。

- **可配置网络带宽、时延、利用率与多级拓扑建模（`mindie_sim/schema/hardware.py`, `prediction/topology_tiers.py`, `compose.py`, `backends/ascend.py`, `backends/nvidia.py`）**：
  - 在 `Link` 与 `Hardware` 中新增一等公民参数 `bw_utilization`、`switch_hops`、`switch_hop_latency_us`，以及节点内子拓扑（`4P`/`8P`）与跨超节点（`d2d_inter_pod`）链路层。
  - 新增 `EP_TIER_PRESETS`（`a2_standard`、`a3_superpod`、`a5_pcie_16p`、`a5_pod`/`a5_ub_pod`）与 `compute_ep_effective_bandwidth`，支持按拓扑层级自动折算有效带宽与多跳交换时延。
- **A5 EP MC2 与 DeepEP 两阶段通信及共享专家重叠建模（`mindie_sim/prediction/deepep.py`, `moe_transport.py`, `topology_tiers.py`）**：
  - `analyze_ep_mc2_communication` 融合 MXFP8 块缩放开销（$33/32\text{ B}$）、投机解码有效接受步长 $(1 - \alpha^{K+1})/(1 - \alpha)$、EPLB 峰均负载不均衡系数以及按交换跳数分解的启动时延。
  - `analyze_deepep_collectives` 与 `analyze_shared_expert_overlap` 支持标准 AllToAllV 与 DeepEP 两阶段（跨节点去重 `deepep_group` + 节点内扇出 `deepep_intra`）通信量对比、`deepep_overlap_ratio` 掩盖分析以及共享专家/Attention 计算重叠评估。
- **Attention-FFN 分离（AFD）与多步微批次流水线 Pareto 求解（`mindie_sim/prediction/afd.py`, `sweep.py`）**：
  - 新增 `effective_pool_link`、`predict_afd` 与 `solve_afd_pareto`，支持异构 Attention/FFN 池跨池短板链路折算、$M$ 步微批次（$m=1, 2, \ge 3$）双向流水线周期建模，以及在 TPOT SLA 约束下的最优池配比 Pareto 前沿求解。
- **MoE 专家与 KV Cache 分层 DDR/SSD 卸载建模（`mindie_sim/prediction/offload.py`, `capacity.py`, `execution.py`, `feedforward.py`）**：
  - 新增 `OffloadPolicy`、`prepare_expert_offload` 与 `analyze_kv_cache_offload`，建模冷专家按需 DDR/SSD 换入、KV Cache 四种预取掩盖策略（`v1..v4`）、MTP 多 Token 命中率折算，以及单层 Staging Buffer HBM 峰值约束。
- **公开硬件 SKU 扩展与保密自定义硬件 SKU 模板（`mindie_sim/backends/nvidia.py`, `custom_hardware.py`）**：
  - 扩展公开可验证的 NVIDIA SKU（`NVIDIA_H20`、`NVIDIA_H800`、`NVIDIA_H200`、`NVIDIA_B300`），严格保持零未公开代号。
  - 新增 `custom_hardware.py`（5 类结构原型模板 `HARDWARE_ARCHETYPE_TEMPLATES`、运行时 SKU 注册表及 `MINDIE_SIM_CUSTOM_SKUS` 外部文件加载器），支持零源码侵入接入私有或未公开硬件规格。
