# mindie-sim 0.9.16

[English](sim-0.9.16.en.md)

本版本系统性闭环 `mindie-agent/sim#36`、`mindie-agent/sim#37`（含外部贡献 PR `#39`）以及 `v0.9.14` 全栈物理建模接线，对齐非因果滑窗/稀疏 `top-k` Attention 截断语义、层粒度批量 DCP 与通用 MLA 原语、`Kimi K3.1` 与 `vllm-ascend` 的混合双格式 KV Cache / DCP Indexer / QSFA / 外置 DSpark 建模，并打通 `local_tp`、`shared_prefix`、`RecoverSSM`、`EPLB` 专家槽位与暴露耗时分解的端到端契约。

- **全栈非因果 Attention 滑窗 / 稀疏 `top-k` 截断一致性（Issue `#36`）**：
  - 在 `attention_common.py` 引入统一的 `_cap_span` / `noncausal_attended_positions`，对 `causal=False` 的滑窗与稀疏 `top-k` 注意力统一施加 `min_positive(window, topk)` 上界，同步消除 `mla.py`、`attention.py`、`context_parallel.py`、`sparse_attention.py`、`catalog/index_pool.py`、`prediction/index_pool.py` 与 `piecewise` 几何求解器在长上下文非因果 Prefill/Verify 下退化为全序列平方复杂度的偏差。
  - 修复 `context_parallel.py` 在 `pcp_dcp` 模式下按运行阶段解析有效 DCP 度（`active_dcp_degree(parallel, phase)`），确保 Prefill（`PCP` 生效、`DCP=1`）与 Decode（`DCP` 生效）物理分片与通信拓扑严格一致。

- **层粒度批量 DCP 归约与通用 MLA 算子降维（Issue `#36`）**：
  - `latent_dcp.py` 将逐请求循环拆分的 `dcp_partial_merge` 收敛为每层单次批量 `all_to_all` + `reduce` 原语（例如 `B=32, 40` 层 Decode 从 `1,280` 个细碎集合通信收敛至 `40` 个），消除启动开销（`startup_ms`）虚高。
  - `generic_latent_attention.py` 新增 `batch_general_mla_ops`，将 `window_compressed` / Verify 等通用压缩 MLA 路径按层批量发射，并与 `mla.py` 共享完整激活与稀疏参数契约。

- **`Kimi K3.1` 与 `vllm-ascend` 混合双格式 KV Cache、DCP Indexer / QSFA 及外置 DSpark 对齐（Issue `#37` & PR `#39`）**：
  - 对齐 `vllm-ascend` `KVCacheSpec` 配对混合分页几何（`paired_sfa.py` / `catalog/cache.py` / `cache_ownership.py`），统一使用 `representation_store(model, runtime, "mla_kv")` 与 `cache_format`（如 `nz_split_fp8_bf16` 每行 `656 B`）核算 MLA KV 读写与容量，并将 `index_store` 按 `owned_positions` 分片计入 DCP rank 本地缓存。
  - 补齐 DCP 模式下的 `dcp_output_pack`、`dcp_output_alltoall` 及 Indexer `all_gather(quantized_k)` / QSFA 执行路径；线性注意力 KDA 层仅在 `parallel.pcp > 1` 时启用 `pcp` 计划，避免被 MLA 的 `dcp` 误伤。
  - 支持 `kimi-k3.1` 外置 DSpark 投机解码草案网络（`catalog/models/kimi-k3.1.json` 及 `draft_context.py` 草案上下文贯通）。

- **`local_tp`、`shared_prefix`、`RecoverSSM`、`EPLB` 与暴露耗时分解端到端接线（Issues `#34` / `#36`）**：
  - **`local_tp`**：`local_projection_tp.py` 升级为 cohort 级 lowering，支持 `mla.o_a` / `mla.o_b` 分组投影、不均匀 token 计数填充对齐（兼容 `communication="path"` 拓扑求解）与空组零工作直通，并与容量账本共用块对齐分片宽度公式。
  - **`shared_prefix`**：打通 `prefix_reads.py`、`batch_prefix_attention.py`、`capacity.py` 与 `efficiency.py`，使共享前缀去重同时体现在 Attention 访存和 KV 容量上。
  - **`RecoverSSM`**：将 `recover_ssm` 状态提交建模为延迟绑定意图，在 `assembly == "model"`（`body_forward`、`analyze_network_efficiency`、`solve_hardware_sla_throughput`）下自动剥离服务级 `accept` 依赖，在 `assembly == "service"` 下重排至 `accept`/`sample` 之后、`token_commit` 之前，彻底消除多 token 验证下的流序环与悬空边报错。
  - **`EPLB` 专家槽位、Cohort 复用与暴露耗时分解**：贯通 `expert_slot_work`、`cohort_reuse`、`collective_startup` / `collective_waves` 与四维时钟 `exposed_breakdown`（`predict` / `operators` / `graph` 输出结构）。
