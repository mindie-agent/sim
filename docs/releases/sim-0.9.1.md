# mindie-sim 0.9.1

本版本新增 **HiSparse / Ascend Sparse KV Offloading 与 `memfabric_hybrid` (`1.3.0`) 物理机制建模与 A5 真机校准**、**Ascend A5 分层集合通信与计算-通信争用校准**、**Profiler 标定摄入与反事实配置顾问（`advise_case`）**，以及 **`xllm` 与 `mindie-sd` 自定义算子族 Lowering**。

- **HiSparse / Ascend Sparse KV Offloading 与 `memfabric_hybrid` 物理建模与 A5 校准（`mindie_sim/prediction/sparse_kv_offload.py`, `capacity.py`, `mla.py`, `disaggregated.py`, `ascend.json`）**：
  - **`memfabric_hybrid` (`acc_offload`) 三大 GVA DMA 内核建模与 A5 (`Ascend950DT_9582`) 校准**：精确建模 `memfabric_sparse_copy_d2h` / `memfabric_sparse_copy_h2d`（`entries < 64` 走 `OffloadMoeSparseCopyKernel`，`entries >= 64` 走 64-AIV `OffloadKVSparseCopyKernel`，每核 `88 KiB` 双缓冲 UB）、`memfabric_kv_exchange`（严格锁定末尾 2 个物理 AIV 核 `aiv_units-2..aiv_units-1` 运行于 `stream="offload"` 侧流）与 `memfabric_entry_gather`（64-AIV，`120 KiB` ping-pong UB），并基于 A5 实测更新 `Ascend950DT_9582` 的 H2D (`100.01 GB/s`, `alpha_us: 20.6`) 与 D2H (`100.15 GB/s`, `alpha_us: 20.8`) 参数。
  - **NPU HBM 与 MemFabric 共享 Host DRAM 双层容量账本**：在 `Memory` 中建模全序列常驻 NPU HBM 的 `index_k`、静态预留的 `topk_buffers`（对齐 `copy_sfa_pool_capacity` 的 2 幂次池容量）、`slot_map` 与 `lru_time_tables`，以及每个 DP 副本跨 TP 共享的 Host DRAM 池（含每层 `6 MiB` 虚拟内存对齐保留区与 `limiting_factor: "host_dram"` 准入拦截）。
  - **GLM-5.x / DeepSeek-V3.2 Full vs. Shared Indexer 层与三态 Decode 状态机**：区分短序列（`request_state=-3`，零 H2D 搬运）、首次长序列填充（`-1`）与长序列稳态 Decode（`-2`，仅拉取 `ceil(topk * miss_rate)` 未命中 Token）；Shared Indexer 层同时跳过 Indexer 打分与 LIM slot-map/LRU 更新，直接复用主层 `miss_plan`。
  - **P/D 分离 `SfaRemoteD2HConnector` 建模**：在 `Memory.pd_handoff_volume_and_stage_hbm` 中建模 `index_k` 按 TP 份数走 D2D HBM、主 `mla_kv` 仅传输 1 份（RD2H 写入 Decode 共享 Host DRAM 池）。
- **Ascend A5 分层集合通信与计算-通信争用校准（`mindie_sim/prediction/hccl.py`, `hccl_graph.py`, `ascend.json`）**：
  - 基于 A5 集群实测标定 `Ascend950DT_9582` 的 HCCS Scale-Up 与 RoCE Scale-Out 分层带宽、分段延迟底噪及计算-通信并发退化系数。
- **Profiler 标定摄入与反事实配置顾问（`mindie_sim/advise.py`）**：
  - 新增 `load_case_from_profile` 与 `advise_case`，支持从 `mindie-profiler` 归一化产物自动提取瓶颈签名并给出反事实并行/调度优化建议。
- **`xllm` 与 `mindie-sd` 自定义算子族 Lowering（`mindie_sim/prediction/family_xllm_ops.py`, `family_mindie_sd_ops.py`）**：
  - 新增 `xllm_custom_xllm_math`（`smla_va`、`sparse_flash_attention_lse`、`moe_grouped_matmul_swiglu_quant`、`mega_gdn_decode`、`onerec_final_beam_select`、`reshape_and_cache_a5`）与 `mindie_sd_custom_ops`（`rainfusion`、`ada_bsa`、`eagle_bsa`、`eagle_qbsa`、`adalayernorm`、`norm_rope_concat`、`laurel`、`quant_four_over_six_a5`、`eagle_ffn`）算子 Lowering。
