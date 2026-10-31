# mindie-sim 0.8.2

本版本面向 `400K`–`1M` 超长上下文序列、少专家合成/剪枝 MoE 结构泛化（`E ∈ {16..128}, K ∈ {2..8}`）以及 A2（`Ascend910B*`）全系列硬件画像提供物理底座与 $O(1)$ 周期流水折叠能力。

- **$O(1)$ 精确周期流水线折叠（`mindie_sim/prediction/refill_steady.py`）**：
  - 新增 `compact_periodic_pipeline`，在 `count > 32` 时提取独立单元引擎上的精确周期递推率与峰值缓冲池水位，将 `StagedGQADecode`（`attention_decode.py`，同构请求且单 KV 分片时，保留完整尾部因果掩码 Tile）、`StagedGQAPrefill`（`attention_prefill.py`，同构请求时）、`ascend_paged_mla_decode_arch35`（`paged_mla.py`）与 `ascend_paged_mla_decode_arch22`（`paged_mla_arch22.py`，单 Job/owner 时）的中间同构 Tile 折叠为耦合流体稳态近似段（并按物理子块 Placement 分组保留外部竞争下的单 AIV 隔离边界）。
  - 在无外部非平稳竞争时与未折叠 `PipelineProgram` 的 `duration_ms` 相对误差 `< 1e-11`、总服务工作量与峰值池分配误差为 `0`，同时将 `400K / 1M` 长上下文 Attention 仿真耗时从数分钟压降至 `< 0.05s`。
- **`400K / 1M` 超长上下文边界泛化**：
  - `mindie_sim/prediction/fai_decode_mapping.py`：支持 `cores <= 64`，并在 `len(requests) >= cores` 且 `q == 1` 时按算子源码 `SplitCoreDecodeBS1GN2` 执行 `kv_block_tile = min(max(1, 128 // group), kv_heads)` KV 头打包与尾部单头拆解重映射，消除高并发长上下文下的 `fai_host_single_token_specialized_decoder_not_selected` 异常；
  - `mindie_sim/prediction/index_pool_native.py`：当请求序列长度超出 `runtime.max_model_len` 且未显式固定 `block_table_columns` 时，自动按实际最大序列长度扩展页表列数。
- **少专家合成/剪枝 MoE 结构泛化（`E ∈ {16, 32, 48, 64, 96, 128}, K ∈ {2, 4, 6, 8}`）**：
  - `mindie_sim/model_catalog.py`（`derive_expert_pruned`）：保留前序 Dense 层（`plan.ffn == "dense"`），仅当 `(e // rg) * rgt < k` 或分组不整除时才重置 `router_groups` / `router_group_topk`，在路由分布重置为 `()` 时同步将 `expert_routing_source` 重置为 `"uniform"`，并同步更新 `auxiliary.mtp` / `auxiliary.dspark` 中的路由专家配置；
  - `mindie_sim/prediction/moe_init_mxfp8.py`：对齐 `moe_init_routing_v3_tiling_arch35.cpp` 的 `useGatherCopy = (topk > 3)` 分支语义，在 `topk <= 3` 时按 `ScatterCopyExpandedXandMXQuant` 逐条排序路由项独立搬运与量化（不做跨 `topk` 输入行去重）；
  - `mindie_sim/prediction/grouped_dequant_arch35.py`：在 `E <= 64`（每个 AIV 仅负责单专家组）时允许跳过空专家组，而在 `E > 64` 且空专家组后同一 AIV 仍有后续非空专家组时保留 `arch35_dequant_zero_group_stalls_later_source_owner` 报错。
- **A2（`Ascend910B*`）DaVinci `arch2201` 硬件画像与多头准入补齐**：
  - `mindie_sim/backends/profiles/ascend.json`：按各 SKU 的 CU 数量与主频补齐 `Ascend910B1/B2/B2C/B3/B4/B4-1` 的 `mte1`、`mte2_aic`、`mte2_aiv`、`mte3`、`fixpipe` 速率、`resource_units`、`service_modes`、`kernel_invocations`（标注 `arch2201_baseline_extrapolation` 溯源）及 `fp32/hf32/int4` 峰值，并修正 `source.pipeline.fields.hbm.rate_bytes_per_s` 与 `A2` 架构溯源声明；
  - `mindie_sim/prediction/body_defaults.py`、`indexed_attention.py`、`indexed_quant_attention.py`：将 Indexed MLA 单卡本地 Query Head 准入上限扩展至 `64`（packed `int8`）与 `128`（`bf16`），并在 `heads > 16` 时按源码 `vecDealM` 双 AIV 拆分（`AIV0`/`AIV1`）定价 `softmax`/`probability`/`nupdate`/`update`/`output` 阶段。
