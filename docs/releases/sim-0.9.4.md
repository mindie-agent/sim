# mindie-sim 0.9.4

本版本完成 **全模型家族（`GLM-5.3` / `GLM-5.3-Flash`、`DeepSeek-V4.1` / `V4-Flash` / `V3.2-Prune` / `V2-Lite`、`Kimi-K3`、`Qwen3.5` / `3.6` / `3.8` 及扩展预设 `LongCat` / `MiniMax-M3` / `Bailing-V3` / `Hunyuan-DiT`）配置字段全覆盖与全量 2,523 份 NPU Profiling 精度校准对齐**：

- **修复 4 键拓扑通信体制选择器在扩展 `transport` 字典下的失配问题（`mindie_sim/prediction/service_regimes.py`）**：
  - 当校准体制选择器声明标准 4 键拓扑签名 `{"dies_per_package", "links", "placements", "ranks_per_node"}` 时，在 `RegimePolicy.resolve` 中对实际 `transport` 轴执行 4 键投影比对，恢复 A3 `Kimi-K3` TP8/TP16 长上下文 Decode 精度（`decode:a3-kimi4-tp16-128k-decode-c12-p4k` 从 `-48.86%` 恢复至 `-2.63%`）。
- **修复 `alltoall` / `alltoallv` 启动开销在已标定 HCCL 体制上的双计问题（`mindie_sim/prediction/execution.py`）**：
  - 在 `Execution._context` 中仅对 `moe_dispatch_quantization == "routed_expert"` 采用 `min(routed.setup_ms, routed.path_latency_ms)`，在消除 A5 `EP=64` 单步启动开销 63 倍重复计数的同时，保持 `routed.setup_ms == 0` 标定体制的精确性（`decode:a3_glm53flash_heldout_c12_p4k` 从 `-15.54%` 恢复至 `-2.98%`）。
- **修复稀疏 MLA `LightningIndexer` 与分池索引（`index_pool`）的工作量守恒与长上下文复杂度（`mindie_sim/prediction/mla.py`, `index_pool.py`）**：
  - 将 Prefill 因果对数统计替换为 $O(1)$ 闭式计算 `_long_selected_pairs`，恢复 `index_scores` 的 Prefill HBM 读取账本并保护 `candidate_filter` 分支；恢复 `index_pool_topk` 的标定归并向量工作量并补齐空前缀尾块（`tail_rows`）边界处理。
- **补齐 HuggingFace 原生配置字段消费与 KV Cache 异构层精确统计（`mindie_sim/catalog/{hf_config,composable,config,cache,deepseek,glm,kimi,qwen}.py`, `prediction/feedforward.py`）**：
  - 支持 `layer_types` 异构混合注意力层逐层 KV Cache 容量精确求和（不再将线性注意力层误按全注意力计入显存）、`routed_scaling_factor` 路由缩放元数据记录，以及 `LongCat` / `MiniMax-M3` / `Bailing-V3` 零未识别/未消费字段告警。
