# mindie-sim 0.9.4

本版本完成 **全模型家族（`GLM-5.3` / `GLM-5.3-Flash`、`DeepSeek-V4.1` / `V4-Flash` / `V3.2-Prune` / `V2-Lite`、`Kimi-K3`、`Qwen3.5` / `3.6` / `3.8` 及扩展预设 `LongCat` / `MiniMax-M3` / `Bailing-V3` / `Hunyuan-DiT`）配置字段全覆盖与分池索引（`index_pool`）工作量校准**：

- **修复分池索引（`mindie_sim/prediction/index_pool.py`）的归并向量工作量与空前缀尾块（`tail_rows`）边界处理**：
  - 恢复 `index_pool_topk` 的标定归并向量工作量，并在 `index_topk` 中显式计入未被池块整除的因果尾行（`causal_tail_rows`），消除空前缀短序列 Prefill 下的零工作量漏计。
- **补齐 HuggingFace 原生配置字段消费与 KV Cache 异构层精确统计（`mindie_sim/catalog/{hf_config,composable,config,cache,deepseek,glm,kimi,qwen}.py`, `prediction/feedforward.py`）**：
  - 支持 `layer_types` 异构混合注意力层逐层 KV Cache 容量精确求和（不再将线性注意力层误按全注意力计入显存）、`routed_scaling_factor` 路由缩放元数据记录，以及 `LongCat` / `MiniMax-M3` / `Bailing-V3` 零未识别/未消费字段告警。
