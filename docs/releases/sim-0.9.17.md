# mindie-sim 0.9.17

[English](sim-0.9.17.en.md)

本次发布系统性对齐 `Qwen3.8-Flash-Next`（`Qwen4Exp`）与 `vllm` / `vllm-ascend` 的真实架构物理（低秩门控 `HyperConnection`、16 质数分表 `102.4 GB` `PLE` 与膨胀短卷积、无权重 `4:1` `QSA` 稀疏注意力、12 槽共享 `Qwen4KVCacheLayout`），并同步修复跨模型目录的稀疏 `GQA` Decode 调度/`index_k` 容量、`Gemma-4` `K=V` 缓存行宽及内置 `kimi-k3.1` 目录注册。

- **`Qwen3.8-Flash-Next`（`Qwen4Exp`）全栈物理对齐**：
  - **低秩门控 `HyperConnection`（`hc_lowrank=320`）**：区分 `GatedResidual`（`GroupedGemmaRMSNorm` + BF16 `down_inject` / `up` 低秩投影 + `hc_silu` + `hc_gate_mix` / `hc_combine` + 末尾 `hyper_connection_mixer`）与 Sinkhorn `mHC`，消除 672 次冗余 FP32 Sinkhorn 启动与虚假 FP32 访存，并对齐层内归一化权重归属。
  - **16 质数分表 `PLE`（`Per-Layer Embedding`）与膨胀短卷积**：修复 1-based `ple_layer_ids=[2]`（对应 0-based 第 `1` 层）及 `16` 个质数分表总长（`320,002,432 × 160`，BF16 共 `102.40 GB`，按 `ETP/TP` 切分，TP=8 单卡 `12.80 GB`），支持 `INT8_SYM` / `W8A8_MXFP8` 表量化与 `engram_cpu_offload` 主机 UVA 卸载，并完整建模 `ple_kv_proj`、`ple_gate`、`ple_short_conv`（`kernel=4, dilation=3`）与每请求私有卷积状态（`ple_conv_state`）。
  - **`QSA`（`Qwen Sparse Attention`）`4:1` 压缩索引与 `Qwen4KVCacheLayout` 12 槽共享池**：对齐无权重 `QSAIndexer`（`index_qk_proj`、`q_layernorm`、`k_layernorm`，不再误套用 FP32 `KPool` 拷贝规则）及 `block_topk = 512` 压缩组选择，并在 `request_cache_breakdown` 中核算 12 个 QSA 物理槽位与 3 组共 36 层 GDN 状态的共享块布局（`qwen4_exp_pool`）及 `language_model_only` 纯文本权重排除。

- **跨模型目录一致性修复（`MiniMax-M3`、`Step-5`、`Gemma-4`、`Kimi-K3.1`）**：
  - 修复 `body_defaults.py` 对稀疏 `GQA`（`index_topk > 0`）误装稠密 `StagedGQADecode` 的问题，打通 `qwen3.8-flash-next`、`minimax-m3` 与 `minimax-m3.1-flash` 的 `predict(scope="body")`、`analyze_network_efficiency`、`solve_hardware_sla_throughput` 与 `graph_costs(target_graphs(..., scope="body"))`。
  - 修复 `catalog/cache.py` 中 `GQA` 提前返回导致的 `index_k` 容量漏计（覆盖 `qwen3.8-flash-next`、`minimax-m3`、`minimax-m3.1-flash`、`step-5-preview`），对齐 `Gemma-4`（`26B-A4B` / `31B`）`k_eq_v=True` 全局层的单倍 KV 行宽与访存，并内置打包 `model_configs/kimi-k3.1.json` 及 `Kimi-K3.1` / `Qwen3.8-Flash-Next-*` 目录别名。
