# mindie-sim 0.9.17

[中文](sim-0.9.17.md)

This release aligns `Qwen3.8-Flash-Next` (`Qwen4Exp`) end-to-end with `vllm` and `vllm-ascend` ground-truth physics (low-rank gated `HyperConnection`, 16-prime `102.4 GB` `PLE` tables with dilated short convolution, weight-free `4:1` `QSA` sparse attention, and the 12-slot shared `Qwen4KVCacheLayout`), while fixing cross-catalog sparse-GQA decode selection / `index_k` cache accounting, `Gemma-4` `K=V` row widths, and built-in `kimi-k3.1` catalog registration.

- **Full-stack `Qwen3.8-Flash-Next` (`Qwen4Exp`) physical alignment**:
  - **Low-rank gated `HyperConnection` (`hc_lowrank=320`)**: Separates `GatedResidual` (`GroupedGemmaRMSNorm` + replicated BF16 `down_inject` / `up` skinny GEMMs + `hc_silu` + `hc_gate_mix` / `hc_combine` + final `hyper_connection_mixer`) from Sinkhorn `mHC`, eliminating 672 synthetic FP32 Sinkhorn launches and inflated FP32 HBM traffic while aligning norm weight ownership.
  - **16-prime `PLE` (`Per-Layer Embedding`) & dilated short convolution**: Maps 1-based `ple_layer_ids=[2]` to 0-based layer `1`, computes the exact 16-head prime vocabulary layout (`320,002,432 × 160` = `102.40 GB` BF16 sharded across `ETP/TP`, `12.80 GB/rank` at TP=8), supports `INT8_SYM` / `W8A8_MXFP8` group-32 table modes and `engram_cpu_offload` host UVA lookup, and models `ple_kv_proj`, `ple_gate`, `ple_short_conv` (`kernel=4, dilation=3`), and per-request `ple_conv_state`.
  - **Weight-free `4:1` `QSA` (`Qwen Sparse Attention`) & 12-slot `Qwen4KVCacheLayout`**: Separates `QSAIndexer` (`index_qk_proj`, `q_layernorm`, `k_layernorm`, `block_topk=512` compressed groups) from FP32 `KPool` rules, exposes the 12-slot QSA/GDN shared pool geometry (`qwen4_exp_pool`) in `request_cache_breakdown`, and honors `language_model_only` vision weight exclusion.

- **Cross-catalog alignment fixes (`MiniMax-M3`, `Step-5`, `Gemma-4`, `Kimi-K3.1`)**:
  - Excludes sparse/indexed `GQA` (`index_topk > 0`) from dense `StagedGQADecode` auto-installation in `body_defaults.py` and auto-seeds `analysis_input` on `scope="body"` target graphs in `graph_costs`, unblocking `predict(scope="body")`, `analyze_network_efficiency`, `solve_hardware_sla_throughput`, and `graph_costs` on `qwen3.8-flash-next`, `minimax-m3`, and `minimax-m3.1-flash`.
  - Fixes `catalog/cache.py` early return so all sparse `GQA` models (`qwen3.8-flash-next`, `minimax-m3`, `minimax-m3.1-flash`, `step-5-preview`) account for `index_k` cache bytes, fixes `Gemma-4` (`26B-A4B` / `31B`) `k_eq_v=True` global layers to charge `1×` KV cache capacity and traffic, and ships packaged `model_configs/kimi-k3.1.json` plus `Kimi-K3.1` and `Qwen3.8-Flash-Next-*` aliases.
