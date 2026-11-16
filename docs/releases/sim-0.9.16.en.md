# mindie-sim 0.9.16

[中文](sim-0.9.16.md)

This release systematically closes `mindie-agent/sim#36`, `mindie-agent/sim#37` (building on external contribution PR `#39`), and end-to-end `v0.9.14` physical modeling wiring across noncausal sliding-window / sparse `top-k` Attention truncation, layer-batched DCP & general MLA primitives, `Kimi K3.1` hybrid paired KV Cache / DCP Indexer / QSFA / external DSpark alignment with `vllm-ascend`, and `local_tp`, `shared_prefix`, `RecoverSSM`, `EPLB` expert slots, and exposed breakdown reporting.

- **Full-stack noncausal sliding-window & sparse `top-k` Attention truncation (`#36`)**:
  - Introduces unified `_cap_span` / `noncausal_attended_positions` semantics in `attention_common.py`, applying `min_positive(window, topk)` consistently across `mla.py`, `attention.py`, `context_parallel.py`, `sparse_attention.py`, `catalog/index_pool.py`, `prediction/index_pool.py`, and `piecewise` geometry so noncausal Prefill/Verify never inflates to full quadratic context spans.
  - Resolves phase-aware DCP degree (`active_dcp_degree(parallel, phase)`) in `context_parallel.py` so `pcp_dcp` uses `PCP` (`DCP=1`) in Prefill and `DCP` in Decode without cross-phase leakage.

- **Layer-batched DCP partial merge & general MLA lowering (`#36`)**:
  - Collapses per-request `dcp_partial_merge` collectives in `latent_dcp.py` into one batched `all_to_all` + `reduce` pair per layer (e.g., `1,280 -> 40` collectives on `B=32`, 40-layer Decode), eliminating artificial `startup_ms` inflation.
  - Adds `batch_general_mla_ops` in `generic_latent_attention.py` so `window_compressed` and multi-token Verify MLA paths emit layer-batched primitives with full activation and sparsity forwarding.

- **`Kimi K3.1` alignment with `vllm-ascend` paired KV Cache, DCP Indexer/QSFA & external DSpark (`#37` & PR `#39`)**:
  - Aligns paired hybrid page geometry (`paired_sfa.py`, `catalog/cache.py`, `cache_ownership.py`) with `vllm-ascend`, computing MLA KV capacity and row reads via `representation_store(model, runtime, "mla_kv")` and `cache_format` (`656 B` rows under `nz_split_fp8_bf16`) while sharding `index_store` by `owned_positions`.
  - Models `dcp_output_pack`, `dcp_output_alltoall`, and Indexer `all_gather(quantized_k)` / QSFA execution under DCP, while keeping linear KDA layers on `pcp` only when `parallel.pcp > 1`.
  - Ships external DSpark speculative draft modeling for `kimi-k3.1` (`catalog/models/kimi-k3.1.json` and `draft_context.py`).

- **End-to-end `local_tp`, `shared_prefix`, `RecoverSSM`, `EPLB` & `exposed_breakdown` wiring (`#34` / `#36`)**:
  - **`local_tp`**: Upgrades `local_projection_tp.py` to cohort-level lowering with `mla.o_a` / `mla.o_b` grouped projections, uniform-count ring/alltoall padding for ragged ranks under `communication="path"`, zero-work pass-through on empty groups, and shared block-aligned shard widths.
  - **`shared_prefix`**: Wires `prefix_reads.py`, `batch_prefix_attention.py`, `capacity.py`, and `efficiency.py` so shared prefix deduplication reduces both HBM Attention reads and KV capacity consistently.
  - **`RecoverSSM`**: Emits `recover_ssm` state commit as a deferred intent stripped under `assembly == "model"` (`body_forward`, `analyze_network_efficiency`, `solve_hardware_sla_throughput`) and bound after `accept`/`sample` before `token_commit` under `assembly == "service"`, eliminating stream-order cycles and dangling edges on multi-token verification.
  - **`EPLB` expert slots, cohort reuse & exposed breakdown**: Connects `expert_slot_work`, `cohort_reuse`, `collective_startup` / `collective_waves`, and 4-clock `exposed_breakdown` reporting across `predict`, `operators`, and `graph`.
