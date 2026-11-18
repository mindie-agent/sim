# mindie-sim 0.9.19: Issues `#49`–`#60`, `#62`, `#63` Closure, CANN 9.2.1 Dual-Silicon Calibration, and Cross-Repo SSOT Unification

[中文](sim-0.9.19.md)

`mindie-sim v0.9.19` closes all 14 issues (`#49`–`#60`, `#62`, `#63`) on the single causal execution engine, aligns CANN 9.2.1 production operator graphs and paired `Ascend950PR` + `Ascend950DT` silicon physics, and hardens 41-model fidelity and cross-repository SSOT contracts through Zero-Silent-Fallback enforcement and a deep risk-seam audit:

## 1. Issues `#49`–`#57` Architectural Closure

- **Engram host-only staging and two-stage streaming topology sweep**: Engram weights stay at the host-staging boundary, and topology sweeps stream in two stages without whole-network residency or duplicate construction.
- **Compressed shared-prefix indexer and analytical online UB TopK**: The shared-prefix indexer reuses compressed indices, and TopK plans online UB analytically while conserving the candidate plane and tiled reads.
- **mHC FP32 / ideal boundary fusion**: mHC fusion separates FP32 and ideal fusion boundaries while preserving Sinkhorn pre-gate coefficient causality, without borrowing private SRAM or an idle sibling core.
- **MC2 speculative capacity and pack-quantizer absorption**: MC2 speculative capacity follows real proposal rows; the pack quantizer is absorbed into routing packing instead of being priced twice.
- **Pure-SWA MQSMLA and 128-head TP support**: Pure sliding-window MQSMLA and 128-head TP sharding keep query/KV geometry consistent.
- **`gathered_dcp` latent/draft KV and `cp>1` index-key window**: `gathered_dcp` unifies the latent/draft KV gather boundary and fixes the `cp>1` index-key window.
- **`kda_recurrent` 64-head state collapse and `recover_ssm_commit` retention**: 64-head state collapses by real groups; speculative decode retains `recover_ssm_commit` after acceptance while ordinary probes emit no commit.
- **C8 sparse MLA batching and `GLM-5.3-Flash` `W8A8_MXFP8`**: Sparse MLA batches shared-prefix external reads, and `GLM-5.3-Flash` uses `W8A8_MXFP8` end to end.
- **Sustainable A5 DRAM MBU and `a5_pod` `per_port` concurrent collectives**: A5 sustainable DRAM bandwidth closes on real service modes; `a5_pod` and 16-rank wide nodes resolve `per_port` intra-node links and concurrent `alltoall` / `alltoallv` protocols.

## 2. Issues `#58`–`#60`, `#62`, `#63`: CANN 9.2.1 Operator Alignment and Paired `Ascend950PR` + `Ascend950DT` Silicon Calibration

- **Generalized CANN 9.2.1 operator-graph bindings (`#58`–`#60`, `#63`)**: Keyed strictly on structural layer geometry, tensor quantization formats, and hardware microarchitecture (zero `model_id` hardcoding):
  - `aclnnMlaPrologV3WeightNz` / `aclnnMlaPrologV4WeightNz`, `aclnnMixedQuantSparseFlashMla`, `aclnnFFNWorkerBatching` / `aclnnTokenDispatch`, and MXFP8 `engram_gate`;
  - 656B C8 packed MLA KV rows, BF16 `KeyPool` ring projection/residency, LTP8 `o_proj_local` AllReduce, and `MoeInitRoutingV3 + SwigluMxQuant`;
  - `QuantBatchMatmulV3` FP8, `LightningIndexer + QuantLightningIndexerGradKlLoss`, fused `KDA`, and `CausalConv1d UPDATE`.
- **Paired `Ascend950PR_958b (1.4 TB/s)` + `Ascend950DT (4.0 TB/s)` silicon calibration (`#62`)**: Adds the `Ascend950PR_958b` (`112 GiB`) hardware profile and models C8 SFA sparse gather, `aclnnMixedQuantSparseFlashMla`, `KeyPool` RowShard, `arch35` QLI V2, `CausalConv1d UPDATE`, L2-derived `MXFP4`/`MXFP8` GMM, `npu_swiglu_mx_quant`, narrow-`N` Split-K BF16 Router, `aclnnMoeGatingTopK`, and monotonic NZ Block-32 Prologue via multi-stage `PipelineProgram` / `PlannedRecipe` builders with zero `ResourceEnvelope` and 100% `plan.work` conservation.

## 3. Zero-Silent-Fallback, `mindie-profiler` SSOT Unification, and Deep Audit Hardening

- **Zero-Silent-Fallback and 41-model fidelity**: Matrix paths default to `SteadyMatmul`; supports `compressed-tensors` `W4A16`, cacheless bidirectional `gqa` FlashAttention, and exact `partial_rotary`/`swigluoai`/`noaux_tc`; every timing entrypoint raises `UnmodeledCapabilityError` with `mindie-unmodeled-structure-hint/1` on unmodeled structures.
- **Cross-repository SSOT**: `mindie_sim.observed_contracts` and `mindie_sim.subgraph_math` serve as the single authority for `mindie-sim` and `mindie-profiler v0.5.17+`.
- **Deep audit closure**: Aligns C11 `accel_*` and Python `Engine` mixed-placement coupled rate normalization, first-tile predecessor handling, and `cohort_size` propagation; eliminates 128-rank eager program rebinding and forced MC2 full-graph DES runs under `run_graph=False`.
