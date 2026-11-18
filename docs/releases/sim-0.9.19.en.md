# mindie-sim 0.9.19: Issues `#49`–`#57` Closure, Zero-Silent-Fallback 41-Model Fidelity, and `mindie-sim` / `mindie-profiler` SSOT Unification

[中文](sim-0.9.19.md)

`mindie-sim v0.9.19` closes issues `#49`–`#57` on the single causal execution engine and hardens 41-model fidelity and cross-repository SSOT contracts under the Zero-Silent-Fallback rule:

## 1. Issues `#49`–`#57` Closure

- **Engram host-only staging and two-stage streaming topology sweep**: Engram weights stay at the host-staging boundary, and topology sweeps stream in two stages without whole-network residency or duplicate construction.
- **Compressed shared-prefix indexer and analytical online UB TopK**: The shared-prefix indexer reuses compressed indices, and TopK plans online UB analytically while conserving the candidate plane and tiled reads.
- **mHC FP32 / ideal boundary fusion**: mHC fusion separates the FP32 and ideal fusion boundaries without borrowing private SRAM or an idle sibling core.
- **MC2 speculative capacity and pack-quantizer absorption**: MC2 speculative capacity follows real proposal rows; the pack quantizer is absorbed into routing packing instead of being priced twice.
- **Pure-SWA MQSMLA and 128-head TP support**: Pure sliding-window MQSMLA and 128-head TP sharding keep query/KV geometry consistent.
- **`gathered_dcp` latent/draft KV and `cp>1` index-key window**: `gathered_dcp` unifies the latent/draft KV gather boundary and fixes the `cp>1` index-key window.
- **`kda_recurrent` 64-head state collapse and `recover_ssm_commit` retention**: 64-head state collapses by real groups; speculative decode retains `recover_ssm_commit` after acceptance while ordinary probes emit no commit.
- **C8 sparse MLA batching and `GLM-5.3-Flash` `W8A8_MXFP8`**: Sparse MLA batches shared-prefix external reads, and `GLM-5.3-Flash` uses `W8A8_MXFP8` end to end.
- **Sustainable A5 DRAM MBU `83.16%`**: A5 sustainable DRAM bandwidth closes on real service modes and the MBU basis.
- **`a5_pod` `per_port` intra-node links and concurrent `alltoall`/`alltoallv`**: `a5_pod` uses `per_port` intra-node links and supports concurrent `alltoall` / `alltoallv` protocols.

## 2. Zero-Silent-Fallback and 41-Model Fidelity

- **`SteadyMatmul` default recipe**: The matrix path defaults to `SteadyMatmul`; no branch may switch on measurement presence or model name.
- **`compressed-tensors` `W4A16`**: `kimi-k2-thinking` / `kimi-k2.5` use INT4 group-quantized weights with BF16 compute, with explicit weight/activation formats.
- **Cacheless bidirectional `gqa` FlashAttention**: `hunyuan-video-dit` uses cacheless bidirectional `gqa` FlashAttention.
- **Exact `partial_rotary`, `swigluoai`, `noaux_tc`**: Rotary positions apply only to declared dimensions; `swigluoai` and `noaux_tc` (sigmoid + correction bias) expand from their real formulas.
- **Explicit `UnmodeledCapabilityError`**: Every timing entrypoint raises `UnmodeledCapabilityError` with a structured `mindie-unmodeled-structure-hint/1` payload for unmodeled structure instead of falling back silently.

## 3. `mindie-sim` / `mindie-profiler` SSOT Unification and Track 1/2 Parity

- **Shared observation contracts**: `mindie_sim.observed_contracts` unifies profiler and sim observation boundaries, units, sources, and implementations.
- **Closed-form subgraph math**: `mindie_sim.subgraph_math` expresses Cube FLOPs, Vector ops, and minimal boundary HBM bytes in closed form, reused by both Track 1/2.
- **Composition and SKU parity**: Track 1/2 share one causal ledger for composite parallelism, SKU efficiency, and SLA solving; speculative analysis geometry counts `q_len + draft_proposal_tokens` verification rows across a `dp>1` cohort while probes below the verification width keep ordinary target rows.
