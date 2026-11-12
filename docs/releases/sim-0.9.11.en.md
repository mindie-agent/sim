# mindie-sim 0.9.11

[中文](sim-0.9.11.md)

This release adds the normalized native-call structural binding and coverage contract, refines piecewise ETP collective reduction and monotone GQA decode occupancy, introduces canonical standalone primitive evaluation, packages offline catalog configurations for seven multinode fleet models, and caches static operator formula text at compile time:

- **Native Call Structural Binding & Coverage Contract (`mindie_sim/prediction/native_binding.py`, `mindie_sim/prediction/native_coverage.py`, `mindie_sim/structure.py`)**:
  - Introduces the `mindie-sim/native-call-binding/1` contract joining `OpSpec` / `OperatorPlan` boundaries to declared ACLNN, Triton, CANN-kernel, `torch`, or HCCL entries with typed ABI `signature_hash` digests, source `sha256` provenance, strict null-value validation, pure structural `graph_id` derivation, and D0 hybrid attention legality checks.
  - Adds emission-denominator coverage and profiler correspondence accounting (`native_call_coverage`, `coverage_from_records`, `join_profiler_rows`), separating `bound` (established/source-declared callable or kernel plus source `sha256` and typed ABI) from `fully_paired` (positive callable and positive resolution across all declared kernels).
  - Structure exports omit `native_calls` / `native_call_coverage` by default; pass `export_op_structure(..., native_evidence=True)` to attach them. The default report stays compatible with the released profiler 0.5.11 `graph_id` verification.
- **Piecewise ETP Reduction Boundary, Monotone GQA Decode Occupancy & Canonical `evaluate_primitive_spec` (`mindie_sim/prediction/piecewise/`)**:
  - Honors explicit reduction boundaries in `etp_allreduce` (`local_experts` vs. `tokens` / `moe_combine`) so pre-combine expert-TP reductions conserve routed expert token volume while post-combine reductions conserve `tokens * hidden`.
  - Unifies split-K wave occupancy for GQA / FlashAttention decode to remain monotone across `kv_len` and KV head-group sharding.
  - Adds canonical `evaluate_primitive_spec` routing standalone declarative primitive specs through the same guarded family compilers used by full subgraph compilation while conserving `operator_ms` across mechanism and core breakdowns.
  - Caches static operator formula text at compile time to avoid repeated formatting in hot loops.
- **Packaged Offline Fleet Model Catalog & Architecture Lowering (`mindie_sim/catalog/fleet.py`, `mindie_sim/catalog/configs/*.json`, `mindie_sim/prediction/`)**:
  - Packages offline HF/catalog JSON configs in both wheel and sdist builds with full alias resolution for 7 multinode fleet models: `minimax-m3`, `minimax-m3.1-flash`, `gemma-4-26b-a4b`, `gemma-4-31b`, `qwen3.6-35b-a3b`, `qwen3.8-flash-next`, and `step-5-preview`.
  - Wires end-to-end physical lowering for `GeGLU` gated activations, sparse `GQA` (`index_topk`) attention/indexer subgraphs, and Gemma `K=V` shared key-value projection and KV-cache geometry.

Actual test and benchmark measurements for this release are recorded in the attached `benchmark-results.json`, `benchmark-results.md`, and corresponding CI runs.
