# MC2 caller token partition

`runtime.kernels.moe_token_partition = {"kind": "vllm_ascend_mc2_tp_slice"}` selects an explicit caller contract. The first domain is DP1, EP=TP>1, ETP1, CP=PCP1, full physical token rows divisible by TP, and no sequence-parallel or latent routed-output transform. Caller ownership is independent of the transport program: the existing analytical dispatch/combine remains available, or an explicit `native_moe_protocol` selects its source phases. Selecting this caller does not install arch35 V3 transport on a V2 capture, choose a quantization mode, or infer a peer window.

The ordinary MoE graph keeps the router projection and shared-expert matrices at full token count M. Zero-copy views then expose M/TP hidden and router-logit rows to selection. TopK, normalization, packing, the native MC2 sender and combine output use those local rows. The global forced-balance expert counts and received local expert rows remain unchanged, so matrix FLOPs, expert-bank bytes and activation geometry do not shrink a second time. MC2 source assignment partitions become one because the source tensor is already sliced.

After native combine, a real TP AllGather reconstructs M token rows. Its payload is `(TP-1) * (M/TP) * hidden * dtype_bytes` per rank, and its input contract is the local tensor rather than the complete output. Shared experts retain full-M computation and their own TP AllReduce. The final add waits for both paths. This is not the ALLGATHER MoE dispatch strategy or layer sequence parallelism.

The analytical transport control retains local source rows and exact integer peer bytes, waits for combine before gather, and leaves global expert work unchanged. It does not claim that its latency represents the V2 source protocol. This permits independently improving the caller and the transport without making simulation contingent on one selected native implementation.

`gather_output_materialization` defaults to `"view"`. The explicit `"per_rank_copy"` alternative emits TP individual TensorMove calls after the gather, each reading and writing one local-row slice, and a completion dependency over all copies. No batch threshold selects this option. It represents an observed caller materialization independently of the source gather operation. Existing `padded_local_tokens`/`valid_local_tokens` in the balanced ledger describe the replica before the caller slice; `moe_token_partition.source_local_rows` is the local source row count. No graph padding is invented.

## Source and capture boundary

Pinned vllm-ascend revision `8c30e4f491f73cba737a1dacc522894e437cdf3d`:

- `ops/fused_moe/prepare_finalize.py:291–306`: save original shape and slice hidden states and logits across physical TP.
- `ops/fused_moe/routed_experts.py:656–675`: `prepare` runs before expert selection; `713–717` calls finalize after the expert kernel.
- `ops/fused_moe/prepare_finalize.py:316–325` delegates non-SP finalize to the parent; `196–228` reconstructs TP output with `dist.all_gather` at line 222.
- `ops/fused_moe/fused_moe.py:199–237,269–281`: already reduced routed output joins separately reduced shared output without another final TP sum.

The retained A5 Qwen3.5-35B c4 CSV independently shows the boundary at rows 2459–2477: DispatchV3 consumes `[1,2048]`, grouped kernels handle `[32,2048]` and 64 local expert banks, CombineV3 returns `[1,2048]`, AllGather runs before the shared TP AllReduce, and ForeachAddList consumes two `[4,2048]` tensors. This sample is a small physical-row population; the principal q4/B4 population has full M16 and local M4. The pinned `force_eplb.py:130–142` uses source-rank-dependent cursor offsets and contiguous physical expert ownership. The simulator's retained balanced scenario uses strided ownership. Direct integer reconstruction for E256/EP4/topk8 and local M4/64/128 gives identical per-rank row classes, active-expert counts and uniform peer assignment counts: respectively 32 active experts with one row, 64 with eight rows, and 64 with sixteen rows. Token identities are not asserted equal, and this equality is not extrapolated to arbitrary routing geometries. No timing or capacity field determines active expert counts. Independent source semantics do not establish captured binary identity.

## Verification

Seven focused caller tests cover full-router/local-selection geometry at TP2/4, zero-active-expert tails, grouped work conservation, causal combine/gather/join ordering, explicit output-copy bytes, and ordinary HCCL source AllGather expansion. Together with native MC2, balanced experts, sequence-parallel frontend and MoE selection regressions: 49 passed.

`data/verification/mc2-token-partition-20261030.json` retains complete 40-layer structural checks for full M16/256/512. Each has 40 native dispatches, 40 native combines, 40 TP gathers, and 80 TP reductions (40 attention plus 40 shared). The explicit M512 copy arm has 160 TensorMove calls. The M16 public body execution completes with no live tasks/programs under 100 MiB in approximately five seconds. It deliberately uses an unquantized native transport control plus the existing separate activation quantizer; its duration is not a measured accuracy result. The separate per-token INT8 MC2 producer binding can be combined without changing this caller contract.
