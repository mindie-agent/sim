# A3 DispatchV2 / CombineV2 native caller boundary

`runtime.kernels.native_moe_protocol` now accepts this explicit contract:

```json
{
  "kind": "arch22_v2_caller",
  "dispatch_quant": "per_token_int8",
  "max_inflight_per_rank": 8
}
```

The peer window is a declared transport concurrency input. Eight is an example,
not an inferred A3 firmware limit. The selection does not identify the captured
FullMesh tiling key, and does not claim that an arch35 protocol runs on A3.
`global_bs` may optionally declare the common source allocation for unequal DP
senders. Zero/default follows the native uniform allocation `local_B × EP`;
an idle sender that still receives rows needs the explicit common allocation.

## Source and caller ownership

The retained `ops-transformer` revision is
`a3925fd49cd256ad478d2faf983bf2961b8ae2c8`. The relevant A3 entry points are
`mc2/moe_distribute_dispatch_v2/op_kernel/arch22/moe_distribute_dispatch_v2_a3.cpp`
and `mc2/moe_distribute_combine_v2/op_kernel/arch22/moe_distribute_combine_v2_a3.cpp`.
Their hashes and the shared quantization-header hash are retained in the planner.
The neighboring `*_a2.cpp` is not the A3 entry point.

The DispatchV2 entry exposes fullmesh, basic MTE and hierarchical variants.
All selected per-token INT8 variants own the following outputs inside the
native call: INT8 expanded rows, FP32 row scales, INT32 source-rank/token/slot
triples, INT64 expert counts and INT32 EP counts. `QuantInit` in
`moe_distribute_dispatch_v2_quant.h:68–111` allocates four bytes of dynamic scale
per row; `QuantDynamicPerToken` performs the float reduction and INT8 casts,
then stores the reciprocal dynamic scale. Neither an external BF16 pack nor
an external INT8 activation quantizer belongs between the caller and GMM.

CombineV2 consumes the expert output, DispatchV2 indices/counts, original
router weights and active mask, then returns the weighted BF16/FP16 token
rows. Its native call owns local return materialization and top-k weighted
reduction. A separate generic `route_unpack` is removed. The old node IDs are
kept as group entry/exit IDs so existing consumer dependencies remain valid.

The ordinary graph therefore contains exactly one native dispatch and one
native combine invocation per MoE layer. Expert-up consumes payload and row
scales from the same DispatchV2 completion. Independent expert-down
quantization remains. The caller's full-row router, right padding, TP-local
views, masked assignment counts, final AllGather and output trim remain outside
the native groups and are unchanged.

Allocation capacity is kept separately from valid expected/integer expert rows.
The source inference formula is `global_bs_real × min(E_local, topk)`, giving
128/384 rows for the GLM78 local-B1/B3 caller, while actual modeled receiver
work is six/24 rows. The exported seven-output ABI includes capacity-shaped
payload, dynamic scales, `128 × capacity` INT32 assist storage, expert/EP/TP
counts and the separate FP32 `expand_scales` buffer. Original GMM work and
source group counts are never enlarged to the allocation capacity.

The retained capture passes an empty DispatchV2 `expert_scales` input but still
returns allocated `expand_scales`; CombineV2 receives that buffer at input 10
and the actual FP32 router weights separately at input 4. Its buffer producer
is retained independently from `dynamic_scales`, which feed GMM. Allocation
alone does not prove initialized content or active read/write traffic, so this
optional buffer receives no invented per-row payload demand. Combine still
waits for the real router-weight producer. The current retained source includes
additional optional EP-count workspace paths; this caller's logical counts do
not assert those branch-specific extended workspaces were used by the capture.

## Analytic coverage and observation boundary

This initial adapter prices a logical transport and local-work lower bound:

- Dispatch reads source rows/IDs/mask, performs one per-source-row INT8
  quantization, sends logical INT8 rows plus four-byte scales and twelve-byte
  triples, and materializes received payload/scales/indices/counts.
- Combine reads expert return rows and indices, reverses the assignment matrix,
  reads router weights and received rows, and writes the weighted result.
- Source count exchange is an independent logical INT32 channel.
- The physical network continues to use the common Engine and explicit peer
  window, topology capacities, link latency, Tx/Rx contention and readiness.

Branch-specific 480/512 framing, replicated windows, scalar/poll/barrier work,
source core partition and overlap are **not reconstructed** without the native
tiling selection. These limits are exported as `program_execution` metadata.
Fullmesh may repeat quantization per expert assignment; this caller-level
lower bound does not infer that branch. No timings or instruction rates are
assigned from Body residuals.

An independently retained complete native-call observation can be declared as
one inclusive `service_regimes` response matching
`kernel: MoeDistributeDispatchV2` or `MoeDistributeCombineV2`. The response
covers the entire group including its network children, and is never repeated
per local phase. The dispatch entry `local_shape` is the real source X shape,
and the combine entry is its received expert-output shape;
received capacity/expected rows are separately exposed by
`native_abi_outputs`, `native_input_capacity_shape`, `activation_payload_shape`, `activation_scale_shape` and
`mc2_v2_caller.received_assignments`.

## Verification

The focused caller suite exercises the actual Engine, including unequal and
idle DP sources, logical wire conservation, received row-scale producers,
return-matrix transpose, TP right padding and trim, late inputs, and a 100us
synthetic complete-call response that occurs once per native group. The
existing arch35 per-token, caller-padding and native graph tests remain intact.
The real 78-layer GLM graph binds 75 dispatch/combine groups with no standalone
expert-up quantizers; the fused W4A8 gate/activation consumer directly waits for
DispatchV2 completion. This is a caller-contract result, not a new measured
accuracy result or evidence of the captured inner tiling choice.
