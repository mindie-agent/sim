# Standard MXA8W4 grouped source program

The explicit `ascend_mxa8w4_nznk` recipe models the retained CANN
`MXA8W4_NZNK / VECTOR_ANTIQUANT / N_FIRST_TAIL_RESPLIT` branch. It replaces the
ordinary grouped matrix envelope, including its old BF16 compute declaration
and packed-weight traffic. It does not add a weight-supply duration to that
envelope, fuse a following activation, or reuse the Kimi SiTU implementation.

```json
{"runtime":{"kernels":{"families":{"grouped_gemm":{
  "kind":"ascend_mxa8w4_nznk","compact":true,"mte1_mode":"base"
}}}}}
```

Selection explicitly declares FP4 FRACTAL_NZ NK weights, FP8 E4M3 activations,
E8M0 group32 scales, BF16 output, no bias, and the retained controller branch.
The ordinary operator must carry integer `matrix_group_rows`. Probability
moments, an allocation capacity, unknown capture group lists, FP8 shared
weights, BF16 input, or a fused SiTU output do not meet this contract. The
initial supported tensor domain is K divisible by64, N divisible by32, with
real integer M tails and empty experts. Hardware must declare one AIC to two
AIVs and the source's independent copy/register resources and local capacities.

## Source mechanisms

The source revision is `a3925fd49cd256ad478d2faf983bf2961b8ae2c8` in
`cann/ops-transformer`; the retained verification JSON contains file hashes.
The capture's packaged binary has not been joined to that revision.

- The host splits N using declared group count, including zero groups. With32
  AICs, N4608 becomes16 tiles of128 and16 of160; N5120 becomes32 tiles of160.
  Output tiles roll across AIC owners over active experts and M tiles.
- BaseM is256. The controller chooses K refill512 only when M≤160 and N≤128;
  otherwise256. A's larger refill depth follows the source80KiB budget.
- Each AIC has two paired AIVs that read disjoint K halves of a packed bank,
  widen FP4 to FP8, and publish into L1. There are two weight slots; reuse waits
  for the actual dependent MTE1/Cube consumers. L0 operands have two slots.
- Weight scale input reads actual scale columns, while UB→L1 scale publication
  uses the source fixed128-column slot even for the final K chunk. The scale
  transpose gathers padded16×8 blocks of uint16 pairs. Activation scales use
  the AIC input pipe. Initial transpose-table DMA is256B on every AIV.
- The Cube precision is FP8. Mmad's16-row/32-byte issue granules contribute
  physical issued FLOPs and MTE1 work. Valid expert rows and logical FLOPs stay
  unchanged and are reported separately. No additional expert bank is read
  merely because M is smaller than16.
- Fixpipe converts FP32 L0C directly to BF16 GM. It has no invented AIV output
  epilogue or FP32 GM relay. The caller's Swiglu and quantization stay separate.

Source local reservations are241KiB UB/AIV,512KiB L1/AIC,64KiB each L0A/B and
256KiB L0C. The reservations are reused and released by the ordinary Engine.
One outer invocation owns all internal phases and any declared inclusive
kernel minimum; no internal phase acquires another launch floor.

## Register service units

`mxfp4_registers.py` uses the already retained950 Reg throughput and single/
dual-issue tables. It preserves calls, active typed lanes and UB payload.
There is **no blanket conversion from B8/B16 elements to FP32 operations**.

For each256 expanded weight values the source calls one `LoadAlign DIST_US_B8`,
two ShiftRight, one ShiftLeft, one Select, one And, and one StoreAlign. The
five bitwise operations have256-element/cycle capacity and can dual issue;
the store is single issue. The projection uses8 issue slots,6 data-register
writes,2 UB accesses and1 UB store for this loop, plus the real per-call masks
and constant registers. Generic `vector_ops` is not also charged. Issue slots
bind to the existing shared vector resource, so vector-capacity overrides
still affect execution.

UB Gather of B16 values has64 elements/cycle, unlike ordinary128-lane B16
loads. Its separate capacity demand uses the existing matching64-lane shuffle
resource, alongside single-issue Gather/Store calls. Register UB stores and
later MTE3 UB→L1 publication are different physical transfers; only the latter
counts on MTE3. Source/compiler instruction pairing and dependency latency are
not reconstructed by this capacity model.

`mte1_mode="base"` explicitly uses the declared shared MTE1 byte rate.
`mte1_mode="directional"` instead requires both `mte1_l0a` and `mte1_l0b`
service modes and binds each operand's physical bytes separately. Missing
directional rates remain a visible modeling error; the recipe does not infer
them or silently cancel the user's MTE1 override.

## Schedule limits and comparison boundary

The finite program overlaps a later refill with current Cube consumption.
Its current shared `RefillPipeline` serializes each complete producer panel,
and an output-tile drain precedes the next tile on that owner. Thus it does
not yet reproduce all AIV read/vector/write overlap between panels or overlap
across output tiles. These limitations are explicit in every result.

`compact=true` coalesces consecutive same-shape tiles on each physical owner.
It scales actual work by their integer repetition count and keeps the same
finite buffers. It is exact for isolated identical schedules and the symmetric
same-shape controls; heterogeneous owner contention is a fluid approximation.
Use `compact=false` for the detailed tile oracle. Neither form reconstructs
cache hits, scalar group-list cache latency or the compiled instruction stream.

The V4.1 DP8 capture records only rank0. Its routed matrices allocate48 rows
and48 banks; those capacities do not establish valid per-expert rows. Source
metadata establishes forced EPLB, but client concurrency does not establish
the other seven replicas' graph padding and routed work. The retained producer
therefore reports separate explicit symmetric-cohort and single-active-cohort
rank0 conditions. These are conditional predictions, not measured accuracy or
a replacement for whole-body comparison. Shared `QuantBatchMatmulV3` in this
capture uses FP8×FP8 ND weights and does not select this recipe.

See `data/verification/v41-mxa8w4-source-20261030.json` and its producer for
source/program/case hashes, real service work, resource release and the four
conditional workloads. No hardware parameter is chosen from a residual.
