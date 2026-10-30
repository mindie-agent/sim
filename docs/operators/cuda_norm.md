# CUDA normalization source model

This module reconstructs standalone `rms_norm` and `fused_add_rms_norm` from
`vllm-v0.25.1/csrc/libtorch_stable/layernorm_kernels.cu` (SHA256
`faaf442f9a520eb7d636acac06b7b52a1765da9b5db13d0d818ff2e1e2537a82`).
CUB reduction uses the supplied CCCL v3.0.0 source at
`e944297b0b51a8e9c0f1b30425370c31c1c0e0a0`. This is a declared implementation,
not the observed compiler/toolchain identity of a historical measurement.
All supporting header hashes are retained in operator metadata.

## Dispatch and supported inputs

`make_op` accepts `rows`, `width`, `dtype` (`fp32`, `fp16`, `bf16`), `kernel`,
`has_weight`, `aligned`, `batch_invariant`, and `input_stride`. Typed OpSpec
callers may provide rank 2, 3 or 4 shapes; rows are the product of leading
local dimensions. Missing dispatch metadata is reconstructed from the typed
shape/dtype/recipe. Provided dispatch metadata must agree with the source.
Higher-rank inputs assume contiguous leading dimensions; arbitrary higher-rank
strides and an implicit host `contiguous()` copy are outside this adapter.

Both kernels launch one CTA per row. The maximum block is 1024 threads for
fewer than 256 rows, and 256 threads otherwise. Standalone chooses
`V = gcd(16 / dtype_bytes, width)` and `block = min(width / V, maximum)`.
Fused chooses `block = min(width, maximum)`; its width-8 template requires
16-byte pointer alignment, width and input stride divisible by 8, and a
non-batch-invariant launch. Otherwise it uses the width-0 generic template.
The float `_typeConvert` specialization exists, so aligned fp32 also selects
the width-8 body. The declared CUDA branch assumes CUDA >= 12 and BF16 SM >= 80.

Standalone second-pass `vec_n_t` pointers require vector alignment. `aligned`
declares sufficient base alignment for this selected vector width. All visited
rows must retain that alignment. A misaligned second-pass address is rejected;
row0's unused stride does not invalidate a single-row case. Fused unaligned
inputs select the real generic branch. `input_stride` describes address gaps;
gap bytes are not charged as payload.

## Typed source work

`SERVICE_ROLES` declares the physical resource for every work category.
`Device.service_modes` supplies each category's explicit rate; the shared
engine maps these demands onto the physical resource, preserving contention.
No typed operation falls back to a generic vector rate. Rates may be analytical
or calibrated; running the simulation does not require measurements.

| Work category | Work unit | Physical resource |
| --- | --- | --- |
| `fp32_add`, `fp32_mul`, `fp32_div`, `fp32_rsqrt` | source scalar value operations | vector |
| `bf16_add`, `fp16_add` | source residual-add element lanes | vector |
| `cvt_bf16_fp32`, `cvt_fp32_bf16`, `cvt_fp16_fp32`, `cvt_fp32_fp16` | converted scalar lanes | vector |
| `cvt_int32_fp32` | integer width promoted to a float value | vector |
| `warp_shuffle_calls` | source PTX shuffle calls per warp | vector |
| `shared_read_bytes`, `shared_write_bytes` | source float-access payload bytes | shared_memory |
| `hbm` | source global element payload bytes | shared HBM |

For width W and scalar size S, standalone reads W*S in its first pass and
(1 + has_weight)*W*S in its second, then writes W*S. Fused first reads 2*W*S
and writes W*S residual, then reads (1 + has_weight)*W*S and writes W*S output.
Input is reread by standalone; residual is reread by fused. No cache credit or
transaction-padding model reduces these source accesses.

First-pass square and accumulation use fp32. Fused also adds the residual in
its scalar type. The width-8 specialization computes a fresh `sum_squares`
inside each vector and then performs the separate outer `variance +=`:
there are W/8 additional fp32 adds per row. Generic has no such extra sum.
Output converts source/optional weight to fp32, multiplies by the inverse RMS
and optional weight, and converts back. Thread0 promotes integer width to
fp32, then performs one division, one epsilon add and one rsqrt.

The counts describe source value operations, not compiler instruction issue.
Packed half/bfloat operations are counted in element lanes; generic torch
scalar-wrapper lowering remains unverified. Compiler FMA fusion, cast folding,
register residency, instruction latency and cache/transaction behavior are not
inferred. In particular warp calls/s is distinct from arithmetic lane ops/s.
Synthetic controls explicitly choose rates and are not hardware calibration.

## Causal program and ownership

One lazy stream per physical compute owner admits one CTA at a time. This is
an explicit single-resident-CTA schedule, not a claim of compiled SM occupancy.
Within a CTA, each source warp independently executes its actual strided loop:
read, dependent typed arithmetic/conversions, then write. The final warp and
final loop iteration retain their actual active element count. There is no
invented all-row load barrier. A warp can enter reduction as soon as its own
first-pass loop is complete.

Every active warp performs five source shuffle calls. At offset 1, 2, 4, 8, 16,
only lanes whose shuffled source lies inside the valid segment execute the
fp32 add. A full warp has 129 adds; a five-lane partial warp has eight; a
one-lane warp has zero. Shuffle issue remains five calls in every case.
Each warp's lane0 writes one aggregate. After the source block barrier,
thread0 reads and adds successor aggregates sequentially, with exactly
active_warps - 1 adds. There is no second warp-shuffle reduction. Thread0 then
computes and stores `s_variance`; the second barrier releases output loops.

Source shared traffic includes four bytes per warp aggregate store, four bytes
per successor aggregate read, four bytes for the variance store, and four bytes
per source output expression's variance access. Compiler hoisting/broadcast
and physical shared-bank transactions are unresolved, so these remain source
payload rates rather than claimed compiled load instruction counts.

All typed services are pinned to the CTA's physical owner. HBM stays unpinned
and shared. Idle SMs cannot lend vector or shared-memory service to one CTA.
Independent CTAs contend through the same engine. Rows and loop iterations are
created lazily, bounding live state by owners, active warps and stage count.

## Shared capacity and its evidence boundary

The source declares `s_variance` and `BlockReduce<float,1024>::TempStorage`.
The supplied CUB storage contains 32 float aggregate slots (128B), a float
prefix (4B), and a `NullType[32]` array whose full definition/padding is outside
the supplied closure. Together with `s_variance` (4B), 136B is a declared
source-payload lower bound, not exact compiled `sizeof` or occupancy evidence.

When geometry declares an `smem` or `shared` buffer (preferring `smem` if both
exist), the program acquires 136B on the
CTA owner's pool before its first load and holds it through its final output
store. The shared engine enforces capacity and releases the allocation. A
smaller declared pool is rejected. When no pool is declared, capacity is
explicitly unconstrained; shared read/write service is still charged. Exact
compiled storage can require more than this lower bound.

## Verification scope

The maintained tests independently interpret CUDA host dispatch, CUB lane
predicates and numerical kernel values, including the width-8 local sum plus
outer accumulation. The scalar dependency counterexample uses the same
12 global bytes and six arithmetic operations: at 1e6 B/s and 1e6 ops/s,
with other synthetic services made negligible, it requires 18 microseconds;
the former simultaneous read/compute/write implementation incorrectly took 15.

Other controls cover typed-rate sensitivity, missing rates, full/partial warp
progress, sequential thread0 aggregation, input readiness, real shared pool
lifetime/capacity/ownership, one CTA's fixed per-SM supply, shared HBM conservation,
rank-3 typed calls, valid source alignment, billion-row lazy construction and
all eight campaign geometries. The verification JSON records these bounded
source/resource checks. No device was executed and no hardware-accuracy
acceptance or historical toolchain identity is claimed.
