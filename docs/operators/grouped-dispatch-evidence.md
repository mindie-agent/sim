# Quantized grouped dispatch evidence

The retained records identify the historical API family and mixed-core launch,
but not the loaded kernel object or tiling. The pinned open-source
implementation provides a concrete, compatible mechanism to model explicitly.
It is not evidence that the historical CANN binary executed that implementation.

The machine-readable companion is
`data/verification/grouped-dispatch-evidence-20261029.json`. It contains
source hashes and line spans, a streaming scan of all eight retained profiler
CSVs, and the bounded read-only remote metadata inventory. No timing parameter
was fitted or changed. All original observations remain `unassigned`.

## What the historical artifacts establish

The standalone probe records `torch_npu=2.10.0.post7`,
`torch=2.10.0+cpu`, `/usr/local/Ascend/cann-9.2.0`, and
`Ascend950DT_9582`. Its caller has INT8 activation, INT8 FRACTAL_NZ weight,
FP32 channel and token scales, BF16 output, and a 256-element direct-count
list with the first G groups active. Eager, Graph1 and Graph8 remain separate
complete-event boundaries.

The service traces provide stronger kernel information, but a different
input contract. Streaming 1,262,293 rows from eight files found:

| Recorded family | Records | Inputs | Launch |
| --- | ---: | --- | --- |
| `aclnnGroupedMatmulWeightNz_GroupedMatmul_GroupedMatmul` | 50,080 | INT8/INT8, BF16 channel scale, FP32 token scale | MIX_AIC, 32 AIC and 64 mix blocks |
| `aclnnGroupedMatmulV5_GroupedMatmul_GroupedMatmul` | 3,756 | BF16/BF16 | AIC, 32 blocks |

These two families must remain separate. The service's retained
`w8a8_dynamic.py.patched` explicitly casts gate/up channel scales to the down
scale dtype before calling grouped matmul. The CSV confirms BF16 scales.
The standalone probe supplies FP32 scales. Actual service group-list values
are not retained, so its active-group histogram is also unknown.

The trace columns contain API/op names, shapes, formats, task/model ids,
launch sizes and pipe counters. They do not contain a binary hash, object
path, numeric tiling key or tiling blob. An op name is not a kernel build
identity.

On October 29, 2026, two bounded read-only inventories checked the retained
remote `/home/hp-qwen35-grouped/prof` directory. The directory had 146 files;
the detailed `p4096_c1_hold` and `p4096_c1_o32` inventories contained
framework records, parser logs and profiler metadata, but no original CANN
PROF directory, database, kernel binary or tiling blob. The metadata records
Level1 pipe-utilization profiling, `data_simplification=true`,
`record_op_args=false`, and `op_attr=false`. Version strings agree with the
probe. Other cells received a filename inventory, not a complete content
audit. No device workload was launched and no historical target file changed.

This closes the current identity search with an explicit unresolved result.
It does not authorize relabeling a compatible source branch as the loaded
historical binary.

## What the pipeline counters add

The following values are arithmetic means of the INT8 WeightNz records in
each retained hold capture. They are individual kernel spans, not complete
standalone events. Routed M and output orientation are known; the realized
expert counts are not.

| Capture | M | Projection | Records | Kernel us | AIC scalar ratio | Fixpipe us | AIV vector us |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| p4096_c1_hold | 32 | gate_up | 4,960 | 29.229 | 95.63% | 0.630 | 0.025 |
| p4096_c1_hold | 32 | down | 4,960 | 29.215 | 98.47% | 3.236 | 0.201 |
| p4096_c4_hold | 128 | gate_up | 6,440 | 37.545 | 93.11% | 1.709 | 0.094 |
| p4096_c4_hold | 128 | down | 6,440 | 39.396 | 98.68% | 9.093 | 0.750 |
| p16384_c1_hold | 32 | gate_up | 5,120 | 30.056 | 95.05% | 0.789 | 0.025 |
| p16384_c1_hold | 32 | down | 5,120 | 30.689 | 98.53% | 4.128 | 0.203 |
| p16384_c4_hold | 128 | gate_up | 6,200 | 34.713 | 94.00% | 1.337 | 0.094 |
| p16384_c4_hold | 128 | down | 6,200 | 36.066 | 98.61% | 7.184 | 0.748 |

This supports prioritizing owner scheduling, core-to-core handshakes and
output transfer over merely increasing dequantization arithmetic. It does
not establish a fixed 29-us control cost: pipe measurements overlap and do
not partition wall time. In particular, the reported scalar ratio cannot be
subtracted from the kernel span and reintroduced as an additive overhead.
The variation between captures is retained rather than normalized away.

The earlier Graph1/Graph8 analysis remains valid: event normalization differs
with replay length and group count. The service kernel records do not supply
the missing kernel-level timeline for those standalone replay events.

## Compatible source branch

The inspected source is
`/Users/maoxx241/code/refs/ops-transformer` at
`a3925fd49cd256ad478d2faf983bf2961b8ae2c8`, committed August 29, 2026.
Its configured origin is the CANN `ops-transformer` repository on GitCode.
The report records hashes for every cited file.

The relevant source chain is:

1. `op_api/grouped_matmul_950_checker.cpp:1071` accepts INT8 inputs with
   BF16 output, BF16 or FP32 channel scale, and FP32 per-token scale.
2. `op_kernel/grouped_matmul_utils.h:29` selects INT32 local accumulation
   for INT8 input. Noninteger scales enable the quantized mixed branch.
3. `op_kernel/grouped_matmul_apt.cpp:337` dispatches the non-transposed
   vector-dequant branch to `GQmmMixRegbaseKernel`.
4. `op_kernel/arch35/quant_adaptive_sliding_window_templates/gqmm_tiling_key.h:92`
   declares one AIC with two AIVs for that specialization.

The source host tiler chooses physical tiles and the vector-dequant branch.
It does not justify importing guessed tile values into a historical capture.
The explicit source adapter therefore receives a declared
`QuantGroupedTiling(base_m, base_n, base_k, used_cores)`. Its selected domain
and tests are documented in `quant-grouped-source-schedule.md`.

## Mechanisms suitable for the shared engine

**Group traversal and physical owner assignment.**
`gqmm_mix_online_dynamic.h:383` traverses all `groupNum` entries, obtains M/N/K
and advances group offsets before skipping empty groups. The ordinary dense
count format therefore still traverses trailing zeros. `quant_block_sch.h:239`
assigns group tiles from the owner after the preceding group's last owner,
with a sliding-window order inside each group. Empty experts have no matrix
work, but cannot simply disappear from the source control domain.

The final-group tail condition in `gqmm_mix_online_dynamic.h:369` tests
`groupIdx == groupNum - 1`, not the last nonempty group. First-G-active
histograms with G below 256 cannot claim a final-group tail redistribution
merely because the last active expert has finished.

**AIC/AIV dependencies.**
`gqmm_mix_online_dynamic.h:88` and `:435` show AIC work notifying two AIVs,
then waiting for their acknowledgements before subsequent work on that
owner. The state persists across group boundaries; `End` drains the final
acknowledgement. This is a local producer-consumer dependency. It is not a
global barrier after every expert, and the source does not provide an
elapsed-time constant for a handshake or an empty-group scan.

**INT32 accumulator transfer.**
The mixed kernel's output tensor is `VECIN/ND_ALIGN` in
`gqmm_mix_online_dynamic.h:59`. `MMCompute` transfers the completed
accumulator into local vector storage. The custom copy policy in
`mm_extension_interface/gqmm_copy_cube_out.h:57` aligns split-M transfers
to an even physical row count and uses Fixpipe for L0C-to-UB movement.
The vector halves subsequently operate on the true row counts.

For a one-row, 256-column INT32 tile, this policy has a two-row physical
extent, while the final BF16 result still has one row. This is an explicit
local transfer/residency difference, not evidence for writing a global
INT32 intermediate to HBM. Its timing must use a declared physical resource.

**Scale loads and output chunks.**
`gqmm_mix_online_dynamic.h:476` splits true rows into ceil(M/2) and floor(M/2)
halves. An empty half returns before loading scales. Each active AIV loads
its column scales, so a two-active-half tile duplicates those local loads.
Row scales cover each half's actual rows.

Each half produces up to four output chunks with two alternating output
buffers and explicit vector/MTE3 dependencies. BF16 output stores use their
true row and column extents. The source vector loop at `:837` also distinguishes
FP32 scales from BF16 scales: the latter require conversion and interleaving.
The caller's scale dtype must therefore survive source selection.

**NZ layout and transactions.**
`quant_block_sch.h:136` uses
`ceil(N/32) * ceil(K/16) * 512` bytes per non-transposed INT8 expert bank,
with NZ-specific column offsets. The historical K/N values are whole tiles,
so bank storage bytes equal logical bytes. That fact alone does not establish
the selected load order, request count, HBM cache misses or transfer latency.
Those require an explicit mapping and resource model; a layout label is not
a bandwidth multiplier.

## Scalar traversal: work is established, latency is not

`quant_utils.h:142` obtains one INT64 `GetValue(groupIdx)` for each
direct-count group. The mixed kernel calls this from `SetMNK` before the
empty-group check, with no AIC/AIV guard around that traversal. For an explicit
32-AIC/64-AIV launch and a 256-entry direct-count vector, the source describes
24,576 count reads, or 196,608 semantic bytes across those scalar streams.
The backing vector itself is only 2,048 bytes. These are source access counts,
not a count of physical DRAM transactions or an HBM traffic estimate.

The retained `GlobalTensor/GetValue` API mirror declares the S pipeline,
returns a scalar value, and requires Data Cache invalidation when another
agent may change the GM contents. The next group index is consecutive; its
address is not computed from the previous loaded count. The current count
does control offset updates, the empty-group branch and scheduling, but that
does not establish 256 serialized DRAM misses. Cache reuse, scalar issue
capacity, response latency and outstanding requests remain unspecified.
Different cores also traverse their own source streams; there is no new
global count-scan barrier.

`SetMNK` additionally references `kListGm_[0]` and `nListGm_[0]` for the
single weight-bank path. The names do not prove repeated GM loads:
`grouped_matmul_utils.h:141` selects either a const tiling object or a GM
pointer using `CONST_TILING`. The historical binary's compilation mode and
resulting instructions are unknown. Neither these repeated references nor
the INT64 counts should be converted into memory transactions by name alone.

The inspected `Ascend950DT_9582.ini` records 32 AICs, 64 AIVs, frequency and
bulk transfer rates, but no applicable `GetValue` issue or response latency.
The retained architecture document distinguishes per-core Scalar units from
SIMD-VF Aux Scalar and SIMT; its 128-byte SIMT cache granule does not establish
the GetValue transaction size. The retained profiler storage-access code has
910B-specific scalar counters, which are not an A5 latency primitive. The
historical pipe-utilization captures do not break scalar time into load
responses, cache misses, arithmetic, branches and synchronization.

Consequently scalar traversal/control is a real unpriced mechanism, and a
plausible part of the small-work gap, but its contribution has not been
identified. An independently supplied scalar issue/response and cache
scenario could be modeled explicitly later. The existing generic response
parameter alone does not identify those values. No 29-30-us residual, total
scalar ratio, or group-count slope is imported as a default timing constant.

## Independent shared-mechanism acceptance

These checks validate causal execution and resource accounting. They are
independent of the historical durations and do not identify historical
dispatch or establish measured accuracy.

The partitioned output review hand-enumerated 64 geometry cases: one or two
expert groups, one/two/three/five rows, eight or sixteen columns, K of four
or eight, and one or two AIC owners. Independent rectangle and microtile
arithmetic matched FLOPs, semantic HBM, AIC operand copies, AIV scale copies,
MTE1, Fixpipe and BF16 output service. Odd-M physical publication, one-row
use of only the first AIV, input readiness at 5 ms, and task/program/buffer
release were checked. A 23-test focused matrix run also passed.

The review exposed two defects that were corrected before acceptance:
missing scale-read/store resources could be skipped, and UB was retained
only during output instead of the full owner invocation. The corrected
model reports missing roles and reserves the owner's UB before its first
operand transfer. A separate 160-byte-per-owner control submitted two calls:
the first finished at 6.0625 ms, the second's first operand load began at
6.0625 ms, and the second finished at 12.125 ms. This verifies the admission
dependency rather than merely checking a final allocation count.

The explicit phase model received a separate 64-case review across rows
one/five, N three/seven, K one/five, groups one/three, owners one/three, and
reuse enabled/disabled. Hand-enumerated owner assignments matched each
owner's MTE2, MTE1, output stores and FLOPs; readiness and complete release
also held. Four ordinary frontend controls (grouped GEMM, GQA/norm,
dynamic INT8 and MLA decode) retained `PlannedRecipe` for every required
source program under explicit phase overrides. Real `operator_costs`
completed for all four, and the focused phase/convergence run passed 24
tests.

The phase review does not repeat or broaden the separate V4 four-DP smoke.
Its retained result remains in `v4-phase-runtime-20261029.json` and
`model-convergence-20261029-v4-phase-graph.json`. Serial phases preserve
declared per-owner work; they do not claim timing equivalence to the detailed
refill/buffer model.

The separate `source-grouped-scenarios-20261029.json` records 20 actual public
calls over ten retained conditions, using declared tiles 16/128/128 and 32
owners. For the first gate/up condition the source device graph predicts
2.123 us while the retained Graph1 event median is 40.576 us; the matching
down condition predicts 0.831 us against 39.828 us. At 256 active one-row
groups, the respective values are 34.644 versus 92.419 us and 24.449 versus
103.880 us. These are different boundaries with unresolved loaded dispatch:
all 400 samples remain unassigned and strict validation has zero pairs.
The new physical-output model therefore has not explained the event gap,
and those residuals do not identify a scalar-load or launch constant.

## Acceptance boundary

The immediate source adapter can preserve group ids, physical owner order,
explicit tiling, local accumulator transfer, scale dtype and AIC/AIV
dependencies on the existing engine. Independent controls should compare
tile ownership, local transfer bytes, padding, scale duplication and causal
completion against source arithmetic. Ambiguous source-tail domains remain
outside its declared support.

Still unidentified are the historical binary/tiling, standalone graph-internal
timeline, served count values, scalar/control issue rates, native Fixpipe
transaction cost and NZ cache residency. These missing facts limit claims
about historical accuracy; they do not block analytical simulation. No
observed duration, scalar ratio or replay residual becomes a timing parameter.
