# Resident QKV and state-output recurrence pipeline

The explicit `vllm_ascend_gdn_decode` binding accepts
`options.recurrent_implementation = "arch35_fp32"` to select the source recurrence.
Its default remains `"analytical"`, preserving the previous caller support on
A3 and other profiles. The selected arch35 buffers require 247,808 bytes per
AIV; A3's 196,608-byte UB is not enlarged or silently replaced with a different
source branch. Both paths expose `recurrence_selection` in operator metadata. This is an implementation selection by
software declaration, not a model-name or measured-duration branch.

The retained source is the local `refs/ops-transformer` mirror of
`attention/recurrent_gated_delta_rule/op_kernel/arch35/recurrent_gated_delta_rule_arch35.h`,
SHA-256 `5590f9a236f83660df7b216af2d535c4cb07a8ccd96fa07ebc6be3f78c9a03c2`.
The selected domain has BF16 inputs/output, FP32 state, scalar g, no vector gk,
integer value/key head ratio, DK/DV at most 512, and at most 256 value heads.
This adapter additionally limits a sequence to eight queries; that is its
selected implementation domain, not a claim that all native variants share it. Query/key
and value resident buffers are each 16 KiB; unsupported dimensions fail only
this explicit source binding. The generic mechanism remains available.

## Physical changes

Host tiling lines 327–329 select `min(AIV_count, batch * value_heads)` blocks.
Its SHA-256 is `785101bff3cc4454ffdf731661b940b69254850c5eb5fab7232bb118085cb9ed`.
Source lines 214–249 assign complete sequence/head work by accumulated query
count. The plan stores O(sequence count + owner count) segments, rather than
all token/head tasks. Each active owner runs a bounded lazy head program.

For each value head, Q/K/V for the entire query sequence are read once and held
in UB (lines 285–327, 405–434). GQA key heads are shared semantically, but the
source reloads Q/K for each executing value head; the plan retains those loads.
Acceptance changes the selected initial-state address. It cannot reduce
verification queries, arithmetic, output writes or speculative checkpoints.

State is processed in DV slices of `8192 / align16(DK)` elements, with the
source's 16-element alignment/cap rule (lines 149–153). Each slice loads its
initial state once. The update chain consists of gamma, decay, key dot,
delta and rank-one update. The state is copied into a single output slot and
its MTE3 write begins **before** the query dot (lines 363–399, 405–446).
The plan therefore overlaps state write with query computation. Reusing the
state output slot waits for its preceding DMA; reusing the attention output
slot waits for its preceding output. MTE3 submission preserves FIFO across
both output kinds and consecutive tokens.

The explicit source UB reservation is 247,808 bytes per active AIV
(lines 188–207). It includes two input-queue slots, one state-output slot, one
attention-output slot, resident state/Q/K/V/gamma and reduction temporaries.
It is released at owner completion. Logical head and tile counts do not create
extra physical cores or repeated kernel invocation floors.

The plan uses elementary vector operation work, including casts, reductions
and one element-operation per UB data copy. That is a resource abstraction,
not an instruction/latency reconstruction. Scalar `GetValue` payload remains
HBM work; it is not relabeled as MTE2 traffic. Scalar issue, metadata caching
and address/queue-flag instruction latency remain explicit unmodeled features.
No observed PMU time, scalar residual or per-model floor fills them.

The value-tile and head handoffs conservatively wait for the preceding bounded
program's outputs to retire. The source can overlap some final MTE3 work with
the next tile/head input. This cross-tile tail overlap is not reconstructed in
this revision; the implemented major overlap is state-write versus query dot.
Source declaration on another hardware profile is a conditional execution
model, not proof of that hardware's vendor dispatch or compiled instruction path.

## Repeated-head compaction

The explicit arch35 program now defaults to compact serial resource cycles.
`build(op, geometry, compact=False)` retains the detailed head-by-head oracle.
Only consecutive heads with the same query length on the same physical owner
are combined; unequal owner assignments and ragged sequence boundaries are
preserved. Every read, write, scalar payload and vector phase uses the shared
`Work.scaled` operation with the **actual number of repeated heads**. Distinct
head QKV/state addresses do not become one cached tensor. UB residency remains
247,808 bytes per active AIV, and the invocation boundary remains one kernel.

This is compaction of the existing conservative head/DV handoff, not a new
claim that the source drains all DMA at every head boundary. `ProcessHead`
(457–474) has no explicit equivalent of convolution's `DrainTaskMte3`.
The already documented cross-tile/head tail approximation remains. Within a
run, the complete serial cycle is time-dilated, preserving isolation timing;
external competitors can see differently phased resource demand.

Prepared operator output exposes this under
`metadata.program_execution.recurrent_head_schedule`: real head count,
resource-cycle count, per-owner counts, conservative handoff and
`serial_cycle_fluid_rephasing`. The shared `OperatorPlan` annotation defaults
to empty and is confined to the `program_execution` diagnostic subobject;
it cannot replace caller/source, layout or shape declarations. Existing
independent diagnostic keys are retained.

The retained producer and report are
`data/verification/gdn-recurrent-compaction-20261030.{producer.py,json}`.
They run both actual programs with identical hardware and invocation
parameters and do not consume observation durations.

| Geometry | Detailed tasks | Compact tasks | Detailed / compact prediction, µs |
| --- | ---: | ---: | ---: |
| Qwen3.6 c4 | 3,822 | 3,822 | 6.999 / 6.999 |
| Qwen3.6 c40 | 36,726 | 9,546 | 27.999 / 27.999 |
| Qwen3.6 c128 | 116,486 | 10,182 | 83.997 / 83.997 |
| Qwen3.8 c4 | 7,638 | 7,638 | 7.000 / 7.000 |
| Qwen3.8 c32 | 58,502 | 10,182 | 41.999 / 41.999 |
| Qwen3.8 c128 | 232,454 | 10,182 | 167.995 / 167.995 |

The largest isolated relative difference is 1.64e-11, from floating
arithmetic. Additional Qwen3.8 c128 controls with concurrent 100 MB HBM,
input-copy and output-copy traffic change duration by +0.324%, +0.275%
and +0.247% under compaction; a vector competitor produces the same
duration to numerical precision. Physical service is conserved, with at most
0.0025 byte roundoff across hundreds of megabytes. These are bounded controls,
not a universal contention-error guarantee. One local c128 evaluation fell
from 1.737 s to 0.075 s; local wall time is not a hardware timing parameter.

The new Engine controls cover q=1/2/3/4/5/8, DK/DV alignment tails, nonuniform
owner assignments, ragged contiguous runs, external HBM/MTE2/MTE3/vector
competition, late input and fixed-buffer release. The conditional comparison
below remains the separate observation evidence; compaction is not a new
calibration or whole-forward acceptance result.

## Conditional new-capture comparison

`data/verification/gdn-native-recurrent-20261030.json` is produced by its sibling
`.producer.py`. It streams each native kernel CSV, records its SHA-256, verifies
one tensor geometry per capture and compares mean duration. The four-query
sequence decomposition is the explicit capture scenario, consistent with the
query rows and batch metadata extents. There is a **known interface mismatch**:
source host line 284 takes B directly from the length-vector extent and line
371 expects B accepted-token values, whereas the new captures expose a B+1
prefix-vector extent and B accepted values. Qwen3.6 c4, for example, records
5 prefix entries, 16 state indices and 4 accepted entries. The producer reports
those raw extents and normalizes the declared four-query scenario to four
sequence lengths. It does not pass the raw prefix descriptor through this
retained host ABI or claim native-interface equivalence. The source header
supplies the normalized-length execution mechanism; actual metadata values and
binary interface transformations are not reconstructed from shapes.

Both programs use the same unchanged resource declarations from
`examples/cases/qwen38-a5.json`; the report pins that file and both program
hashes. No observation enters prediction. The data retain `split=unassigned`.

| Capture | Observed mean µs | Generic serial µs | Source pipeline µs | Signed difference |
| --- | ---: | ---: | ---: | ---: |
| Qwen3.6 c4 | 9.567 | 9.653 | 6.999 | −2.568 µs |
| Qwen3.6 c40 | 32.850 | 38.612 | 27.999 | −4.851 µs |
| Qwen3.6 c128 | 92.435 | 115.837 | 83.997 | −9.13% |
| Qwen3.8 c4 | 11.102 | 9.653 | 7.000 | −4.103 µs |
| Qwen3.8 c32 | 41.519 | 57.918 | 41.999 | +0.480 µs |
| Qwen3.8 c128 | 183.795 | 231.673 | 167.995 | −8.60% |

Small-call rows are within 15 µs absolute difference; the two calls above
50 µs are within 15% relative difference. This is a conditional comparison of
one native operator family, not heldout calibration, whole-forward accuracy,
TPOT, or the sum of overlapping profiler activity. Gating, convolution and
caller materialization residuals remain separate work.

Nine new core tests independently check tail geometry, per-head QKV loads,
FP32 state/checkpoint bytes, elementary vector work, owner assignment, both
copy-bandwidth axes and zero live allocations. A one-owner DK=DV=16 oracle has
1.920 ms of preceding vector service, overlapping 1.024 ms state DMA with
0.512 ms query/cast service, then 0.032 ms attention output: 2.976 ms total.
A second token cannot reuse the state or attention output slot prematurely.
The tests use synthetic resource rates to make those dependencies observable;
those rates are not deployed hardware calibration.
