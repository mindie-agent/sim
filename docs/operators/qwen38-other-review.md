# Qwen3.8 decode caller and residual review

This review reads the retained one-rank A5 TP1/DP1/EP1 capture. It changes no
runtime parameter, calibration role or execution model. The accompanying
`data/verification/qwen38-other-review-20261029.json` preserves the raw SHA-256,
ten exact sampler windows, per-window counts and activity sums, and source hashes.

## What the 4.488 ms historical `other` actually contains

The old reducer matches `ChunkGatedDeltaRuleFwd`, gating and convolution as GDN,
but misses `RecurrentGatedDeltaRule`. It therefore puts 48 recurrent calls into
`other` in every window. Moving only that explicit native family produces
**1.9492255 ms GDN and 3.975191 ms other**, taking each median from the corrected
per-window sums. This is not an addition of independently computed medians.
Original classifications and raw events remain preserved.

| Historical other leader | Calls/window | Median activity sum ms | Mean call us |
| --- | ---: | ---: | ---: |
| InplaceCopy / Slice | 289 | 0.715943 | 2.478087 |
| RecurrentGatedDeltaRule | 48 | 0.512314 | 10.660517 |
| Adds / Add | 189 | 0.381421 | 2.015616 |
| DynamicMxQuant | 80 | 0.334530 | 4.178982 |
| AddRmsNormDynamicMxQuant | 80 | 0.319443 | 3.985691 |

These are activity durations, not critical-path components or independent
invocation startup measurements. Their medians need not add to the family median.

## The next useful GDN boundary is the complete caller

The capture has, for each of 48 GDN layers, six InplaceCopy slices, one contiguous
materialization, one flat concatenation and one output zero. The corresponding
432 events have a **1.239715 ms median activity sum per window**, about 4.83% of
the observed 25.661551 ms median step span. That percentage is a prioritization
scale, not an additive attribution of wall time.

The descriptors identify concrete payloads: two `[4,96] -> [4,48]` gate slices,
two `[4,10240] -> [4,2048]` Q/K slices, a `[4,10240] -> [4,6144]` V slice,
a `[4,128,128] -> [4,48,128]` output copy, a contiguous
`[4,16384] -> [4,10240]` QKV view, a flat `8192+8192+24576` concatenation,
and `[4,48,128]` zeroing. These shapes and repeated counts support a caller
boundary; they do not establish each compiled copy's original Python statement.

The retained image source is
`data/observations/sources/gdn-image-c3d00a196905/gdn.py.txt`,
SHA-256 `e9ed210b4ca4431ee40916f06f0272e5e781eb04b61b084bacea216799fbca98`.
`AscendGatedDeltaNetAttention.forward` at lines 298-360 splits the projection,
materializes a/b and allocates a zeroed caller output. Its core at lines 532-575
rearranges QKV, performs two L2 norms and invokes recurrent GDN; lines 725-737
copy the result into the caller output. The independently retained FLA ctypes
reference additionally shows `npu_causal_conv1d_update` copying its result into
`out` at lines 2737-2739. The image inventory records binary/source limitations;
this is not proof that every reference helper executed in this capture.

The ordinary decode path in `prediction/recurrent.py` currently remains generic.
The `gdn_prefill` source selector applies only when a view is prefill. Several
contiguous copies are already represented; missing source caller operations
must not be appended a second time beside them.

A narrowly scoped decode caller adapter can reuse `TensorLowering`:

- Preserve projection views and use `contiguous` for actual consuming contracts.
- Represent independently materialized Q/K/V before the native recurrent ABI;
  use `concatenate_flat` only where the selected wrapper or descriptor proves it.
- Represent the caller's output zero as a real write, then `copy_into` with both
  producer readiness and destination identity. A contiguous source does not
  remove an explicitly requested copy.
- Group caller operations, normalization, gating, convolution and recurrence
  for descriptive complete-boundary comparison while retaining their separate
  resource work and native launch boundaries.

The observed 1.239715 ms must not become this adapter's latency. Payload,
per-owner mapping and real dependencies determine its prediction. The current
recurrent-core reference is already 0.463346 ms versus observed 0.512314 ms;
caller completeness has greater value than further polishing that core alone.

## Quantization, MTP and timeline limits

The two quantization families have 80 calls each, with `[4,17408]`/`[4,6144]`
dynamic MX quantization and `[4,5120]` fused residual norm. Their combined
per-window median is **0.652450 ms**. The earlier stable prediction reference
has the same 80+80 occurrences and logical shapes but 0.006760 ms isolated cost.
This supports refining per-owner row reduction, conversion and control work;
it does not support extra invocations or a shared timing floor. Native blocks
are 48 and 4 respectively. Reported scalar counters average about 1.85 and
1.63 us/call, but pipeline counters overlap and are not independent parameters.

The draft stream contains three matrix/SwiGLU bodies with physical M=4, whereas
attention queries are 4,1,1 and draft logits all use M=1. Valid tokens, padded
matrix rows and attention-query geometry therefore need separate contracts.
Do not infer more speculative proposals from the matrix descriptors.

All ten windows contain 1,817 events and no repeated complete-row identities.
No hcom, allreduce, allgather, alltoall, HCCL or AICPU-named event occurs in these
windows. This capture does not support attributing its MTP residual to
communication. It says nothing about the separate multi-rank Qwen captures.

Clipped all-stream unions are 22.719733-22.757330 ms; window spans are
25.090836-26.196194 ms. The uncovered intervals are 2.343852-3.449197 ms.
They are not additional device work, measured startup, or proven global idle.
Wait columns are never added to duration. `operator_details.csv` and
`api_statistic.csv` retain aggregate host durations without per-window enqueue
timestamps; they cannot establish input-ready or host-launch attribution here.
