# Explicit grouped source execution

The public `quantized_grouped_source` call builder executes a declared source
schedule through the common matrix and causal Engine. It is a separate
candidate from `quantized_grouped`, which remains the analytical historical
callback binding. Neither builder reads observation durations.

Run the retained control:

```sh
python -m mindie_sim call \
  examples/calls/quantized-grouped-small.json \
  examples/calls/quantized-grouped-source-call.json
```

The declaration supplies the complete direct expert counts and physical
`base_m`, `base_n`, `base_k`, and `used_cores`. These are caller-selected
mapping inputs. The implementation does not recover a historical tiling key
or reproduce the host tiler's dispatch decision.

## Executed mechanisms

The source adapter retains original expert identities, including empty
experts. Owner rotation persists across nonempty groups, and the final-group
tail condition uses the original last group rather than the last active
expert. Source-derived rectangular tiles enter the existing
`HierarchicalMatmul` program; no second matrix pipeline or event engine was
introduced.

The common core refills K panels and transfers operands to matrix-local
buffers. A finite group boundary prevents a source owner's next output tile
from running before its current output is acknowledged. Other owners remain
independent. The generic matrix recipes retain their existing overlap.

The declared mixed output path now executes:

1. INT32 accumulator publication through the existing Fixpipe resource into
   vector-local storage. The split-M publication aligns its physical row
   extent to an even number.
2. Two vector partitions using their actual ceil-half/floor-half row counts.
   A zero-row partition consumes no scale, vector or output service and cannot
   lend its idle throughput to its sibling.
3. Per-partition row scales and duplicated column-scale reads for each active
   vector partition.
4. Up to four output chunks per partition, with two alternating output slots,
   vector-to-store dependencies and useful BF16 output extents.

Declared UB buffers remain resident for the owner's whole invocation.
Missing output/read resources are visible errors for this explicit path.
No accumulator spill to global memory, bandwidth multiplier, new resource
capacity or measured completion constant is added.

`PartitionedMatrixOutput` expresses a transfer role, physical partitions,
alignment, buffer geometry and finite output chunks. Its execution mechanism
is independent of model and vendor names; the source adapter chooses the
declared values. It uses existing physical resource and buffer identities.

## NZ access and logical work

The INT8 bank layout is `[E,N/32,K/16,16,32]`. A full-K rectangle of consecutive
whole N32 blocks is contiguous. A partial K panel consists of one physical
storage range per N32 block, separated by the full-K block stride. The
schedule's lazy `iter_weight_spans` exposes those ranges and preserves bank
offsets across empty experts.

These are storage addresses, not a claim about MatmulImpl DMA instructions,
cache misses or transaction startup. The current executable core charges the
whole-tile payload bytes and declared private copies. HBM reuse remains the
explicit ideal shared-cache assumption. No inferred NZ efficiency scales
those bytes.

Inner matrix dimensions use divisors of the declared source macro tile, at
most 128 per axis. This is the common core's declared micro mapping, not a
reconstruction of MatmulImpl's internal instruction dispatch. It also supports
macro dimensions such as 192 without silently substituting another outer tile.

## Independent acceptance

The DSH implementation was committed independently as `1f82da6f`.
Acceptance commit `49eaea96` corrected an actual source-reading error:
the pinned no-transpose scheduling split granule is **128**, while the NZ
storage block width is **32**. Those two values cannot be interchanged.
The same correction adds finite owner counts and partial-K NZ storage ranges.
The commits were integrated as `316856a8` and `71fc3ee6`.

An oracle compiled from the pinned source scheduling functions checked 410
geometries: 337 supported schedules match each owner's expert, offset and
extent; 73 rotated-final-split cases are rejected explicitly. A confirmed
ambiguous case is two one-row experts, N256, a 16x256 macro tile and eight
owners. The adapter does not silently repair the source's repeated rectangles.
The ten retained first-G-active historical geometries are supported.

Independent output checks cover odd rows, empty second vector partitions,
multiple K panels, multiple owners, delayed input readiness, resource traffic
and final buffer release. A second invocation cannot start using an owner's
reserved UB while the first still owns it. Public callback replay checks
preserve two complete source kernel boundaries and exact physical work.
Generic phase recipes cannot replace a source-required binding.

These checks establish executable geometry and causal/resource contracts.
They do not establish historical kernel identity or measured accuracy.

## Remaining difference

The trace investigation in `grouped-dispatch-evidence.md` makes control and
output behavior a high-value priority: small-M served kernels have high
reported scalar occupancy, and down projections have substantially more
Fixpipe time. Those counters overlap and cannot be converted into additive
overhead. The served channel scales are BF16, whereas the standalone
historical probe uses FP32; their group counts and timing boundaries also
differ.

The source loops over every original group, even when most are empty. The
ledger records the number of source group visits across AIC/AIV owners.
Scalar issue, cached GetValue response, native handshakes, compiled inner
loads, host allocation and graph-internal gaps remain unresolved. The
shared count-vector HBM fetch is still an explicit analytical assumption,
not a priced reconstruction of those scalar instructions.

The new source scenario therefore cannot be declared to have closed the
historical 63%-98% event difference. Its selected tiling and newly executed
dataflow are inspectable inputs for subsequent refinement. The retained
historical observations, their `unassigned` roles, and the old implementation
remain unchanged.
