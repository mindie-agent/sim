# Kimi BF16 native MC2: physical phases and remaining control gap

This repairs the explicitly selected `arch35_fullmesh` caller. Formal window8
transport cases remain unchanged; two separate fullmesh candidates live under
`examples/cases/high-concurrency-20261030/kimi-native-mc2/`. Retained source
revision `a3925fd49cd256ad478d2faf983bf2961b8ae2c8` is not identified with the
captured binary, and the DispatchV3 fullmesh tiling key remains unavailable.

Compact profiler tensor contracts establish BF16 dispatch input rows 1/4 for
c1/c32, width 3584, top-k 8, 16 experts, TP8/EP8. Output capacities 16/64 are
not valid assignment counts. The existing forced-balance scenario remains an
explicit assumption; route values and active peer histograms are unavailable.

## Source corrections

The retained `moe_distribute_dispatch_v2_a5_full_mesh.h` provides:

- Lines 1748–1801: each AIV initializes and reads its own routing-ID array.
  Count and payload work then proceed on separate owner domains.
- Lines 972–1019: successful count-status read, reduction/clear, per-count-owner
  sum and a 32-byte sum record sent to every count owner.
- Lines 1028–1162: sum exchange, prefix predicates, local count output plus one
  complete offset workspace per AIV, and ready flags published to every AIV.
- Lines 1221–1280: every AIV checks and clears ready flags; each nonempty
  source/expert interval reads all prefix offsets; AIV0 writes INT64 counts.
- Lines 1348–1354: local materialization reads before writing payload/triples.

For C count owners, A AIVs and E experts, local count-workspace writes include
`C*C*32` sum bytes, `(A+1)*E*4` prefix bytes and `C*A*32` ready bytes. These are
local HBM/MTE services, not extra network messages. Zero-count experts and
incoming-only ranks participate. The second masked status `Duplicate` writes
one flag element per record; it no longer counts eight elements per flag.

`moe_distribute_combine_v2_a5_mte.h` lines 1512–1533 partition reception by
**output tokens**, not top-k assignments. A one-token return has one receiving
AIV even when reading eight expert slots. Lines 1330–1352, 1498–1502 and
1597–1601 include BF16/FP16-to-FP32 casts, FP32 multiply/add, accumulator
initialization and one final output cast. This replaces generic unpermutation
work while retaining router-weight dependencies, one payload read and one
final output store. Dispatch and return each retain one outer invocation.

The same Engine executes these ordered phases. Whole-phase service envelopes
remain an approximation: source token-level UB windows/inter-token overlap,
scalar prefix additions, control instruction latency, extra polling and actual
receiver ownership are unresolved. No complete native UB-capacity validation
or compiled instruction timing is claimed.

## Conditional comparison

`data/verification/kimi-mc2-bf16-20261030.json` records runtime, input and compact
observation hashes. Previous native predictions are pinned to `ffa61916`.
These are isolated first-KDA-MoE native groups, paired to retain inverse traffic;
expert GEMMs are excluded. Their spans are not additive body corrections.

| Capture | Operation | Observed mean µs | Prior native µs | Updated native µs | Absolute gap µs |
|---|---|---:|---:|---:|---:|
| c1 | DispatchV3 | 31.335 | 2.314 | 2.315 | 29.019 |
| c1 | CombineV3 | 14.544 | 2.191 | 3.989 | 10.555 |
| c32 | DispatchV3 | 34.197 | 2.726 | 2.728 | 31.469 |
| c32 | CombineV3 | 14.584 | 2.628 | 4.541 | 10.043 |

The retained window8 pure-wire controls are 1.928 µs for c1 and 2.312 µs for
c32 per direction. Their boundary is narrower than the full native events.
Prefix corrections preserve real traffic but barely change the critical path
for 16 experts. Dispatch's large gap remains. Combine is within the declared
15 µs small-call tolerance, **not** within 15% relative error.

## Bounded PMU check

Only the first 278/291 local CSV rows were streamed for c1/c32, retaining four
dispatch and four combine samples per capture. All have BlockNum64, MIX_AIV
and zero reported Wait Time. Sample means are:

| Capture/op | Task µs | AIV µs | Vector µs | Scalar µs | MTE2 µs | MTE3 µs |
|---|---:|---:|---:|---:|---:|---:|
| c1 dispatch | 30.809 | 14.870 | 0.897 | 3.657 | 1.535 | 0.605 |
| c32 dispatch | 34.383 | 14.683 | 1.077 | 3.839 | 1.925 | 0.704 |
| c1 combine | 14.608 | 14.325 | 0.514 | 7.638 | 3.240 | 0.487 |
| c32 combine | 13.921 | 13.675 | 0.928 | 6.479 | 3.698 | 0.658 |

PMU pipes overlap; these aggregate metrics are not per-owner critical-path
traces. Task minus AIV time cannot identify launch overhead. Roles are strongly
asymmetric: one count AIV versus 63 payload AIVs for 16 experts. Peer waiting,
scalar address/control work, instruction/cache effects and selected branch
remain unresolved. The 30–34 µs task duration cannot be assigned wholesale to
Vector or used as a fitted protocol floor. Scalar PMU time cannot simply be
added to phases that already overlap it.

## Verification

70 focused tests pass. New real-Engine controls cover token-owner limits,
two active AIVs sharing a copy pipe, read/reduce/write ordering, prefix traffic
including idle experts, incoming-only ranks, delayed input and rate response.
Existing controls retain wire conservation and one invocation per direction.

Both 93-layer ordinary candidates complete with zero live tasks: c1
23.668259 ms and c32 30.740134 ms on the report's recorded branch runtime.
These are execution smoke results, not body-accuracy evidence. Reproduce with:

```sh
python data/verification/kimi-mc2-bf16-20261030.producer.py \
  --body-smoke -o data/verification/kimi-mc2-bf16-20261030.json
```
