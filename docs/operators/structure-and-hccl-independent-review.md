# Structure identity and explicit HCCL source review

Development-run reports cited below are historical context and are not shipped. Current acceptance is documented in [verification](../verification.md).

The review found no blocking defect in the streamed structure identity, NHR
round contracts or narrow CCU contracts. This is a mechanism/identity review,
not captured dispatch proof or measured performance acceptance.

## Complete streamed identity

The retained control
compares the complete exporter immediately before `40e4300e`, the current full
exporter, independently encoded canonical JSON and streamed selection. All 16
inputs were equal: body/model scopes, nonuniform and idle DP participants,
TP+EP, source-call expansion on/off, selected native fusion and serialization
projection. Three target subsets per input (including empty) and reversed
replica insertion order preserved identity. The controls cover 996 nodes,
46 subgraphs, two source calls and four selected fused resource records.

Complete cohort peer declarations are built before serialization. This does
not create a new cross-rank execution-dependency encoding: node dependencies
keep the exporter's existing replica-local convention, while complete bound
peer matrices remain in metadata. Four spools bound serialized record memory;
the generated cohort and selected target records still occupy memory. The
review does not claim constant memory for arbitrary cohort size.

The combined affected structure/HCCL run passed 47 tests, with four existing
profiler-integration tests skipped because the required companion development
API (`normalized`/`tensor_contracts`) is unavailable. Independent controls neither
require those skips nor claim capture matching coverage from them.

## NHR and CCU

The independent source/Engine receipt
records pinned HCCL `c2d6790464654ccf58aca39448de4b65fdc7b184` hashes,
252 rank/round arithmetic controls and 74 actual Engine executions.

NHR preserves cyclic peers, first-RS transmit-copy elision, remote WriteReduce
source read plus destination read/write, and final-AG remote output reads.
Standalone AG's local final copy overlaps the last network round; AR's copy
follows its single communication thread. Every directed payload appears once
on the wire. Controls cover power-of-two ranks 2–64 and independent copy,
reduce, control, link and HBM rates with input ready at 17 ms.

CCU tests cover V1/V2, ranks 2/3/4/8, the <=4096-byte one-shot reduction domain,
32 KiB copy tails and eight-slot group boundaries. Remote one-shot loads land
in CCU SRAM, not HBM scratch. AllGather local copy remains read→wait→write→wait;
source peer writes may overlap it. Missing CCU service rates are explicit
unpriced work, not evidence of zero control/reduction cost. SRAM handle
capacity and submessage arrival progress remain outside the implemented model.
Source selector throughput guesses were not imported as physical rates.

Public CLI checks
passed all 12 HCCL controls, adding NHR AR/AG/RS and CCU V1/V2 AR/AG to the prior
five. They check real frontend selection, one complete invocation, late input,
positive local/wire work and zero live resources. This receipt is a checkout
run; final installed-wheel acceptance belongs to the parent verifier.

## A3 native matching is distinct from missing simulation work

The retained semantic inventory
uses the exact two case hashes from the profiler's `30dcea1e` borderline ledger
on reviewer runtime `ecce7b78`. Captured q/B are retained; context 4096 is an
explicit conditional probe, because the ledger does not observe context values.
It does not revise the historical body residual.

- V4 Flash's unmatched 62 Compressor events coexist with 41 ordinary compression
  and 21 index-compression blocks, each with matrix and pooling work. Native
  matching is missing; the whole family is not absent from the graph.
- The unmatched TransposeBatchMatMul has 43 `grouped_output_a` matrix candidates
  with nonzero FLOPs/HBM. Native layout/fusion compatibility must be resolved
  before replacing that work, rather than adding the native event's elapsed time.
- DequantSwigluQuant/DequantSituQuant coexist with output representation work,
  activation and quantization programs. V4 declares 344 dynamic quantizers;
  Kimi declares 25, plus four Situ and three grouped-Situ programs. Matching a
  fused native boundary must replace its equivalent constituent work.
- Kimi's four-layer graph already contains nine AllGather and eight
  ReduceScatter calls, three explicit arch22 KDA plans, and native-unmatched
  generic MoE dispatch/combine. New source protocol programs replace these
  mechanisms only within a supported explicit ABI domain; V2 transport is not
  silently relabeled as the existing arch35 V3 program.
- Kimi FIA's corrected conditional reference is all 32 sequences (160 us), not
  one 5 us program. Physical 16-head native output versus 12 logical heads and
  unknown context still prevent an unconditional comparison. The 1.375535
  ms/F `reduce_scatterAicpuKernel` overlay is controller elapsed time and must
  not be added to communication service.

Useful next substitutions are source-compatible A3 short convolution, the
actual A3 paged MLA head/dispatch domain, and explicitly selected HCCL templates
where captured attributes establish them. An arch35 MLA buffer contract that
exceeds A3 UB cannot be enabled by removing the capacity check. None of the
unmatched native activity values is a residual floor or an additive body budget.
