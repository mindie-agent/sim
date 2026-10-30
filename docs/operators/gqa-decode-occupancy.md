# Explicit GQA decode occupancy mapping

`staged_gqa_decode` accepts an optional `target_cores` owner budget. It first
counts independent batch × KV-head × query-tile tasks, then splits each task's
KV loop by `ceil(target_cores / query_tasks)`. The explicit `kv_splits` value is
a minimum. Real nonempty key tiles cap the number of contributors. Split
accumulators and their barrier-dependent reductions retain their HBM traffic,
vector service and ownership.

```json
{"recipes": {"flash_attention_decode": {
  "kind": "staged_gqa_decode", "target_cores": 20,
  "query_tile": 64, "key_tile": 128, "slots": 2
}}}
```

This is an analytical mapping option. It does not infer an Ascend host selector
or change the existing fixed-split default. The actual hardware still limits
simultaneously executing owners. No rate, workload multiplier or measured body
duration enters the mapping.

The long-context A3 Qwen control exposed the significance of this distinction.
At K126976, Qwen3.6 TP4 has six local query heads, one KV head and D256. With
q4/B1, a fixed single partition leaves one producer owner scanning the entire
KV extent. The same staged program with 1/4/10/20 splits predicts
2694/679/276/145 microseconds respectively. It charges all KV bytes in each
case and adds actual partial-output traffic for multiple splits.

Complete-model controls are retained in
`data/verification/longctx-gqa-occupancy-control.json`. All four finish with
unchanged package fingerprints and zero live programs/tasks. These are
conditional predictions, not an accuracy scoreboard. In particular, removing
the oversized GQA contribution exposes underestimated GDN/FFN contributions;
opposing errors cannot establish whole-model accuracy.

Independent source review found two important limits. The arch22 FIA host
splitter balances work across query boundaries, whereas this simple mapping
gives every query the same split count. For 12 equal query tasks and 20 cores,
the simple mapping creates 24 halves: four owners execute two halves. A
global balanced partition can avoid that tail. For ragged KV lengths, equal
partition counts likewise underfill the long request's usable owner budget.
This option therefore remains explicit while source dispatch is established.

The retained native captures launch 20 AIC and 40 AIV blocks; that establishes
the launch grid, not the number of KV splits. ND tensor storage also does not
identify the API's TND/BSND scalar layout. arch22 D256 can select the legacy FAI
path, so the newer FIA `split_core.cpp` schedule is not automatically the
captured one. Tests verify causal arithmetic, nonempty partitions, physical
traffic, delayed-input ordering and actual resource cleanup, independently of
these dispatch uncertainties.

## FAI host ownership

`kv_mapping="fai_flash_decode"` selects the host partition from
`FAInferTiling::splitBN2S1GS2` and `fillCoreInfoForFlashDecode` in
ops-transformer revision `a3925fd49cd256ad478d2faf983bf2961b8ae2c8`,
`attention/fused_infer_attention_score/op_host/flash_attention_infer_tiling.h`.
This is the legacy paged TND route's small-query flash-decode mapping, separate
from the newer FIA `SplitCore` helper. The source branch requires positive
q <= 16, q × query-head group <= 128 and enough KV work relative to the query
and owner counts. Other branches raise a visible domain error when this mapping
is explicitly selected. Existing defaults are unchanged.

```json
{"recipes": {"flash_attention_decode": {
  "kind": "staged_gqa_decode", "target_cores": 20,
  "kv_mapping": "fai_flash_decode", "query_tile": 128,
  "key_tile": 128, "slots": 2
}}}
```

Each owner gets `ceil(sum(q * heads * KV) / cores)` scheduling work and
consumes whole 512-row host blocks across request and head boundaries. Excess
work from the last block stays on that owner; it is not a credit against the
next owner's budget. The device program can retain smaller operand tiles that
divide 512. All split producers finish before the merge stage, preserving the
source's global synchronization boundary.

The original C++ host, compiled locally with only field-accessor ABI stubs,
matched 192 independently enumerated schedules. The retained fixture includes
equal and ragged cases, query boundaries and tail blocks; no NPU runtime or
observed duration was involved. Common-engine tests independently check work,
owner placement, input readiness, global merge ordering and resource release.

Independent review additionally matched 873 valid schedules spanning ragged
query and KV lengths. Three source cases emitted an invalid terminal owner
starting at `batch` after the final request had completed. The simulator stops
after the last real block; it does not reproduce that host bookkeeping defect.
The tests retain those counterexamples. This is a source-compatible partition
with bounded completion, not byte-identical behavior for malformed source
terminal records.

For twelve q4 requests at K126976, the uniform two-way split gives four of
twenty owners two halves. The FAI budget crosses query boundaries instead.
The same analytical device pipeline changes from 2701 to 1621 us, including
the extra partial-output traffic. At B1/B4 it changes 145/543 to 150/547 us.
These controls are retained in `data/verification/fai-host-mapping-control.json`.
They establish a scheduling correction, not native timing accuracy: the
analytical device pipeline still does not reconstruct the FAI GM score/P/PV
scratch path, and the captured operator's API layout and binary tiling key
must agree before this host mapping can be called the observed dispatch.
