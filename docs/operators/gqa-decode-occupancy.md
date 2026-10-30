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
