# V4 Pro SCFA context conditions, October 30

Development-run reports cited below are historical context and are not shipped. Current acceptance is documented in [verification](../verification.md).

The previous -29.21% difference compared a context-zero SCFA prediction
(835.600 us) with the pooled mean of 120 observed calls (1180.346 us).
It is not a same-context accuracy result. The retained serving configuration,
request and repeated tensor descriptors support a four-chunk interpretation;
the per-event `seqUsedKV` and selected-index values were not captured, so the
exact occurrence-to-context mapping remains conditional.

The report
retains the original request, capture identities, event row numbers and time
windows, four bounded public graph executions, serviced resources and source
manifests. No runtime, physical rate, observation, selector or cache-hit
parameter changed.

## What the capture establishes

The independent `serve_prefill.sh` declares `BATCH=1024` and passes it to
`--max-num-batched-tokens`. `pre4096.json` records one successful request with
4096 prompt tokens and one completion token. The raw
`pre4096.kernel_details.csv`, SHA-256
`5ffe5625942e4caf787ce047438576527b8513452d6dd6871b24682a19968abd`,
contains 248 `SparseAttnSharedkv` events.

Sorted by exact timestamps, their shape/dtype/layout sequence repeats exactly
four times. Each 62-event sequence has one SWA-shaped call, 31 CFA-shaped calls
and 30 SCFA-shaped calls. All query tensors are `[1024,8,512] BF16`; the SCFA
compressed-index slot is `[1024,1,1024] INT32`. Sequences are delimited by
descriptor count and equality, not a duration threshold.

| Chronological descriptor sequence | SCFA calls | Observed mean us | Minimum us | Maximum us |
| --- | ---: | ---: | ---: | ---: |
| 1 | 30 | 625.833500 | 617.908 | 632.427 |
| 2 | 30 | 975.139767 | 958.781 | 990.100 |
| 3 | 30 | 1397.102733 | 1375.912 | 1414.671 |
| 4 | 30 | 1723.308167 | 1687.627 | 1748.145 |

This is evidence for repeated graph structure and different chronological
populations. It does not establish layer identity, exact cache contents,
metadata owner ranges or the `seqUsedKV` value for any individual event.

## Predeclared context controls

The four contexts below were declared before execution. Each uses
`target_graphs` and `graph_costs` on one ordinary-frontend-selected SCFA
operator with q1024, eight heads, width512, window128, compression ratio4 and
top-k1024. All retain the original explicit offchip scratch condition,
20-AIC cyclic ownership declaration, 1.2 TB/s HBM rate and existing device
overrides. External inputs become ready at 11 ms after a graph start at 7 ms;
the reported operator duration subtracts that 4 ms input wait.

| Declared context | Predicted us | Serviced HBM bytes | HBM supply lower bound us |
| ---: | ---: | ---: | ---: |
| 0 | 835.600371 | 981,664,000 | 818.053333 |
| 1024 | 1813.126252 | 2,108,857,600 | 1757.381333 |
| 2048 | 2908.729620 | 3,310,255,360 | 2758.546133 |
| 3072 | 3762.680160 | 4,419,919,104 | 3683.265920 |

Under the explicit assumption of 30 calls at each of these four contexts,
the mean prediction is 2330.034101 us against the pooled observed
1180.346042 us: +97.40%, or +1149.688059 us. The report marks this
`conditional_reference_only` with `accuracy_pass: null`. It neither assigns
the contexts to observed occurrences nor selects a context by timing.
With this condition, both V4 Pro and V4 Flash's all-offchip source controls
overpredict; their earlier opposite residual signs did not demonstrate
opposite physical mechanisms.

## Workspace boundary and arithmetic bounds

The source writes aligned FP32 QK/PV workspace, reads the current scratch,
reads prior accumulators on nonfirst tiles, publishes nonfinal accumulators,
and writes final BF16 output. These transfers are already represented.
The aligned writes in the Cube source are not accidental padded-head work.
No inspected source directive establishes that every GM scratch transfer
misses cache.

The following alternate ledger removes only scratch from HBM while retaining
external input and output traffic. It is an arithmetic HBM supply bound,
not a resident Engine run, measured residency, chosen cache capacity/rate or
cache-hit estimate. Scratch transfers would still require their private
copy pipes and a declared cache service if executed.

| Context | Remaining external HBM bytes | Scratch read/write bytes | HBM-only resident bound us |
| ---: | ---: | ---: | ---: |
| 0 | 411,933,952 | 569,730,048 | 343.278293 |
| 1024 | 699,137,280 | 1,409,720,320 | 582.614400 |
| 2048 | 978,033,920 | 2,332,221,440 | 815.028267 |
| 3072 | 1,248,591,104 | 3,171,328,000 | 1040.492587 |

For comparison, the retained V4 Flash q4096/context0/top-k512 offchip
control services 8,417,286,400 HBM bytes, giving a 7014.405333 us supply
bound. Its scratch-resident arithmetic endpoint leaves a 2317.002453 us
HBM bound. Neither endpoint identifies a historical hit rate.

Source revision `175ef4d49c2072f9233619636f0309f4f88b4e2c` derives the selected
row count from actual KV length and query position in
`sparse_attn_sharedkv_scfa_kernel.h:363`. Its AICPU metadata builder partitions
contiguous work ranges by cost; the current source scenario explicitly uses
cyclic one-query-per-core ownership. Historical metadata and binary dispatch
remain unresolved. Workspace placement and metadata geometry therefore remain
separate mechanism questions; no residual-derived multiplier is justified.

## Execution identity and reproducibility

All four children record equal before/after runtime identities:

- Git revision: `26f5adb48dd418584cd18c79d6a63ca8ab42afd7`.
- Package SHA-256: `10c4eb73da5301c30817c81fc47142acd6ab5d45350eb444cdb6375964f553ee`.
- Scope: 213 `.py` and `.json` files below `mindie_sim`.
- Algorithm: iterate sorted package-relative paths; append UTF-8 path,
  a NUL byte and hexadecimal file SHA-256 to an outer SHA-256.

External C++ source hashes were checked after execution and are labeled
`post_run` in the report. They are not represented as runtime before/after
checks.

The retained producer
is byte-identical to the executed producer, SHA-256
`51bcd82403398865c7f6254615e1c6d0df1821c5573ce58bac15bbca77088ea9`.
It regenerates the raw capture and run evidence; the report also contains
subsequent source-audit and boundary annotations.

```sh
.venv/bin/python data/verification/v4pro-scfa-context-20261030.producer.py \
  --root "$PWD" --output /tmp/v4pro-scfa-context-replay.json
```

Replaying requires the original mounted capture paths and the retained
`a3-pipeline-convergence-20261030.json` selected case. Each child has a
15-second budget and their combined budget is 60 seconds. This run completes
in 11.68 seconds, with maximum child RSS 54.3 MiB. Each graph retires all
tasks, programs and allocations. Task counts are 32,932, 35,025, 52,388 and
54,481; peak live task counts are 700, 740, 775 and 825. The retained 64-event
trace excerpt confirms input readiness for its positive-demand events, not a
complete timeline inventory.
