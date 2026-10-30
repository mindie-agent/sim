# Kimi TP8 sequence-parallel HCCL call services

Two explicit 93-layer c32 cases now bind all **187 AllGather + 186
ReduceScatter** calls through the normal model frontend. This includes the final
output AllGather. Original generic calls are replaced, not retained alongside
the source program. Per-rank AG input is 4 × 7168 × BF16 = **57,344 bytes**;
RS input is 32 × 7168 × BF16 = **458,752 bytes**.

The preserved p4k complete-forward population provides 8,602 AG and 8,556 RS
parent communication events. Their pooled medians set two observable call-service
parameters: **10.7015 µs AG**, **15.8495 µs RS**. The response is inclusive, applied
once to the whole native group. Source HBM work, concurrent directed links and
causal dependencies remain active, and a slower physical execution can exceed
this response. No Body time, model name, or token count selects a timing branch.

Only rank8, BF16, MESH_1D and these exact payload domains are assigned by this
calibration. A single payload/rank condition cannot identify separate peer,
copy-bandwidth and synchronization coefficients. The response therefore retains
the observable complete-call boundary instead of inventing those coefficients.

The p16k operator data were not used in this fit. Historical observations retain
`unassigned`; this is a new analysis split, not a claim of pristine historical
heldout data. Its AG/RS pooled medians are 10.755/15.9655 µs, giving **−0.50% /
−0.73%** median errors. One p16k AG lasts 961.961 µs; it remains in mean/max
statistics and Body timing instead of inflating the fitted steady call service.

## Source realization and capture evidence

The capture reports `hcom_allGather_` / `hcom_reduceScatter_` parents with
`Accelerator Core=COMMUNICATION`, `Block Num=0`, and no tensor or PMU fields.
Neither these events nor the retained launcher identify AIV/AICPU/CCU expansion.
The new cases explicitly choose **conditional AICPU_TS Mesh1D** for the physical
source plan; the observed response is not attributed to a proven AICPU binary.

Pinned HCCL `c2d6790464654ccf58aca39448de4b65fdc7b184` supplies:

- `src/ops/all_gather/algorithm/template/aicpu/ins_temp_all_gather_mesh_1D.cc:115`
  calls LocalDataCopy, pre-sync, concurrent mesh gather, post-sync; lines 246–301
  distinguish output and scratch copies.
- `src/ops/reduce_scatter/algorithm/template/aicpu/ins_temp_reduce_scatter_mesh_1D.cc:111`
  orders pre-sync, scatter, post-sync and output work. Lines 172–221 retain the
  local copy and ordered local reductions for separate scratch buffers.

These are the already implemented source contracts. No source control count was
converted into a fitted per-notify latency, and no new executor was inferred.

## Full Body results

Runtime base `ab37ffe5` plus observable-service implementation `2f48346b` (local
cherry-pick `46f17f1a`); package hash and every case/input hash are in the receipt.
Original source-complete case fields, compute programs and hardware rates stay
unchanged. The results below are conditional source-compatible full Body spans.

| c32 context | Observed mean ms | Predicted ms | Signed gap ms | Relative gap | Wall time | Peak RSS |
|---|---:|---:|---:|---:|---:|---:|
| p4k | 40.0187 | 28.4113 | −11.6074 | −29.00% | 18.02 s | 195.31 MiB |
| p16k | 44.6662 | 33.5981 | −11.0681 | −24.78% | 35.08 s | 202.58 MiB |

Both calls release all tasks, programs and buffers. This change closes the
AG/RS call-service mismatch; **the complete Body is still outside 15%**. Remaining
Body residual is not assigned to the now-aligned communication calls.

The one-layer HBM scan at 0.5×/1×/2× yields
**0.318252 / 0.265666 / 0.263771 ms**, with identical physical work. An additional
severe-HBM-slowdown test makes each collective exceed its observed service;
the response does not overwrite hardware execution. Four focused frontend
checks passed in 1.42 seconds, including all 373 boundaries and once-only fees.

Reproduce the cases, fit, source caller controls and two bounded Body workers:

```sh
python data/verification/kimi-hccl-c32-20261030.producer.py --run \
  -o data/verification/kimi-hccl-c32-20261030.json
python -m pytest -q tests/test_kimi_hccl_source_caller.py
```

The producer reads small profiler aggregation receipts, preserves their raw CSV
identities, and never whole-reads the 83 MB capture. Workers are serial, capped at
60 seconds each, with a 512 MiB acceptance budget. The analytic source case can
still run without `service_regimes`; missing observations never gate simulation.
