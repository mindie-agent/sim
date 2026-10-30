# MXFP4 physical owner template reuse

The panel compiler now separates physical arithmetic from rank labels and unit IDs. This is an exact compilation optimization. It retains the existing finite-refill approximation and creates the same Engine events in the same order.

## Cache contract

`compiled_panels` retains its bounded 1,024-entry cache of physically bound immutable IR. A new 128-entry template cache compiles the same source builder once per physical geometry. `cache_info()` still describes the bound cache; `template_cache_info()` describes the inner cache, and `cache_clear()` clears both.

A template key includes resource alias relationships, rates, unit counts, compute peaks, directional service modes, shared-client counts, source tile geometry and actual repetition count. For each physical pipe it retains the relative placement of the AIC and both AIV programs. It does not assume one pipe per owner: two AIVs can share a pipe, an AIC and AIV can alias one physical resource, or an owner can occupy multiple units. Physical resource sort order is retained so nominal producer compilation keeps its numeric iteration order.

Binding replaces resource names and relative unit IDs with their original physical identities. Stage order, work, caps, dependencies and numeric values remain unchanged. Cached definitions are immutable; execution state, callbacks, allocations, and operator metadata are not retained. Buffer allocation and final output remain with the caller program.

## CPU control

The retained report is `data/verification/v41-owner-template-profile-20261030.json`; its producer reuses the existing bounded source-case harness. This is a three-layer diagnostic of the full 40-layer case, with q1, B8 on each of eight declared DP replicas, context 2048, original quantization and source recipes. The symmetric peer workload remains a declared condition, not inferred observed `group_list` values.

Both runs use `cProfile`, the same 25-second bound and the same input hash. Baseline runtime is `0e2d144f`; current runtime is `99610ae7`, whose only runtime-file change is `prediction/mxfp4_panels.py`.

| Quantity | Original bound-only cache | Two-level cache |
|---|---:|---:|
| Wall seconds | 8.3861 | 6.6279 |
| CPU seconds | 8.3060 | 6.5212 |
| Bound cache misses | 512 | 512 |
| Physical template compilations | 512 | 3 |
| Peak RSS MiB | 91.45 | 91.33 |
| Simulated body ms | 2.549926873324732 | 2.549926873324732 |
| Actual Engine tasks | 119,796 | 119,796 |
| All submissions, including nominal producer controls | 119,961 | 119,961 |

CPU fell 21.49% in this instrumented control. Complete Engine summary, parameter report, per-family elapsed totals, task counts, physical service and body span have identical fingerprints. Both runs finish with zero live tasks/programs. This is neither a full 40-layer speed claim nor an accuracy result.

Before optimization, the main disjoint CPU branches were panel compilation (2.107 s), resource rate allocation (1.433 s), event submission (1.221 s), ordinary steady-matrix lowering (0.765 s), and Engine advance's own arithmetic (0.487 s). `_settle` and `_complete` cumulative times also contain lazy compiler callbacks and must not be labeled pure scheduling cost. Within rate allocation, partition filling cost 0.533 s and coupled filling 0.487 s.

The largest task families were MXA8W4 (48,048), ordinary GEMM (33,152), MQSMLA (8,872), HC pre (8,784), and dense down projections (6,456). MXA8W4 plus ordinary GEMM account for 67.7% of all submissions. The cache does not reduce their event count.

After optimization, rate allocation still costs 1.458 s, submission 1.252 s, and ordinary steady lowering 0.768 s. Cold template reuse cannot by itself turn the previously measured 39.7-second full model into a sub-five-second model. Safe further work can investigate immutable ordinary-matrix lowering or equivalent resource-indexing arithmetic. Removing zero-work wrapper events is excluded: an independently established counterexample changes allocation arbitration order.

## Verification

71 affected tests pass. The new controls compare complete source IR and actual Engine traces, resource service, late input and cleanup against the uncached `panel_parts` oracle. They cover eight ranks, 32 owners, 16/32/64/128-unit pipes, AIC/AIV shared-pipe aliases, AIV read/publish aliases, directional MTE1, reversed resource-name order, shared-client changes, rates from 0.5x to 2x, real tails, and bounded eviction. No physical rate or modeled duration changed.
