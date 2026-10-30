# Explicit per-DP body geometry

`target_graphs` and `predict(scope="body")` accept either the existing positive
integer `sequences`, or a list of nonnegative counts in the first complete EP
cohort's DP-replica order. At least one replica must be active. For TP1/DP8/EP8:

```python
predict(case, scope="body", phase="decode", tokens=1, context=2048,
        sequences=[1, 0, 0, 0, 0, 0, 0, 0])
```

`sequences=1` still means one sequence **per participating replica**, hence
eight global queries in this example. The list above means one global query.
`predict_pd(..., scope="body")` accepts the same declaration for each pool's
complete cohort. Scalar declarations remain useful when the pools have
different cohort widths; a list must match each pool it describes.

Each DP replica selects graph capture and padding independently. An empty
replica receives no dummy query, attention, router or graph padding. It still
joins expert dispatch/combine and computes the expert rows routed to it.
Graphs can therefore contain captured active replicas and eager empty
participants. Graph/operator costing keeps their replay scopes separately;
it does not force an idle participant to inherit another replica's capture.
Pure body probes continue to exclude host submission.

Forced balancing retains one exact global integer expert histogram. A DP
replica now chooses its heaviest local TP shard rather than reusing global
EP rank zero everywhere. With TP1, local histograms sum exactly to the global
assignment count. With TP greater than one, each simulated DP replica remains
a representative of its heaviest local TP shard: multiplying its matrix work
by TP is an upper bound, not an exact physical-rank sum. Metadata exposes
`represented_ep_ranks`, `representative_rank`, `rank_assignment_rows`, and
`rank_timing_assumption`. The built-in serving frontend uses the same replica
selection; custom serving lowerers keep their existing callable contract.
The same replica identity reaches MTP/DSpark draft preparation and proposal
generation. An idle replica that receives expert rows uses the active cohort's
prefill/decode phase (or `mixed` for a mixed cohort), rather than an `idle`
phase that would change the physical operator identity. Model-scope analysis
still excludes sampling/commit while preserving their aliased dependencies.

This does not recover a token-to-expert assignment matrix. Communication
continues to use the explicitly labeled uniform-peer expectation, and combine
transposes that directed ledger. Exact balanced expert counts and expected
communication payload are different model assumptions; fractional expected
peer rows must not be presented as observed integer routing.

## V4.1 evidence and controls

`data/verification/v41-dp8-cohort-geometry-20261030.json` retains source identities,
180-row bounded CSV samples for each source, and actual first-layer Engine
controls. No full kernel CSV scan or hardware run is performed by its producer.

The c1 and c8 captures both retain only DP0/rank0. The profiler identifies
152 and 145 complete 40-layer target bodies, respectively, all with rank-zero
M/B/q = 1/1/1. Their body P50 values are 13.1292645 and 12.802981 ms. Native
GroupedMatmul has allocated M=48 and a 48-entry group list; list values and
valid received rows are absent. The benchmark records one/eight running
requests, but its `hc_bench.py:193–215` sums gauges across engines and discards
rank labels. Client concurrency, this global gauge, and buffer M=48 do not
establish peer query padding, valid rows, or expert assignments. No conclusion
that c1 is symmetric or single-active follows from these receipts alone.

For the same first V4.1 MoE layer, q1/context2048:

| Conditional cohort | Global expert rows | Rows by rank | Dispatch + return wire | Graph ms |
|---|---:|---|---:|---:|
| `[1] * 8` | 48 | `[6,6,6,6,6,6,6,6]` | 860,160 B | 0.195475 |
| `[1,0,0,0,0,0,0,0]` | 6 | `[1,1,1,1,1,1,0,0]` | 107,520 B | 0.170506 |

The first case reads 75,202,560 bytes of gate/up expert weights per rank.
The second reads 12,533,760 bytes on each of ranks 0–5 and none on ranks 6–7.
All ranks retain collectives; both actual Engine runs finish with zero live
tasks/programs and preserve the directed transpose and wire ledger. These are
conditional one-layer controls, not full-body accuracy results. Neither row
is installed as the measured cohort geometry.
