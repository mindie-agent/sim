# Candidate studies

Development-run reports cited below are historical context and are not shipped. Current acceptance is documented in [verification](verification.md).

`mindie-sim study MANIFEST` connects the ordinary modeling paths into a finite
candidate comparison. It does not fit timings or require measurements.

Start with operator costs to find cost concentration. Run graph composition
where dependencies, communication or shared resources matter. Evaluate actual
requests separately for memory admission and service constraints. Independent
accuracy comparisons can then use measurements at their original boundaries.

```sh
mindie-sim study examples/studies/resource-tradeoff.json -o /tmp/study.json
mindie-sim accuracy explicitly-bound-accuracy-input.json -o /tmp/accuracy.json
```

## Input and execution

```json
{
  "candidates": [
    {"id": "baseline", "case": "../quickstart.json"},
    {
      "id": "memory",
      "case": "../quickstart.json",
      "overrides": {"machine": {"device": {"rates": {"memory": 1600000000000}}}}
    }
  ],
  "probes": [
    {"id": "prefill32", "phase": "prefill", "tokens": 32},
    {"id": "decode1k", "phase": "decode", "tokens": 1, "context": 1024}
  ],
  "options": {"level": "graph", "serving": true, "top_families": 8}
}
```

The CLI resolves case files relative to the manifest. A case may also be an
inline object. The Python `mindie_sim.study(candidates, probes, **options)` API
accepts case mappings or parsed cases; overrides require a mapping.
Candidates currently use colocated cases. P/D prediction remains available
through `predict`; its pools can be inspected as separate colocated cases.

Overrides merge mapping fields recursively and replace lists/scalars. They
modify the declared input structure, not arbitrary attributes of a compiled
model. Explicit `devices` and `layer_plans` must be edited as complete lists;
a top-level geometry label does not rewrite those records. Flat runtime parallel
axes are canonicalized before merging so an override cannot disappear behind
`runtime.parallel`. Conflicting simultaneous axis declarations are errors.
Changing CP/PCP on a normalized case also requires a consistent `cp_plan`.
Study rejects changed per-layer geometry scalars when explicit `layer_plans`
already exist and the override does not supply new plans.

Probes accept `phase`, `tokens`, `context`, `sequences`, `scope` and
`media_tokens_per_sequence`. Phase is prefill or decode. Tokens and sequences
are positive integers per participating replica; context is nonnegative.
Default scope is model. The workload is the first independent DP cohort,
including all participants when EP spans DP.

Operator screening accepts the complete replica mapping through the public
`operator_costs` API. Explicit cross-DP ring/pairwise calls are isolated as a
joint cohort once per physical signature, then wire work is attributed to
its actual sources. Their per-replica completion spans contribute to the
isolated sum; that sum is not the cohort's wall time. The `operators` CLI and
multi-model census use the same path. A missing participant remains an error.
An envelope isolated per replica has a different communication boundary from
a joint cohort. Operator-level study rankings keep these scopes separate;
the signature records each transport call's isolation so mixed dispatch/combine
choices cannot collapse into one category. It excludes algorithm selection:
different algorithms sharing the same isolation boundary remain comparable.
Use complete graph evaluations to compare isolation mechanisms at one boundary.

These are explicit target-forward probes: they bypass request admission and
exclude draft generation, sampling and token commit. `serving: true` runs the
case's original request workload separately; it does not reinterpret probe
geometry as serving requests. A probe can therefore exceed a scheduler's token
budget while the service result uses that budget normally.

| Option | Meaning |
| --- | --- |
| `level: operators` | Default: sum isolated device programs for the cohort |
| `level: graph` | Execute dependencies, streams and contention through the common engine; include isolated cost concentration |
| `include_submission: true` | At graph level, include declared host submission in graph elapsed time |
| `serving: true` | Also report actual-case service metrics, memory, assumptions and unsuccessful requests |
| `top_families` | Number of family/recipe contributions to show, default 12; omitted costs remain explicit |
| `max_evaluations` | Candidate count times probe count plus one per candidate with serving, default 64 |

`--primitives STORE` uses the same explicit primitive store as `predict` and
`graph`. Execution is sequential and retains compact reports rather than all
candidate graphs. The evaluation-count budget is not a wall-clock guarantee;
individual graph size still determines modeling time and working set.

Manifest JSON and every referenced file must load before execution. Missing or
malformed files return exit 2 and a visible input error on stderr, without a
study report. Once inputs load, expected case/modeling errors stay in their
candidate/probe records. Other candidates continue; a partial/failed study still
writes its report and exits 2. Unexpected implementation errors are not converted
to fabricated candidate results. A modeled but infeasible serving configuration
retains its constraint failures and is distinct from a failed simulation.

## Reading the result

Each successful evaluation includes its geometry, participating replicas,
declared physical-device count, cost concentration, program/boundary composition
and calibration basis. Graph results additionally contain elapsed time,
resource/precedence bounds and resource service accounting.

Cost concentration sums isolated durations times occurrences. It is useful for
choosing the next mechanism to investigate. It is not graph sensitivity,
measured error or removable critical-path time. Work totals describe the
representative local shards used by the engine, not an all-rank FLOP inventory.

Rankings group equal declared model structure, geometry, routing, cohort work
and timing boundary. Provenance-only model names do not split groups. Different
cohort sizes or routing realizations do, even if token/FLOP totals happen to
match. Failed candidates cannot become zero-cost winners. Device counts are
visible, but there is no implicit hardware-price normalization.

The result is a predicted ordering within an explicitly enumerated candidate
set. It is not a global optimum, measured ranking, automatic dispatch choice or
accuracy verdict. Parameter changes can expose a ranking reversal worth
investigating; they do not identify the correct physical parameter by themselves.

The explicit serial `phase_matmul` approximation can also reverse a ranking
by omitting matrix/copy overlap. Use it to screen cost concentration and graph
structure; compare finalists with the detailed mapping where that overlap
can change the decision. Conserved work alone does not establish conserved
timing or rankings. See `major-cost-audit.md` for a concrete same-work reversal.

## Connection to existing modeling approaches

The pinned official sources in [modeling-fidelity.md](modeling-fidelity.md)
inform the workflow: LLMCompass separates hardware declarations and operator
graphs; Vidur separates compute, communication and implementation overhead
profiles; AISimulate separates forward estimation, request replay and candidate
evaluation. The relevant guides were reread at those same pinned revisions for
this wave. No external simulator was run or benchmarked.

The resulting integration reuses `operator_costs`, `target_graphs`, `graph_costs`
and `predict`, all on the existing engine. It adds no alternate timing engine,
whole-forward residual fit or hardware/model-name correction.

The executable resource-tradeoff example uses illustrative parameters. It is
intended to exercise prefill/decode cost concentration and different resource
sensitivities. Actual deployment decisions still need appropriate hardware and
implementation declarations. For retained original observations, see
[capture adapters](capture-adapters.md); for heldout point/trend/ranking checks,
see [independent accuracy](accuracy-regression.md).

## Framed communication and finite credits

`examples/studies/communication-tradeoff.json` exercises the public study path
with the quickstart model at TP=2. All candidates use the same logical workload
and two physical devices. The alternatives change only shared-link bandwidth
or the finite credit count of the `allreduce` transfer recipe:

- `one-credit`: 1,000,000 wire bytes/ms and one outstanding transaction.
- `twice-link-bandwidth`: 2,000,000 wire bytes/ms and one transaction.
- `eight-credits`: the original bandwidth and eight transactions.

The illustrative link uses 384 payload bytes and 16 header bytes per
transaction, with 128-byte message alignment. The allreduce recipe holds a
credit until its 0.002ms post-service response. Link startup latency is zero, so
the response is not charged again as a fixed collective startup. Other
communication families retain their ordinary recipe; the per-operator ledger
shows the program selected for each one.

```sh
mindie-sim study examples/studies/communication-tradeoff.json \
  -o /tmp/study-communication.json
```

The 2026-10-29 run completed six graph evaluations. Predicted device-graph
durations, in milliseconds:

| Probe | One Credit | Twice Link Bandwidth | Eight Credits |
| --- | ---: | ---: | ---: |
| Decode, context 1024 | 0.193173 | 0.171573 | 0.081173 |
| Prefill, 17 tokens | 1.810478 | 1.654478 | 0.354478 |

These results demonstrate the selected model's credit-window sensitivity.
They do not recommend these parameters for a device or establish measured
ranking accuracy.

The retained
communication study report
contains the complete compact study result and a separate ledger obtained by
running the public `operators` CLI on the same six candidate/probe cases.
The study CLI itself summarizes operators rather than embedding this detailed
ledger. Each ledger keeps useful bytes, framed wire bytes, transaction counts,
physical resources, selected recipe and executed service. Its per-replica
sending-rank totals are 32,768 useful / 43,200 wire bytes for decode and
294,912 useful / 312,000 wire bytes for prefill, identical across candidates.
They are distinct from the study's representative per-rank `work.link_bytes`.

This example exercises message framing, shared link service, finite per-path
credits and the common graph engine. Collective round counts determine message
framing and declared startup; the transfer recipe does not add inter-round
causal dependencies or separate peer queues. It does not model router buffers
or infer a collective library's channel schedule. No serving workload,
measurement, calibration or hardware run is part of this example.
