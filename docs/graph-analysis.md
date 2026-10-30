# Subgraphs and pure-model execution

`graph_costs` runs declared operator DAGs through the same `Execution`, recipes
and resource engine used by serving. It adds no request arrivals, scheduler
decisions, cache admission, sampling or token commits.
Measurements remain optional.

```python
from mindie_sim import graph_costs, target_graphs

graphs = target_graphs(case, phase="decode", tokens=1, sequences=2,
                       context=4096, scope="model")
report = graph_costs(case, graphs)
print(report["duration_ms"])
print(report["isolated"]["operator_sum_ms"])
```

```sh
mindie-sim graph examples/quickstart.json \
  --phase decode --tokens 1 --sequences 2 --context 4096 --scope model \
  -o /tmp/model-graph.json
```

`tokens` is query tokens per sequence; `sequences` is sequences per participating
DP replica. `target_graphs` selects the first independent DP cohort. An EP group
that spans DP includes all its replicas and derives routing from their combined
workload. This is explicitly not the sum of every independent data replica or a
request throughput result. TP/PCP ranks retain the engine's representative-shard
assumptions; PP stages retain actual stage resources and transfers.

`body` starts with already available hidden inputs and uses unpadded requested
geometry. Pass `external_inputs={"analysis_input": ready_time}` when using its
graphs through the Python API. The CLI declares this input itself.
`model` preserves the frontend's embedding, normalizations, selected logits,
capture padding, graph barriers and source-kernel expansion.

`draft_prepare=True` (CLI `--draft-prepare`) explicitly includes the declared
draft cache preparation after target prefill. It requires prefill/model scope
and a runtime draft mode with model auxiliary geometry. The default excludes
this branch. For an MTP1 capture, declare `model.auxiliary.mtp.layers=1` and
its actual geometry; a catalog's three-layer DSpark is a different workload.
Dependencies through excluded sampling/commit nodes are contracted to their
device predecessors. This remains a pure device-model graph. The scoreboard's
scheduled-step driver instead retains sampling, commit and the complete
iteration completion boundary.

Media allocation must be explicit for synthetic analysis requests. If the case
contains media inputs, prefill `model` scope requires
`--media-tokens-per-sequence N`; `N=0` explicitly excludes the tower. A positive
value assigns the declared processed media geometry to each analysis sequence.
The frontend still lowers that geometry using the model's media specification.
Media work is not silently lost when analysis request IDs differ from workload
request IDs. Decode and body scope do not execute the tower.

## Boundaries

The default `device_graph` boundary includes device computation, materialized
copies, transfers, collective setup and shared physical resource contention.
Host scheduler/submission/completion costs are excluded.
`--include-submission` selects `submitted_device_graph`, including those declared
host costs on the same engine. Neither boundary is TTFT or TPOT.

The target frontend's selected capture scope is inherited, including for
primitive identity matching. `--replay-scope` explicitly overrides that scope
while retaining already lowered operator shapes; this is a counterfactual
submission policy. Custom operator lists without a target scope marker default
to `none` and can pass `replay_scope` explicitly.

For a custom subgraph, pass a topologically ordered sequence of `OpSpec` records.
A mapping from replica IDs to such sequences describes concurrent replica
graphs. Every dependency cut at the boundary must have an explicit
`external_inputs` ready time; unknown predecessors are errors. Omitted/empty
stream names use the main stream. Data dependencies, stream order and physical
contention are separate constraints.

EP cohorts synchronize their inputs, execute each participant's actual recipe,
and join their completions. They preserve transfer startup, per-invocation
supply caps, phases and optional calibrated service. Summing demands into a
substitute task would lose these mechanisms.

## Reading the result

- `duration_ms`: elapsed time from the declared graph input boundary until all
  selected graphs finish.
- `timeline`: physical operator boundaries after source calls expand. It keeps
  semantic dependencies and stream predecessors. Collective rows distinguish
  local readiness, rendezvous waiting and common readiness. An operator's
  elapsed interval may include resource or buffer contention.
- `isolated`: each physical signature executed independently on the matching
  replica/stage, with the same recipe and primitive scope. Its weighted sum is
  work concentration, not an overlapped graph duration. Across a DP cohort the
  sum includes all participating replicas.
- `bounds.precedence`: a longest path of isolated costs with data, stream and
  collective rendezvous dependencies. `bounds.resource_capacity_ms` is the
  maximum actual serviced work divided by physical resource capacity. Their
  maximum is a lower bound for this declared model, not a measured critical
  path, confidence interval or achievable optimization saving.
- `resources` and `execution`: actual physical work, capacity, busy intervals,
  live tasks and per-owner buffer peaks. Buffer peaks cover the executed
  programs; they do not replace serving's weight/KV/workspace capacity checks.

Use `--skip-isolated` to omit the extra independent runs and bound calculation.
Detailed engine trace is disabled by default and bounded by `--event-limit`.
The retained state scales with one input graph and active finite programs, not
with generated output history.

Structural tests cover independent overlap, shared bandwidth, stream order,
external arrivals, submission boundaries, capture, media, source families and
EP startup/supply. Catalog graph runs are structural execution, not real-device
timing acceptance. Layered timing and configuration ranking require matched
independent observations.
