# Participant-aware collective dependencies

`collective-plan` runs an explicit, non-streaming collective DAG on sim's
existing shared-resource engine. A collective waits for every declared
participant's producer, services a single physical wire-work graph, and releases
per-participant consumer gates. A zero-payload participant still takes part in
the rendezvous. Unrelated groups have no added dependency on each other.

```bash
python -m mindie_sim collective-plan \
  --input examples/collectives/shared-link.json \
  --output collective-result.json
```

This command is analytical and needs only the Python standard library. The
example uses normalized synthetic capacities; its milliseconds are **not NPU
measurements or an Ascend performance prediction**.

## Explicit input contract

The input declares `schema_version: 1`, `resources`, `producers`, `collectives`
and `consumers`. `name` and `source` describe the case. No physical machine or
transport algorithm is inferred from a model or a parallelism label.

| Field | Meaning |
| --- | --- |
| `resources[].name/capacity/unit` | Work capacity per millisecond; e.g. `link:fabric`, `100`, `bytes` means 100 wire bytes/ms. Each physical resource must have one unique name. |
| `resources[].scope/policy` | Passed to `ResourceSpec`; the default policy is `equal_share`. Scope is an annotation; sharing is determined by the resource name. |
| `producers[]` | Tasks with `task_id`, resource `demands`, optional prior `dependencies` and `release_ms`. All are registered before collectives. |
| `collectives[].kind/group/epoch/stage/layer` | Complete operation identity. `kind` is one of `all_reduce`, `all_gather`, `reduce_scatter`, `all_to_all`, `broadcast`, `barrier`. All five fields are mandatory; numeric fields are nonnegative integers. |
| `participants` | Explicit unique nonnegative rank IDs, including empty/zero-payload ranks. No world group is assumed. |
| `payload_bytes_by_rank` | Logical payload evidence, keyed by the exact participant set. Zero is valid. These bytes do not create engine work. |
| `producers_by_rank` | One or more known producer task IDs for **each** participant, even with zero payload. Missing and extra members are errors. |
| `link_stages` | Topologically ordered physical communication stages: `name`, `link_bytes`, optional prior stage `deps`. Root stages wait for the participant join; completion waits for all terminal stages. An empty graph is allowed for `barrier`. |
| `link_stages[].link_bytes` | Explicit physical work for the whole collective on existing `link:*` resources with unit `bytes`. Each stage is charged once. Multiple physical wires may be declared. |
| `consumers[].rank/wait_for` | `wait_for` contains complete collective identities. The consumer depends on that rank's gate for each listed identity. Unknown epochs/groups and nonparticipants are rejected. |
| `consumers[].task_id/demands/dependencies/release_ms` | Normal engine task fields, as for producers. All consumers are registered after the collective graphs. |

The descriptor is an execution dependency contract, not an implementation of
AllReduce or AllToAll byte formulas. **Logical payload is deliberately separate
from physical wire work.** The caller must supply the algorithm, route, padding
and aggregate-vs-edge accounting in the stage work and `source`. Repeating
per-rank payload or multiplying by group size is not done automatically. Two
groups that name the same physical `link:*` resource compete for its capacity;
different physical resource names are disjoint in the engine.

Producer/consumer task `demands` may include separately declared work. Such work
is not part of the collective byte bill. The report separates per-collective
wire accounting from the engine's total service counters.

## Example: slow empty member and one shared wire

`examples/collectives/shared-link.json` has four independent local resources and
one shared wire with capacity 100 bytes/ms:

| Group | Participant producer completion | Logical payload bytes by rank | Declared physical wire bytes |
| --- | --- | --- | --- |
| `a` | rank 0: 2 ms; rank 1: 5 ms | rank 0: 1024; rank 1: **0** | 500 |
| `b` | ranks 2 and 3: 1 ms | 512 each | 600 |

These physical byte values are an intentionally normalized fixture, **not an
AllToAll traffic formula**.

1. Group `b` starts at 1 ms, while group `a` is waiting for rank 1. It receives
   100 bytes/ms until 5 ms, completing 400 bytes.
2. Group `a` starts at 5 ms despite rank 1 having no payload. From 5 to 9 ms,
   both receive 50 bytes/ms. Group `b` finishes its remaining 200 bytes at 9 ms;
   group `a` has completed 200 of its 500 bytes.
3. Group `a` receives the full 100 bytes/ms from 9 to 12 ms and finishes at
   12 ms. One millisecond of local consumer work finishes at 10 ms for group
   `b` and 13 ms for group `a`.

Total serviced wire work is **1100 bytes**, not 2200 or the sum of logical
payloads. With separate wires, group `a` finishes at 10 ms and group `b` at
7 ms. The unit tests check both cases, the actual piecewise service rates,
zero-work gates, exact group membership and independent epoch/stage identities.

## Reading the original sim JSON output

| Output | Evidence |
| --- | --- |
| `input_sha256`, `effective_input`, `source` | Canonical input signature and exact declared input; the source names the assumption, not a measured claim. |
| `collectives[].descriptor` | Participant, producer, payload and wire-stage contract. |
| `collectives[].producer_completion_ms` | Completion time of every producer for every member. |
| `collectives[].rendezvous_ready_ms` | Zero-work join completion after all member producers. |
| `collectives[].complete_ms` | Completion after all terminal wire stages. |
| `collectives[].task_ids` | Join, wire-stage, completion and per-rank gate IDs for locating engine events. IDs hash the structured identity to avoid delimiter collisions. |
| `collectives[].byte_accounting` | Declared physical bytes vs bytes integrated from actual service intervals; zero-work joins and gates add no traffic. |
| `execution.trace[].service` | Actual allocation by resource and time interval. Shared-wire throttling is visible here. |
| `execution.serviced`, `byte_accounting.all_work_serviced_link_bytes` | Total service including separately declared producer/consumer work. |

Validation failures return the CLI's existing `{status: "error", command,
reason}` JSON with exit code 1. Unknown producers, mismatched member sets,
duplicate identities, forward stage dependencies and unknown physical links
cannot produce a successful empty report.

## Python API integration

```python
from mindie_sim.collective_plan import CollectiveDescriptor, register_collective

# The existing engine already owns physical link resources and every producer.
descriptor = CollectiveDescriptor.from_mapping(declared_collective)
registered = register_collective(engine, descriptor)
engine.add_task(
    "rank9.consumer", {"rank9:vector": 1000},
    dependencies=(registered.consumer_gate(9),),
)
result = engine.run()
```

`register_collective` validates the complete descriptor, all producer IDs,
physical resource units and generated-ID collisions before appending tasks.
It does not create resources or change engine allocation policy. The returned
gates can also be passed into `add_costed_ops(..., op_external_deps=...)` when
subsequently registering a costed consumer graph.

Registration order is **all producers → collective → consumers**. Existing
`ResourceEngine.add_task` still requires known dependencies. For multiple epochs,
different keys prevent accidental merging; the caller must explicitly provide
any desired epoch-to-epoch dependency. List order alone does not serialize
collectives. A strategy adapter can use the Python API incrementally to build
such dependencies.

An epoch distinguishes identity; it does not implicitly order operations on the
same communicator. Express the required order explicitly in producer dependencies.

## Boundary of this change

The first version models non-streaming completion rendezvous and explicit
physical link service. It does not infer startup latency, local endpoint
HBM/MTE/AIV work, algorithms, topology routing, transport feasibility or
streaming chunk overlap. Those costs need explicit future stage extensions or
upstream cost/strategy adapters. No endpoint implementation is certified here.

The existing `predict` model graph is unchanged; there is no automatic EP/DCP
adaptation or global DP barrier. This API is the reusable building block for a
later strategy PR that supplies actual participant groups and splits model
graph registration at the collective boundary.

```bash
python -m pytest -q tests/test_collective_plan.py
```
