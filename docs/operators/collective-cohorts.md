# Cross-DP collective cohorts and balanced pairwise alltoall

`Topology.joint_plan` binds one deterministic transport plan for complete
physical collective groups across DP replicas. It does not infer a runtime
algorithm from a model, vendor or hardware name. The declared `algorithm`
selects only the transport schedule:

```python
plan = topology.joint_plan(
    op,
    {replica: useful_bytes_per_source_rank, ...},
    algorithm="ring",       # or "pairwise"
)

completions = plan.start_participants(
    engine, "expert.dispatch", ready_by_rank, owner="replica:0",
)
```

`participants` maps a replica id to the total remote useful bytes carried by
each physical source rank in that replica. Every physical rank in the selected
groups must be accounted exactly once. The plan keeps the unique groups
returned by `group_axis` and `source_stage`; missing ranks, duplicated ranks,
partial cross-DP membership and accidental merging of unrelated groups are
rejected. A zero volume is legal, so an expert can be idle while its peers send.

The plan exposes:

- `ranks`: all unique physical ranks in its groups.
- `groups`: the complete physical groups and their transport records.
- `family`: the declared collective family.
- `start_participants`: one completion event per rank, after that rank's last
  send service and its own last receive. Coverage of `ready_by_rank` is checked
  before any work is submitted.
- `start`: a joint completion event for existing single-readiness callers.
- `validate_path_model("direct" | "store_forward")`: direct plans reject a
  physical route with more than one hop.

There is no collective-wide ready gate. A delayed rank delays only traffic whose
source or destination is that rank.

## Ring schedule

The ring schedule keeps the existing receive-forward recurrence. For a ring of
`N` ranks, each rank sends `2*(N-1)` messages for allreduce and `N-1` for
allgather or reduce-scatter. Every sent message carries an equal uniform chunk.
`joint_plan` therefore requires equal per-source-rank volume across each
physical group; unequal tensor chunks cannot be inferred from a total.

The first send waits both source and destination entry. A later send waits the
rank's own previous first-hop service and the arrival of its predecessor's
previous message. It does not wait for every rank, nor for its own outgoing
propagation to finish. The same-ready path retains the existing asymmetric
propagation behavior: one slow edge does not become a global round barrier.

`Topology.ring_plan(op, replica, volume=None)` retains its local declaration
behavior. Complete local groups may delegate to the joint ring plan, while an
incomplete group that spans other DP replicas is still rejected. Existing
callers, messages and round counts are preserved.

## Pairwise alltoall schedule

The pairwise schedule declares `N-1` rounds. In round `r`, source index `i`
sends to `(i + r + 1) % N` and receives from `(i - r - 1) % N`. A source's total
volume is split uniformly over its `N-1` peers.

Round 0 waits both source and destination entry. A later send waits the source's
own prior first-hop service and its own prior contiguous receive. Receives may
be posted eagerly once a rank enters, so destination entry is sufficient for
later rounds; the schedule never adds an all-rank round barrier. Early arrivals
are retained only as actual in-flight events until the preceding round is
delivered. Rounds and messages are generated lazily; there is no retained
history of all rounds.

Every non-empty message is routed once by `Topology.route_message`. Each hop
executes wire service, then propagation, then the next hop's service. Framing
is applied once per message per hop. A zero-volume source creates no service,
wire bytes or latency; it still advances the logical round counters so the rank
can complete after its real receives. An all-zero group completes at readiness.

This is a balanced uniform-peer abstraction, not a hot-expert alltoallv
predictor. It does not have a per-peer byte matrix. `alltoallv` is rejected
until that explicit peer contract exists.

The frontend currently declares the same per-source volume for dispatch and
combine. That is valid for balanced equal-source workloads. It is not a general
inverse for unequal or idle DP sources: the actual return matrix is the dispatch
transpose, and for `N >= 3` peers it cannot be recovered from one source total.
General nonuniform probability combine remains an unresolved transport-contract
gap; no average, global-total or guessed inverse routing is used to hide it.

## Recipe integration

`PairwiseCollective(path_model="direct")` is transport-only and parallel to
`RingCollective`. Its `lower` requires only `link` native work. The per-hop
events replace aggregate `context.setup_ms`, and `minimum_ms` remains a real
completion constraint. It rejects non-transport native work rather than deleting
local copies or reduction arithmetic. A bound pairwise plan still executes when
the local source has no outgoing bytes: the cohort may have incoming traffic
for that rank.

The public recipe registry accepts `pairwise_collective` as well as
`ring_collective`. For a complete cross-DP EP cohort, `Execution.add_group`
binds one plan using each replica's per-physical-source volume and local ready
event. Each replica's operator completes after its own physical ranks; a slow
peer does not force already-completed ranks to wait at a final global barrier.
A primitive completion constraint applies only to that replica and begins at
its local ready event. It does not change wire service or delay unrelated
replica completions.

For example, a runtime can explicitly bind ordinary MoE dispatch/combine:

```json
{
  "kernels": {
    "recipes": {
      "expert_dispatch": {"kind": "pairwise_collective"},
      "expert_combine": {"kind": "pairwise_collective"}
    }
  }
}
```

The operators declare family `alltoall` and recipe `expert_dispatch` or
`expert_combine`, so the `recipes` map selects the implementation. A
`families.alltoall` binding remains valid as the family-level fallback.

The common frontend divides the token-assignment payload by
`min(TP, EP)` for both the probability-moment and forced-balanced paths. Its
`op.link_bytes` therefore denotes total remote useful bytes per physical source
rank, and separate ETP groups retain their own needed copies. Dispatch and
combine record `source_assignment_partitions`,
`traffic_model="uniform_peer_payload"` and their `routing_basis`. If a frontend
partitions source tokens among TP ranks, this field must already contain the
partitioned payload; execution does not partition it again.
The pairwise planner divides that volume among remote peers and omits self
transfers.

## Analysis boundaries

A single replica's `operator_costs` cannot isolate a cross-DP causal collective
without its peers' work. It reports
`operator_collective_requires_complete_cohort:use_graph_costs`.
`graph_costs` accepts the complete replica mapping and isolates each distinct
cohort once, with simultaneous inputs. Its `isolated.operators` retains one row
per replica: `duration_ms` is that replica's own completion span, while
`cohort_duration_ms` is the whole-cohort span. The sum of replica spans is
explicitly labeled and is not a collective wall time.

Per-row `serviced` and `communication` attribute wire to that replica's physical
source ranks. They never repeat the whole cohort's wire work on every replica.
`isolated.collective_cohorts` retains the actual shared-engine totals and
per-rank completions; attribution is checked against total executed service.
Repeated equivalent calls reuse one isolated result with occurrence counts.

The graph timeline retains each causal collective's local ready time. A joint
collective contributes zero to the conservative precedence bound, rather than
adding its entire isolated span after the latest peer entry. Resource-capacity
bounds remain active. Aggregate transport recipes retain their existing
rendezvous semantics; choosing a causal recipe is an explicit modeling choice.

## Controls and limits

`tests/test_collective_cohorts.py` uses scalar send/receive recurrences and hand
timings, not comparisons between production helpers. It covers delayed rank
readiness with other ranks progressing, zero-source/nonzero-receive, all-zero
groups, framing tails, per-pair versus shared capacity, multihop dependencies,
complete cross-DP ring groups, unequal-volume rejection, repeated lazy schedules
and released tasks/programs.

Independent acceptance adds 48 direct-topology cases with different entry
times, zero senders, unequal sender totals, pair-specific capacities,
propagation delays and transaction frames. A scalar rank recurrence calculates
every completion and directed link's wire bytes without invoking the production
routing or accounting helpers. These controls also check that bound zero-local-
work recipes retain incoming peer traffic.

These are analytic mechanism controls. Link rates, transaction framing and
propagation are declared inputs, not hardware accuracy claims. The pairwise
schedule assumes eager receive posting after a rank enters; a runtime that
requires a blocking receive-post handshake needs a different declared
mechanism. Cross-DP reduction compute, cut-through forwarding, packet
interleaving and explicit unequal peer matrices remain outside this module.

`tests/test_joint_collective_execution.py` also exercises the public frontend,
graph and serving paths. Its three-rank control has source volumes `(0,4,8)`,
source entry times `(0,2,0)` and independent 1 B/ms directed capacities. Replica
completions are `(6,8,8)` ms, so replica 0 can run a 1 ms consumer and finish it
at 7 ms while peers are still communicating. Isolating the cohort removes the
producer delay and gives `(4,8,8)` ms. The distinction is reported directly.

## Physical-source contract acceptance

The October 29, 2026 source-volume correction is independently recorded in
`data/verification/collective-source-audit-20261029.json`, including reviewed
revision and source hashes. For TP=EP=4, four tokens, top-k=2 and 32 BF16
elements per payload, probability routing previously charged 1,536 remote
bytes while forced balance charged the correct 384 bytes. Both already
computed two expert rows per physical rank. Applying the common source
partition restores 384 bytes without changing expert arithmetic or transport
rates.

Independent controls cover 32 combinations of TP/EP values 1, 2, 4 and 8 under
both routing policies: 24 execute public collective graphs and eight verify
the EP=1 frontend has no expert exchange. The oracle counts physical sources,
peers, needed ETP copies and each peer's 16-byte payload/4-byte header framing.
All executed useful/wire totals match and tasks/programs are released. A separate
41-test focused acceptance covers the new ordinary frontend contract and the
existing cohort dependencies. The corrected recipe declaration selects real
pairwise rounds through ordinary graph and predict calls.

The return-direction gap remains reproducible after that correction. With
DP=EP=2, TP=1, four source tokens only on DP0, top-k=1 and uniform probabilities
over two experts, both ranks compute two expected expert rows. Dispatch correctly sends
128 bytes from rank 0 to rank 1. Combine currently sends another 128 bytes in
that same direction, instead of returning rank 1's results to rank 0.
`predict` accepts and serves this case, so equal total wire bytes and zero
live tasks do not establish correct origin ownership. The report retains the
actual directed service events. This correction fixes source replication,
not general nonuniform combine; forced-balanced equal-source workloads retain
their symmetric supported contract.
