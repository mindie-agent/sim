# Declared ring collective rounds

`ring_collective` executes a uniform-chunk transport schedule on the ordinary
physical `Engine`. It is an explicit configuration choice for `allreduce`,
`reduce_scatter` and `allgather`; the simulator does not infer which collective
algorithm a runtime selects. Ordinary frontend link work is transport only.
Reduction arithmetic and its local memory traffic are not silently invented.

```json
{
  "runtime": {
    "kernels": {
      "families": {
        "allreduce": {"kind": "ring_collective"},
        "reduce_scatter": {"kind": "ring_collective"},
        "allgather": {"kind": "ring_collective"}
      }
    }
  }
}
```

`examples/ring-collective.json` is a small synthetic frontend/serving control:

```sh
python -m mindie_sim graph examples/ring-collective.json --phase prefill --tokens 4 --scope model
python -m mindie_sim predict examples/ring-collective.json --event-limit 1000
```

## Causal and physical contract

For a ring group of `N` ranks, allgather and reduce-scatter send `N-1` messages
per rank; allreduce sends `2*(N-1)`. The frontend's `link_bytes` is already the
total useful traffic per rank, so each uniform message carries
`link_bytes / rounds`. A generic operator can declare its collective kind in
`metadata.collective`; accounting and execution use the same declaration.
Explicit `metadata.steps` must agree with the declared ring algorithm.

A rank's next message depends on two events: its own previous first-hop service
has completed, and the previous message from its predecessor has arrived.
There is no collective-wide round barrier. A rank does not wait for its own
previous propagation delay before sending again if its required input has
already arrived. Physical link capacity is shared by all concurrent tasks,
including other operators and other ring calls.

`Topology.route_message` owns framing for both accounting and execution. It
aligns each message and adds the declared protocol header on each physical
hop, before aggregation. Each hop executes wire service followed by its
propagation latency. A subsequent hop receives useful payload and applies its
own framing; the previous hop's header does not become downstream payload.
The aggregate `CommunicationWork.setup_ms` remains an accounting estimate for
the legacy envelope, but the ring program does not add it to the already
explicit propagation events.

The default `path_model="direct"` requires one physical hop for each ring
neighbor. `path_model="store_forward"` explicitly selects whole-message,
sequential forwarding for multiple hops. This does not model cut-through,
packet interleaving or NIC credit returns. The sender can overlap its later
first-hop message with an earlier message's propagation or downstream hop.

Plans keep one route descriptor per rank. Execution retains per-rank counters
and current predecessor events, plus actual in-flight messages; completed
round history is not retained. In-flight work can exceed one message per rank.
There is no hidden one-outstanding-message limit. A bounded diagnostic trace
is optional and uses the existing Engine trace limit.

## Current consumers

- `Topology.ring_plan` binds groups, rank order, physical routes, message
  payloads, frames and propagation delays.
- `Execution._context` supplies that plan lazily through
  `RecipeContext.collective_plan`; resource accounting uses the same topology.
- `RecipeBook.from_mapping` registers `RingCollective`. `operator_costs`,
  `graph_costs`, `target_graphs` followed by `graph_costs`, and ordinary
  `predict` all reach the same program and Engine.
- Independent TP/PCP/ETP groups retain their existing rank placement. Complete
  cross-DP ring groups bind one joint cohort plan through `Execution.add_group`;
  an incomplete local group that spans other DP replicas is still rejected
  rather than executed while silently omitting remote dependencies.

`RingCollective` rejects nonzero native work other than `link`. This preserves
the boundary: a future reduction-compute implementation must compose its actual
work and receive dependencies, rather than lose declared compute under a
transport recipe.

## Source absorption

The previous topology module is retained at revision
`d141b68616015aff8bbac59e0073417b65b5ebd5`:

- `mindie_sim/a3_comm_topology.py`, SHA-256
  `940a69d452e94f7c88217e20e69b5467365f2e40f635f1cc586c20187e652e36`.
- `mindie_sim/a3_transfer_mechanism.py`, SHA-256
  `d89ed9ac0422fb6aea7a5669c629ba894b2d5008b63516ead19a3f488c364f08`.

Its useful/wire distinction, placement and dependent-transfer concerns now have
separate generic mechanisms. This round schedule extends the earlier
message-level absorption documented in `historical-communication.md`; it does
not import empirical timing multipliers or claim all old protocol state is
represented.

The receive-forward dependency was checked against official NVIDIA NCCL source
at fixed tag `v2.26.6-1`, not an asserted latest release:

| Source path | Git blob | Reviewed scope |
| --- | --- | --- |
| `src/device/all_gather.h` | `854ebbf3aaf6d3a190696e412d8bc811167db424` | Ring initial send, receive-forward loop and final receive, lines 1-85 |
| `src/device/all_reduce.h` | `81da55401645bba1eb957dc6f30eadf23ec4e39f` | Ring reduce-scatter/gather receive-forward order, lines 1-112 |
| `src/device/reduce_scatter.h` | `5d8de28197b046b4de735eda372be925b51c1bd9` | Ring send, receive-reduce-forward and final receive, lines 1-82 |

Repository: `https://github.com/NVIDIA/nccl/tree/v2.26.6-1/src/device`.
These sources establish causality, not the simulator's bandwidth, compute
cost or complete NCCL protocol. The abstraction deliberately does not reproduce
channel dispatch, chunk-selection rules or compiled device instructions.

## Executed counterexamples

`tests/test_collective_rounds.py` uses independent send/receive recurrences and
public graph/serving entry points, rather than comparing two calls to the same
helper:

| Declaration | Expected result and distinguishing behavior |
| --- | --- |
| Four ranks, 1 B/message, 1 B/ms links, only edge 0 to 1 has 10 ms propagation | Allgather/reduce-scatter complete in 13 ms; allreduce in 26 ms. The old aggregate envelopes take 33/66 ms with identical 12/24 B physical work. |
| Same asymmetric ring trace | Rank 2 round 1 starts at 1 ms, before rank 0 round 0 arrives at 11 ms, proving the absence of a global barrier and a self-arrival gate. |
| Two allgathers on separate streams using those links | Shared capacity gives 16 ms and 24 B total; neither call creates private link capacity. |
| Two-rank allreduce, 34 B/rank, 16 B payload plus 4 B header, 1 ms propagation, 1000 B/ms links | Four 17 B messages consume 160 wire B and 2.08 ms. External readiness at 8 ms gives completion at 10.08 ms. |
| Two-hop explicit store-forward, different frame sizes and 0.2/0.3 ms hop delays | 1.16 ms, 160 B on each shared physical link, second-hop service waits for first-hop arrival. Default direct mode rejects this declaration. |
| Real frontend graph and `predict`, explicit family configuration | Both service equal link bytes; serving traces include real round tasks. |
| Repeated ring primitive, 100 versus 1000 rounds | Equal peak live task counts and zero remaining tasks/programs; completed rounds do not accumulate. |

The generic-family framing control also requires communication reports and
actual Engine service to agree on 160 B. These are analytic causal controls,
not hardware accuracy measurements.

## Remaining scope

Uniform message chunks are an explicit assumption. Per-message frame tails are
modeled; unequal tensor chunks, element-size partition tails, channel/slice
selection, local reduction compute, packet credits, cut-through and
topology-aware algorithm selection are not yet represented. Complete cross-DP
ring groups use the joint cohort plan described in `collective-cohorts.md`.
No new device measurements were taken, historical raw data was not changed,
and no measured residual is used to select or adjust this schedule.
