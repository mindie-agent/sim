# Historical communication mechanism absorption

This scope restores message-level useful/wire accounting, physical link sharing
and rank-group ownership in the ordinary prediction path. It does not establish
collective timing accuracy or reconstruct a vendor collective algorithm.
Analytic simulation requires no communication measurement.

## Source provenance and limited absorption

The preserved implementation revision is
`d141b68616015aff8bbac59e0073417b65b5ebd5`. It is inspected with `git show`;
the old runtime is not executed or imported.

| Source at that revision | Content identity | Absorbed scope | Current consumer |
| --- | --- | --- | --- |
| `mindie_sim/a3_comm_topology.py`, `alltoall_plan_over_links`, lines 272-378 | SHA-256 `940a69d452e94f7c88217e20e69b5467365f2e40f635f1cc586c20187e652e36`; Git blob `2d061c4ed5be1cc7b87b7e01ba0a8492cbcd824c` | Each remote peer contributes a distinct message; useful and rounded wire volumes differ; shared-pipe demand includes every participating sender; dedicated pair capacity differs from shared-port capacity. | `prediction/topology.py`, `prediction/communication.py` |
| `mindie_sim/a3_transfer_mechanism.py`, `wire_bytes`, lines 176-193 | SHA-256 `d89ed9ac0422fb6aea7a5669c629ba894b2d5008b63516ead19a3f488c364f08`; Git blob `2fef8d64969cc1d982ddc4f565e907a7b8818c70` | Message alignment, transaction rounding, per-transaction protocol overhead, and separate useful/wire counters. | `MessageFlow.on_link`, `schema.hardware.Link` |

The old all-to-all helper takes the full pre-exchange payload, divides it with
integer floor, and retains a local share. The current frontend already produces
the useful *remote* traffic per sending rank. Reapplying the old local-share
factor would undercharge it. The new path splits that existing remote total
among remote peers. It does not copy the old floor/remainder behavior.

Other functions in these modules are not absorbed by this scoped mapping.
In particular, the old `price_collectives` explicit lane/slot chains,
`TransferStream` setup-scope formulas, and `identify_path` fitting method are
not claimed here. Raw communication calibration artifacts remain unchanged;
none trains this byte-accounting mechanism.

## Open-source mechanism reference

The inspected primary source is LLMCompass at fixed revision
`2e015fd2ee750e6cad8d7152df52551e5b41ef20`:

- `software_model/communication_primitives.py`, Git blob
  `1229eb4665a4e13c95a7b3c3cebb00e3598d160f`, especially
  `AllReduceMultiPCB.simulate`.
- `hardware_model/interconnect.py`, Git blob
  `dd541e9edbbb8d2ebb262bccf903e0903f176da7`, especially `LinkModule`.

Both files were read from the repository's GitHub connector. The collective
model calculates chunk payload and header-bearing physical traffic separately,
and distinguishes per-direction from bidirectional bandwidth. These mechanisms
motivate retaining message boundaries and explicit physical capacity semantics.
The current implementation does not import LLMCompass's fixed hardware rates,
its particular FC/ring timing formula, or its assumption about internal-link
bandwidth. No LLMCompass run or accuracy reproduction was performed.

The source can be inspected at these fixed paths:

- `https://github.com/PrincetonUniversity/LLMCompass/blob/2e015fd2ee750e6cad8d7152df52551e5b41ef20/software_model/communication_primitives.py`
- `https://github.com/PrincetonUniversity/LLMCompass/blob/2e015fd2ee750e6cad8d7152df52551e5b41ef20/hardware_model/interconnect.py`

## Physical contract

An optional link declaration supplies:

```json
{
  "name": "declared-fabric",
  "members": [0, 1, 2, 3],
  "capacity_bytes_per_ms": 1000,
  "semantics": "per_pair",
  "transaction_bytes": 16,
  "alignment_bytes": 1,
  "protocol_overhead_bytes": 4
}
```

For useful message payload `P`, alignment `A`, transaction payload `G`, framing
overhead `H`, and repeated-message count `R`:

```text
aligned_payload = ceil(P / A) * A
transactions = ceil(aligned_payload / G) * R
wire_bytes = transactions * (G + H)
useful_bytes = P * R
```

A large message uses multiple transactions. Every final transaction is padded
under this declared contract; `transaction_bytes` is not merely a maximum packet
payload with an unpadded tail. Alignment applies once to each logical message,
and the header applies to every transaction. These parameters are declarations,
not inferred NIC facts or fitted utilization.

Without `transaction_bytes`, the existing continuous-byte model remains:
wire bytes equal useful bytes and transaction count is unknown (`null`).
No hidden one-byte rounding is applied to expected/fractional routing volumes.
Declaring alignment or framing requires an explicit transaction payload size.

Each hop frames the original useful message using that hop's contract.
Headers and padding from an upstream link do not become downstream tensor
payload. Routed useful bytes and total wire bytes count every hop; endpoint
useful bytes count each message once.

Capacity semantics are:

- `shared`: every flow through the link consumes one common aggregate capacity.
- `per_port`: every flow leaving a rank shares that source port's capacity.
- `per_pair`: each ordered source/destination pair has its own capacity;
  opposite directions are independent, and repeated flows on the same directed
  pair contend. Capacity is bytes/ms **per direction per pair**. This declaration
  does not assume an additional shared source-NIC bottleneck.

Existing `mesh` and `fully_connected` declarations retain their previous
per-port interpretation. They are not silently upgraded to independent pair
wires. `direct` retains its existing shared-resource interpretation. A case
requiring independent pair wires declares `per_pair` explicitly.

## Collective and ownership boundary

`OpSpec.link_bytes` is the already-lowered total useful remote traffic per
sending rank. `Topology.messages` retains these current algorithm assumptions:

- All-to-all/all-to-all-v uses one equal-payload message to each remote peer.
  The `alltoallv` family does not yet supply an uneven peer-size matrix.
- A ring allreduce splits per-rank traffic over `2 * (group_size - 1)`
  transmissions. Other existing ring collectives use `group_size - 1`.
  Explicit `steps` overrides that declared round count.
- PP transfers send one message between corresponding source/destination
  stage ranks.

In the default resource envelope the round count is used for message framing
and the existing path-latency estimate, without executable per-round
dependencies. The opt-in `ring_collective` now executes rank-specific
receive-forward dependencies on the same Engine; see `collective-rounds.md`.
Its direct or explicitly declared whole-message store-forward paths preserve
service/propagation overlap. Packet-level multi-hop fill/drain, finite peer
credits and cross-DP joint ring schedules remain separate refinement work.

Groups follow declared physical placement. ETP groups are constructed inside
each TP slice, independently for every PCP copy. EP groups spanning DP collect
the corresponding PCP slice from each replica in that EP cohort. PP stages and
different EP cohorts remain separate. Each participant invocation charges only
its own sending ranks; joining a complete cross-DP cohort therefore charges
every sender exactly once.

`Topology.accounting` supplies physical wire work and declared frame geometry
to `Execution._context`; `Topology.demand` retains the compact demand interface.
The ordinary `operators`, `graph` and `predict` paths therefore share this
mechanism. Each isolated operator row additionally contains `communication`,
with sending-rank useful bytes, endpoint useful total, routed useful total,
wire total and transaction counts. Its boundary is
`sending_ranks_in_replica`; it is not necessarily one physical rank or the
entire cross-DP collective.

The existing `execution.work.link_bytes` keeps its frontend/representative
operator accounting. `execution.serviced["link:..."]` is physical service
across routed resources. Those totals have different boundaries and must not
be divided into a communication efficiency without accounting for senders,
hops, transactions and padding. An explicit transfer recipe may add a separate
finite-credit contract. It consumes the link's existing wire frames; it does
not round or add headers again. The optional recipe `transaction_bytes`
denotes payload granularity and must agree with a link's explicit declaration.
Omitting it uses the link frame geometry. A conflicting declaration raises a
modeling error instead of silently changing physical bytes. Unframed resources
continue to use the transfer recipe's own transaction contract.

## Executed counterexamples

On 2026-10-29:

```sh
.venv/bin/python -m pytest -q tests/test_communication_topology.py \
  tests/test_parallel_mechanisms.py tests/test_prediction_system.py
```

Result: **26 passed in 0.14s**, including nine new communication tests.

- Four all-to-all ranks, 30 useful remote bytes/rank, 16-byte transactions and
  four-byte headers: 12 independent messages require **240 wire bytes**.
  Rounding each rank's aggregate would incorrectly charge 160 bytes.
- Four-rank allreduce, 42 useful bytes/rank: six seven-byte transmissions/rank
  require **480 wire bytes** under the same framing. Aggregating before
  rounding would charge 240 bytes.
- Unframed 120-byte group traffic at 1000 B/ms requires **0.12 ms shared**,
  **0.03 ms per source port**, or **0.01 ms per directed pair**. Concurrent
  transfers sharing the same pair double its completion time; a disjoint pair
  completes independently.
- A two-hop 17-byte PP message charges 40 wire bytes on each differently
  framed hop, preserving 17 endpoint useful bytes and 34 routed useful bytes.
- ETP and cross-DP EP controls independently enumerate expected PCP, PP and
  cohort rank sets and count every source once.
- A real frontend model graph and its serving execution charge the same
  physical communication work; the graph's isolated rows account for it.

These are arithmetic, ownership, resource-contention and public-path checks.
They are not measurements of an NPU/GPU collective, and do not complete the
remaining historical communication mechanism or observation migration.

Cross-lane review found and repaired an integration counterexample: two
one-byte messages framed as 16 payload bytes plus four header bytes must
occupy 40 wire bytes. Applying a 16-byte transfer granule to that already
framed aggregate had charged 48 bytes. The finite-credit recipe now transmits
two 20-byte frames. One outstanding credit and a one-ms response delay yield
`2 * (20 / 1000 + 1) = 2.04 ms`, with exactly 40 bytes serviced.

The integrated run was:

```sh
.venv/bin/python -m pytest -q tests/test_communication_topology.py \
  tests/test_transfer_program.py tests/test_graph_analysis.py
```

Result: **30 passed in 0.31s**. This includes the single-framing-owner
counterexample and the explicit conflict case; the finite-transfer mechanism
is still distinct from collective round rendezvous or peer-slot scheduling.
