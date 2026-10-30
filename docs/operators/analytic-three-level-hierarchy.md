# Explicit three-level collective composition

`runtime.kernels.analytic_hierarchy.allreduce` composes six phases on the existing
Engine: ReduceScatter over dimensions 0, 1 and 2, then AllGather over dimensions
2, 1 and 0. It is an analytic model, selected only by this explicit configuration.
It does not claim an HCCL executor, expansion mode or automatic selector result.
Existing `runtime.kernels.hccl` source contracts and defaults are unchanged.

```json
{
  "analytic_hierarchy": {
    "allreduce": {
      "axes": ["tp"],
      "rank_dimensions": [2, 2, 2],
      "levels": [
        {"reduce_scatter": "mesh", "allgather": "mesh", "protocol": "ack_data"},
        {"reduce_scatter": "mesh", "allgather": "mesh", "protocol": "ack_data"},
        {"reduce_scatter": "mesh", "allgather": "mesh", "protocol": "data_signal"}
      ],
      "signal_bytes": 4
    }
  }
}
```

The current domain is three nontrivial integer rank dimensions, whose product
equals the collective group size, and an input divisible into equal integer
element shards across that group. Each level must explicitly choose the supported
`mesh` primitives and notification protocol. `rank_order`, when supplied, is a
permutation of positions in the physical communicator. Dimension 0 varies fastest;
varying one dimension while fixing the other coordinates defines an actual group.
The ordinary caller supports uniform `tp`, `etp` and `pcp` axes. Ragged shards,
cross-DP caller expansion and native three-level selector inference are outside
this initial domain.

For input bytes N and dimensions `(a,b,c)`, RS inputs are `N, N/a, N/(ab)`.
AG inputs are `N/(abc), N/(ab), N/a`. A direct RS sends each destination its
output shard, retaining a local copy and reducing the received scratch buffers.
AG copies its resident shard and receives the other shards into their output slots.
These declared separate buffers preserve all intermediate HBM reads and writes;
they do not assume unproven alias optimizations. Across all ranks the payload
wire sum is exactly `2*N*(R-1)`. Per-rank SUM work is `N*(R-1)/R` bytes.

`ack_data` declares reverse ACK events before payload and forward completion
events after payload; `data_signal` omits the ACK. Each signal routes through
the same topology as data. A local wait only depends on the remote arrival;
it adds no separate round-trip latency. Omitting `signal_bytes` retains remote
propagation while leaving unknown frame traffic uncounted. The example's 4-byte
frame is a synthetic caller declaration, not an HCCS/UB framing specification.

Optional `hccl_copy`, `hccl_reduce` and `hccl_control` rates price the associated
byte/operation ledgers. Without those declarations, HBM and network simulation
still run and omitted service work remains visible. No measurements are required.
Intermediate groups use explicit whole-phase barriers; this version does not
invent inter-level chunk overlap. Scratch is a capacity ledger, not a physical
pool reservation. No aggregate observed response is attached per phase or level.

The graph carries `analytic_hierarchy_group`, `selection_basis` and analytic
coverage metadata, with no native kernel kind or source revision claim. The
whole collective has one logical invocation. Network accounting distinguishes
physical wire bytes from separate TX and RX capacity service: those two capacity
counters must not be added and reported as two wire transmissions.

## Verification

`tests/test_hccl_analytic_hierarchy.py` uses an independent contribution-set oracle
for `(2,2,2)`, `(2,3,2)` and `(3,2,4)`, including a non-contiguous physical rank
permutation. Each final output shard contains every input rank exactly once.
The tests execute the ordinary frontend and Engine, check all three level
bandwidth scans, HBM/control-rate scans, receiver incast, one late physical rank,
remote frame accounting and task release. Existing native hierarchy tests remain
part of the affected verification scope.

The public example is `examples/communication/analytic-three-level-control.json`:

```sh
python -m mindie_sim graph examples/communication/analytic-three-level-control.json \
  --phase decode --tokens 1 --context 16 --scope body --skip-isolated
```

`data/verification/analytic-hierarchy-three-level-20261030.producer.py` retains
compact public-API scan results and input/runtime hashes. All its rates are
synthetic controls; these checks establish composition and sensitivity, not
hardware prediction accuracy.

The retained run completed 15 scan evaluations in 0.95 seconds (38.4 MiB peak
RSS on this host). The complete example's public CLI graph took 0.225524416 ms
of simulated time, with zero live tasks/programs. The affected suite passed
136 tests in 6.71 seconds. For the 8-rank, N=128-byte control, each call declares
1,792 payload wire bytes, 96 notification frames, and 1,232 bytes of local HBM
work on the representative rank. Three-level bandwidth, HBM and controller
rate scans retain exactly the same modeled work.
