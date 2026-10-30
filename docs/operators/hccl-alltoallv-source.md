# Explicit single-channel HCCL AlltoAllV source model

`hccl_alltoallv.contract(selection, send_matrix, rank)` now describes one
AICPU Mesh1D rank's local work and finite peer streams. `AlltoAllVGroup` executes
all ranks on the existing Engine and physical Topology. This is an opt-in
contract API with actual Engine controls. It is **not installed by the ordinary
frontend** and does not claim to identify a captured HCCL dispatcher.

The declaration uses `expansion="AICPU_TS"`, `collective="alltoallv"`,
`template="mesh_read"` or `"mesh_write"`, `buffer_mode="separate"`, and one
single-level `MESH_1D`. Read denotes the explicitly selected PCIe Read branch;
write denotes the explicitly selected non-PCIe Write branch. The implementation
covers one channel per peer. Multi-channel port-group ratios, symmetric remote
memory, subcommunicator reorderings, and executor chunking are outside this
selection. No SKU, model, measured duration, or presumed topology selects it.

The matrix contains exact **bytes**, including self-send cells. Every cell must
be a whole number of `dtype_bytes` elements. `selection.payload_bytes` is that
rank's complete row sum, including the diagonal. No row histogram, capacity,
client concurrency, or probability estimate is silently converted into traffic.

## Source and dependencies

Pinned repository revision:
[`hicann/hccl c2d6790464654ccf58aca39448de4b65fdc7b184`](https://github.com/hicann/hccl/tree/c2d6790464654ccf58aca39448de4b65fdc7b184).
The retained files and SHA-256 values are in
`data/verification/hccl-alltoallv-source-20261030.json`.

- `ins_temp_all_to_all_v_mesh_1D.h:21`: maximum 16 concurrent peers.
- `ins_temp_all_to_all_v_mesh_1D.cc:142–197`: left/right peer groups and scratch
  slot indices. Later peers reuse only their slot; there is no global round
  completion barrier.
- `...cc:259–305`: one initial and one final thread synchronization. In separate
  Read mode, the first group performs its pre-copies and an additional join and
  release before starting transfers. The self-copy runs on the main thread.
- `...cc:364–427`: later Read pre-copy on its slot; Write post-copy after receive;
  multi-channel per-peer joins are not part of the single-channel selection.
- `...cc:430–473`: zero/zero peers issue no transfer or channel notifications;
  one-way peers use Send or Recv; two-way peers use SendRecv.
- `...cc:583–616`: local pre-copy reads user input and writes scratch. Post-copy
  reads scratch and writes user output.
- `alg_data_trans_wrapper.cc:366–475,596–700`: ACK and DATA_SIGNAL notification
  dependencies and direct peer read/write payload. Notification counts are
  physical control demand, not measured latency constants.

For off-diagonal outgoing bytes `S`, incoming bytes `R`, and diagonal `D`:

| Local service | Read | Write |
| --- | ---: | ---: |
| HBM bytes | `3*S + R + 2*D` | `S + 3*R + 2*D` |
| local copy bytes | `S + D` | `R + D` |
| useful outgoing network bytes | `S` | `S` |

Every cohort therefore has `4*off_diagonal_bytes + 2*diagonal_bytes` HBM demand
in either mode. Each directed message is routed and framed **once per hop**;
receiver plans do not charge a second copy of the link work. HBM endpoint work
and wire service overlap as a whole-peer phase envelope. Sub-message progress
is not modeled. Zero peers do not create a remote-ready barrier; a receive-only
rank still participates in the sender's ACK and arrival dependencies.

`scratch_bytes` is a sufficient minimum using 16-or-fewer common-stride slots
and this rank's largest relevant peer cell. The actual executor can reserve a
larger stride or segment messages. This field is not an observed allocation and
is not added to weight/cache capacity automatically.

## Execution contract

```python
from mindie_sim.prediction.hccl_contracts import select
from mindie_sim.prediction.hccl_alltoallv import AlltoAllVGroup

matrix = ((8, 16), (0, 4))
selection = select(
    {"expansion": "AICPU_TS", "template": "mesh_read", "dtype_bytes": 2},
    collective="alltoallv", ranks=2, payload_bytes=sum(matrix[0]),
)
group = AlltoAllVGroup(selection, matrix, physical_ranks, topology)
done_by_rank = group.start_participants(
    engine, "call", ready_by_rank, contexts_by_rank,
)
```

Each context binds `hbm` and may bind `hccl_copy` or `hccl_control` to explicitly
declared hardware services. Absent copy/control services remain visible under
`unpriced_service_work`; HBM demand still executes. No AIV, Cube, rate, launch
floor, per-peer startup, or `EP-1` serial link estimate is invented. The eventual
ordinary caller must supply its outer invocation boundary once.

The sparse contract stores O(R) peer descriptors per rank. All-rank execution
uses O(R²) plan/events, independent of token iterations. It never constructs one
dense R×R exchange matrix per peer and does not introduce another scheduler.

## Verification

`tests/test_hccl_alltoallv.py` covers independent byte equations, left/right slot
coverage through 65 ranks, real 2/8/17/18/33/34-rank execution, one-way and zero
peers, a late receiver, local diagonal copies, distinct Read/Write copy order,
slot-specific backpressure, transposed returns, noncontiguous physical rank IDs,
multi-hop framing, and separate bandwidth/copy/control changes. The 34-rank
uniform control finishes in three peer waves rather than 33 serial starts.

The new and existing HCCL/direct-peer tests passed **51/51 in 4.23 s**. The
compact report records synthetic resource controls, not hardware accuracy.
