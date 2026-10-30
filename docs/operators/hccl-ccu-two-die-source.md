# Explicit two endpoint-die CCU_MS AllGather

The public model and serving paths accept `runtime.kernels.hccl.allgather`
with `expansion=CCU_MS`, `template=mesh_2die_broadcast_v1`,
`topology=TWO_DIE_REGULAR`, and a complete `channel_die_map`. The ordinary model
logits AllGather in [the TP8 control case](../../examples/communication/ccu-two-die-allgather-tp8.json)
uses this binding through `target_graphs`, `graph_costs` and `predict`.
No hardware SKU selects the template or proves that a device supports CCU.

`channel_die_map[source][peer]` is the source channel's endpoint die, 0 or 1.
Source and peer index the selected group's logical positions; physical rank IDs
come from the machine's device order and may be noncontiguous.
The diagonal is `null`. These identifiers are two CCU owners on each rank,
independent of NPU rank IDs, compute-die packages and SIO/HCCS link placement.
Physical messages still follow the declared `machine.links` graph. The default
single-die CCU_SCHED mem2mem AllGather remains a different implementation.

## Source and executable domain

The implementation follows upstream HCCL revision
[`c2d6790464654ccf58aca39448de4b65fdc7b184`](https://github.com/hicann/hccl/tree/c2d6790464654ccf58aca39448de4b65fdc7b184).
[The source receipt](../../data/verification/hccl-ccu-two-die-20261030.sources.json)
records inspected file hashes. Relevant boundaries are:

- `topo_host.cc:1178–1240`: `TWO_DIE_REGULAR` requires nonempty endpoint-die
  channel counts differing by one. It does not group NPU ranks into SIO pairs.
- `ins_v2_all_gather_sole_executor.cc:497–503`: the broadcast implementation is
  `CcuMSAllGatherSoleMesh2Die`, with one-level Mesh1D matching, regular endpoint
  topology and separate input/output. The registration is excluded under the
  source's `HCCL_CANN_COMPAT_850` compile guard.
- `ccu_temp_all_gather_2dies_mesh_1D.cc:55–139,143–199`: classify channels by
  endpoint die; assign self-copy to the smaller peer group; launch two kernels
  after inter-thread synchronization and join them afterward.
- `ccu_kernel_all_gather_2dies_mesh_1D.cc:82–146`: exchange output addresses and
  tokens, run `GroupBroadcast` or `GroupBroadcastWithoutMyRank`, then notify
  peers and wait for completion.
- `ccu_kernel_alg_base.cc:465–510,650–750`: each owner loads the common input
  into its own CCU buffer, waits, writes its peers, optionally copies self, and
  waits for those operations. The header's default is CCU V1.

The current domain is one equal-payload executor repeat, one fragment of at
most 4096 bytes, an even rank count from 4 through 16, and separate input/output.
Each rank must have the same outgoing and incoming peer count for each endpoint
die. The source's regular classification alone does not imply this stronger
condition: a map can remain regular while swapping endpoint degrees between
ranks. Such a map is rejected because the present engine models one local
representative shard; it cannot faithfully price per-rank CCU owner skew.
The cyclic explicit TP8/TP16 maps meet the supported condition. Noncontiguous
physical rank IDs retain the same logical endpoint map and work.

## Physical work and dependencies

For R ranks and N input bytes per rank, the two owners share an input address
but each loads N bytes into its own buffer. The smaller peer group alone copies
N bytes to the local output. Every peer sends N bytes to that output. Local work
therefore remains:

| Work | Amount |
| --- | ---: |
| HBM reads and writes | `(R+2)*N` bytes |
| CCU LocalCopy | `3*N` bytes |
| Logical CCU SRAM accesses | `(R+2)*N` bytes |
| CCU notification/event issue | `5*(R-1)+4` calls |
| Inter-thread controller issue | 4 calls |
| Full-group useful payload | `R*(R-1)*N` bytes |
| Full-group remote notifications | `3*R*(R-1)` messages |

No HBM scratch or reduction is introduced. Each owner has one active input
fragment. The source allocates 128×8 CCU buffer handles per owner; these handles
are reported separately from active payload, NPU UB and physical SRAM capacity.

The hardware declaration must supply `ccu_copy`, `ccu_buffer_bytes` and
`ccu_control`, each with exactly two `resource_units`. Work is pinned to the
endpoint owner; a finished owner's unused capacity cannot accelerate the other
owner. The two owners share the representative rank's HBM supply. Optional
`hccl_control` prices the outer inter-thread notification issue. Without it,
that issue remains explicitly unpriced in the local phase's metadata.

PreSync issue precedes separate output-address and token notifications. Both
notifications retain their own transaction frames. Their actual route arrivals
precede local NotifyWait issue. The data broadcast follows input copy and its
event wait. PostSync again uses record issue, a real notification route and
local wait issue. A wait never adds another network round trip. An unspecified
`signal_bytes` retains propagation with unknown framing; an explicit value is
framed once per notification by the normal Link model.

The caller keeps whole-group notification and peer-phase barriers. It does not
simulate submessage progress or overlap across multiple CCU fragments. One
outer collective invocation contains two CCU source branches; their two native
kernel launches and address setup are inventoried but not separately priced.
The source-compatible declaration is not a captured dispatch, installed binary
identity, A3/A5 capability claim or measured communication-accuracy result.

## Verification and reproduction

`tests/test_hccl_ccu_two_die.py` checks 4/8/16-rank work equations, shared-input
reads, one self-copy, two pinned owners, payload/notification framing, input
readiness, notification propagation without repeated RTT, independent capacity
scans, noncontiguous physical rank order, ordinary model and serving calls, and
the boundary excluding skewed maps and larger fragments. The combined affected
CCU/HCCL run passed 78 tests.

[The public caller scan](../../data/verification/hccl-ccu-two-die-20261030.json)
contains 36 TP8/TP16 evaluations at 0.5×/1×/2× copy, SRAM, control, HBM, port and
switch capacities. It keeps physical wire bytes separate from simultaneous
Tx/Rx/switch capacity service. Capacities that are not limiting can plateau;
separate unit controls make each service limiting in turn. All rates in this
case are synthetic analytical declarations, without selector heuristics or
measurement fitting.

```sh
PYTHONPATH=. python3 data/verification/hccl-ccu-two-die-20261030.producer.py
python3 -m mindie_sim predict examples/communication/ccu-two-die-allgather-tp8.json
```
