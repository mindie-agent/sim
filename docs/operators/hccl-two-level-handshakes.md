# Two-level HCCL and remote notification boundaries

`parallel_mesh_nhr_route0` composes a source-compatible, explicitly selected
AICPU AllReduce path on the ordinary frontend and common Engine. It does not
select an executor from a device name or claim a captured runtime selected it.

```json
{"hccl":{"allreduce":{
  "expansion":"AICPU_TS",
  "template":"parallel_mesh_nhr_route0",
  "topology":"MESH_1D",
  "topology_levels":2,
  "level0_size":2,
  "max_loop_bytes":1048576
}}}
```

`level0_size=L` declares consecutive rank groups; level1 groups contain ranks
with the same position in each level0 group. The first domain is one equal-shard
loop, `R/L` a power of two, SUM with 1/2/4-byte elements, distinct user buffers,
and an explicit 100% mesh-first split. `max_loop_bytes` declares the largest
payload known to fit that source loop; it is not inferred from hardware bandwidth.
Uneven tail shards, the other parallel lane, multiple loop segmentation,
MultiJetty, PCIe, three levels, CCU two-die and DPU executors remain unsupported
by this template. Existing single-level templates retain their existing behavior.

## Source-derived stages and aliases

Pinned repository: [HCCL c2d67904](https://github.com/hicann/hccl/tree/c2d6790464654ccf58aca39448de4b65fdc7b184).
The local source used is `/Volumes/data-tmp/tmp/sim-hccl-source-20261030`.

- `src/ops/all_reduce/algorithm/executor/ins_v2_all_reduce_parallel_executor.cc`:
  `GetParallelDataSplit` at 630–645 preserves an explicitly supplied ratio;
  `GenInsQues` at 1035–1244 runs four stage barriers. With ratio 1, the executed
  lane is level0 Mesh RS → level1 NHR RS → level1 NHR AG → level0 Mesh AG.
  The other lane remains empty. `PrepareResForTemplate` / `PrepareResForTemplate23`
  at 668–740 retain two template-main threads and their local barriers.
- `RunTemplateInter0/Inter01` at 1311–1343 and 1387–1420 uses matching internal
  HCCL-buffer offsets. NHR's internal RS and AG consequently need no pre/post
  copy. Internal AG cannot take its user-output last-read optimization.
- `src/ops/reduce_scatter/algorithm/template/aicpu/ins_temp_reduce_scatter_nhr.cc`
  at 220–304 contains these copy alias checks;
  `src/ops/all_gather/algorithm/template/aicpu/ins_temp_all_gather_nhr.cc`
  at 456–546 skips internal input and output copies.
- Final Mesh AG reads peer scratch directly to user output, with only the local
  shard copied: `ins_temp_all_gather_mesh_1D.cc:156–210,262–310`.
- The reserved scratch upper bound follows `all_reduce_parallel_scratch.h:20–25`.
  This report exposes the reservation; the current collective frontend still
  does not reserve a separate global HCCL scratch pool in the Engine.

## A wait is not an RTT

`src/ops/op_common/algorithm/template/wrapper/alg_data_trans_wrapper.cc`:

- `DoSendRecvBatchTx:253–285` and `SendRecvWrite:424–461` explicitly record a
  remote ACK, wait for the matching peer ACK, transfer data, record DATA_SIGNAL
  and wait for the incoming completion. The modeled remote notification is one
  directed event, not a round trip per record *and* per wait.
- `SendRecvBatchWriteReduce:481–516` enables notify fusion; the last valid
  WRITE_REDUCE becomes WRITE_REDUCE_WITH_NOTIFY (`106–131,158–176`). The fallback
  `RunWriteReduceAndNotify:202–249` does the same. NHR RS completion therefore
  waits for the existing payload arrival; it adds no second propagation event.
- Batch read has reversed ACK/completion direction. Final Mesh AG retains the
  actual read protocol; NHR's internal AG uses its ordinary write protocol.
- Executor PreSync/PostSync and template thread synchronization remain local
  `hccl_control` operations and graph dependencies. They never acquire link RTT.

A remote signal is routed with `Topology.route_message`, including zero-byte
signals. Each hop uses coupled Tx/Rx/switch capacities and one propagation event.
Unknown control frame bytes remain zero; they are not replaced with invented
one-byte packets. Optional `signal_bytes` declares known per-notification bytes;
ordinary per-message link framing then applies once at each hop. Signal payload
bytes, physical wire bytes and simultaneous capacity-byte demand remain distinct.
No HCCS, UB or RoCE frame sizes, controller rates, or delay constants were added.

The plan reports controller operation counts, and prices them only when the
hardware declares `hccl_control`. Remote propagation and payload service run
without that declaration. No measured body residual sets any parameter.

## Executable evidence

`examples/source-families/hccl-two-level-control.json` is a synthetic eight-rank,
two-layer ordinary frontend control with L=2 and H=4. It creates four AllReduce
calls; all local phases and remote notifications remain inside each one native
invocation. The retained producer/report is
`data/verification/hccl-two-level-20261030.{producer.py,json}`.

For one N=64-byte call:

| Quantity | Result |
|---|---:|
| Directed payload wire before framing | 896 B = 2(R−1)N |
| Explicit remote notification edges | 80 |
| Representative rank HBM work | 472 B |
| Copy / reduce bytes | 64 / 56 B |
| Controller operations | 62 |
| Payload propagation rounds | 6 |
| Independent notification propagation rounds | 10 |

Thus a uniform 5 µs per-hop latency adds 16×5 µs per call, not a latency for each
local wait and not another completion latency after a fused reduce-write.
For all four calls, 0.5×/1×/2× declared level1 bandwidth produced
0.488513/0.296513/0.200513 ms, with identical serviced work. An illustrative
10 million controller operations/s changed 0.296513 to 0.321313 ms; it is a
sensitivity control, not a measured device parameter.

Tests use actual Engine executions for late receiver readiness, source stage
ordering, exact wire/HBM totals, local-vs-remote control counts, framed signals,
link bandwidth scaling, two-hop receiver incast, and task/program release.
They also retain existing Mesh/NHR/AIV controls. These results establish source
composition and physical responses, not whole-model performance accuracy.

## Topology names are not hardware equivalence claims

The pinned `ins_v2_allreduce_sequence2die_executor.cc:344–367` registers
`CcuMSAllReduceSequenceMesh2Die` and its CCU-SCHED equivalent as
`TopoMatchOneLevel`, `maxTopoLevelNum=1`, `MESH_TYPE_TWO_DIE_REGULAR`.
That is not evidence that an arbitrary A3 two-die system uses CCU, nor that its
source has two HCCL hierarchy levels. Its die-specific kernel path needs its own
contract before support is claimed.

Likewise `DpuAllReduceSequenceMeshNHR` is a different source path:
`ins_v2_all_reduce_sequence_executor.cc:270–495,565–567` selects local Mesh RS →
inter DPU Mesh RS → inter DPU NHR AG → local Mesh AG. The new AICPU template does
not relabel itself as that DPU executor or assume DPU's fence/control services.
