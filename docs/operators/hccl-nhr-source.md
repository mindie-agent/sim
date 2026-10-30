# AICPU NHR source program

Development-run reports cited below are historical context and are not shipped. Current acceptance is documented in [verification](../verification.md).

Explicit `AICPU_TS` / `template: nhr` implements AllReduce, AllGather and
ReduceScatter with a single channel per peer, separate user input / scratch /
output buffers, non-PCIe transport, one topology level, power-of-two ranks and
equal integer shards. Topology may be `MESH_1D` or `MESH_1D_CLOS`; this selects
the source schedule, while the ordinary topology still supplies physical links.
No model, SKU, observed latency or automatic dispatch heuristic selects it.

```json
{"runtime":{"kernels":{"hccl":{"allreduce":{
  "expansion":"AICPU_TS","template":"nhr","axes":["tp"]
}}}}}
```

The source is [HCCL c2d67904](https://github.com/hicann/hccl/tree/c2d6790464654ccf58aca39448de4b65fdc7b184).
Its algorithm ranks follow the communicator list. ReduceScatter sends to
`rank - 2**step` modulo R, and AllGather reverses the distances and direction.
These are cyclic neighbors, not XOR partners. Each round carries an exact
directed matrix; only its actual peer has payload. Later rounds wait for both
the local endpoint work and network completion, preserving cross-round reuse.

The caller composes these phases on the same Engine as other operations.
Network phases own link bytes only. An ordinary exchange charges source HBM
read plus destination write; a remote WriteReduce charges source read plus
destination read/write and a declared optional `hccl_reduce` service.
ACK / DATA notifications and waits remain separate readiness phases with
optional `hccl_control` work. Absent service rates remain visible as unpriced
work without disabling physical HBM/link simulation.

The current source has two optimizations that a textbook NHR formula misses:

- The first RS round sends its transmit shards from user input, so PreCopy
  copies only the remaining half of the input into scratch.
- The final AG round reads directly into user output. The final local copy
  excludes those received shards. Standalone AllGather uses a second thread
  to overlap that local copy with the final network round; AllReduce finishes
  its single-channel AG before its final local copy.

For R > 1 and equal shards, these yield the following per-rank work. N is the
complete local input for AllReduce/ReduceScatter and the one-rank contribution
for AllGather. These are physical ledgers, not measured performance formulas.

| Collective | Sent bytes | HBM bytes | Local copy bytes | Remote reduction bytes |
|---|---:|---:|---:|---:|
| AllReduce | `2N(R-1)/R` | `7N-5N/R` | `N` | `N(R-1)/R` |
| ReduceScatter | `N(R-1)/R` | `4N-N/R` | `N/2+N/R` | `N(R-1)/R` |
| AllGather | `N(R-1)` | `3RN` | `N+RN/2` | `0` |

Source locations at the pinned revision:

- `ins_temp_all_reduce_nhr.cc`:219–299 selects initial/final buffer paths;
  308–345 skips first transmit PreCopy;349–435 executes RS;438–585 executes
  AG and final direct output reads;588–636 performs the final local copy;
  640–729 constructs cyclic round schedules.
- `ins_temp_all_gather_nhr.cc`:94–104 reserves two threads per channel;
  166–206 selects the output path;309–350 overlaps final copy and read;
  422–453 constructs rounds;456–559 accounts initial/final local copies.
- `ins_temp_reduce_scatter_nhr.cc`:154–243 selects direct first-round TX;
  249–329 handles local copies;338–438 executes remote reduction;
  447–499 constructs the same cyclic RS schedule.
- `alg_data_trans_wrapper.cc`:245–270 and317–338 order ACK, batch transfer,
  DATA notify and wait.481–513 submits WriteReduce descriptors.

Non-power-of-two ranks, unequal tail shards, symmetric/aliased buffers, PCIe
Read paths, multiple channels, multi-level decomposition and executor message
segmentation are outside this implemented source contract. They do not silently
become fractional shards or generic mesh timing. The explicit source request
reports unsupported; the generic analytic model remains independently available.
Scratch demand is reported but its allocation is not yet scheduled. No empirical
4µs kernel,1µs task or cost-selector floor from HCCL is used as a hardware rate.

Verification includes
integer contribution propagation, exact directed bytes, true Engine readiness,
rate response and an AllGather overlap counterexample. These verify execution
semantics and physical accounting, not the selected path in a captured binary.
