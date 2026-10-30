# HCCL hierarchy and joint service boundary audit

The current two-level `parallel_mesh_nhr_route0` implementation remains a narrow
source-selected path. An explicit three-level analytic composition can use the
same Engine, physical message routing, receiver capacity and switch links. It
must have a different declaration from a native HCCL selector and cannot inherit
its synchronization counts merely because the stage names resemble it.

## Verified source scope

Reviewed HCCL revision `c2d6790464654ccf58aca39448de4b65fdc7b184` in
`/Volumes/data-tmp/tmp/sim-hccl-source-20261030`:

| Source path under `src/ops/all_reduce/algorithm/executor/` | Supported source fact |
|---|---|
| `ins_v2_all_reduce_parallel_executor.cc:1035–1244,1526` | Parallel Mesh/NHR has two payload routes with distinct stage order; the implemented route0 retains only the declared mesh-first lane. |
| `ins_v2_all_reduce_sequence_executor_aicpu_3level.cc:791–847,857–865` | Real three-level sequence: RS L0, RS L1, RS L2, AG L2, AG L1, AG L0. L0 uses Z-axis detour templates; L1 can be skipped under its own source conditions. Tail slices, scratch capacity and symmetric-memory aliases change work. |
| `ins_v2_all_reduce_omnipipe_executor.cc:1210–1245` | Separate two-level and three-level OmniPipe registrations. Three-level AICPU requires top-level UBOE, symmetric L0 and eight devices per module. DPU uses different L2 templates. |
| `ins_v2_allreduce_sequence2die_executor.cc:344–367` | CCU two-die registrations use `TopoMatchOneLevel`, not a generic two-level sequence. Two physical dies do not imply two HCCL hierarchy levels. |

These registrations establish that native three-level algorithms exist. They do
not establish selection by any retained A3/A5 capture or justify relabeling the
implemented route0 as an OmniPipe/CCU/DPU execution.

## Small explicit analytic extension, if needed

A separate `analytic_hierarchy` contract can declare ordered rank dimensions,
per-level RS/AG primitives, per-level protocol and chunk size. Its equal-shard
initial domain should require the rank dimensions' product to equal the actual
communicator and payload divisibility by that product. It need not implement a
second execution engine or a model-specific dispatch branch.

For dimensions `r0,r1,r2` and N input bytes per rank, RS input sizes are N,
N/r0 and N/(r0*r1). Reverse AG inputs are N/(r0*r1*r2), N/(r0*r1), and N/r0.
Each level expands its groups to physical endpoint IDs and emits existing
local-work phases and message flows. Declared group barriers provide the
conservative sequential composition. Cross-level chunk overlap must be explicit;
it cannot be obtained by summing isolated times or replacing each level with its
own full-payload allreduce.

The ordinary network path already provides `hop.demand` and coupled Tx/Rx service.
A transfer therefore consumes link, receiver and declared switch capacities
once, while wire bytes remain one count per traversed hop. A remote signal takes
one propagation per actual hop. A local event wait is only a dependency; adding
RTT to it would double-count synchronization. An unknown signal-frame byte count
stays unpriced in wire work while propagation still runs.

Initial acceptance should include a contribution-set oracle, rank-group coverage,
RS/AG byte conservation, unequal level bandwidth scans, receiver incast and late
readiness. HCCL scratch reservation remains an explicit missing pool accounting
feature; the current hierarchy reports its size but does not claim to enforce a
separate scratch-pool capacity.

## Joint observable service fits are whole-call responses

Reviewed profiler `data/observations/sources/arch3510-physical-services/`
`network-joint-service-regimes.json`, schema1, source baseline SHA256
`9f5fa01a4d099842be53915bebbcf5122bc86de5f6df1493e9dc4ea56fbf9e88`.
No raw capture events or new predictions were needed for this audit.

| Declared domain | Payloads | Fixed us | Effective GB/s | Maximum original-group residual us |
|---|---:|---:|---:|---:|
| A5 AIV AllReduce, R2, OneShot, 3 blocks | 40,960 / 491,520 B | 7.835 | 13.665 | 219.698 |
| A5 AIV AllReduce, R4, TwoShot, 60 blocks | 65,536 / 1,048,576 / 2,097,152 B | 25.625 | 97.489 | 219.582 |
| A5 AIV AllGather, R4, Mesh, 64 blocks | 16,384 / 262,144 / 524,288 B | 11.324 | 26.027 | 2.895 |

The first fit is a two-point secant, not independent shape validation. The other
two are in-sample three-point fits. A5 R2 TwoShot and A3 R8 AG/RS have only one
payload each, so no independent intercept and slope can be identified.

These fits include peer readiness, protocol and local work at the complete
collective-device-task boundary. Their fixed terms are not independently measured
RTT; their slopes are not isolated link rates. In particular, pooled-payload
residuals near zero coexist with >200 us original-role/first-step residuals for
the AllReduce groups. The retained variation must remain visible.

Apply such a response at most once to the matching complete collective kernel
boundary, using the inclusive service response already supported by Sim. Match
hardware, expansion, template, rank count, block limit, dtype and bounded payload
domain. Do not apply it to every phase, hierarchy level, peer or signal. Do not
write these coefficients into `link.bytes_per_s`, switch latency or an A3/A5
controller default. Hardware-scaling assumptions, if used for an effective
service response, must remain distinct from independently measured pipe rates.
