# A3 package links and HCCL endpoint-die algorithms

The registered `Ascend910_9362` topology exposes a 448 GB/s SIO link within
each declared two-rank package and a 196 GB/s HCCS_SW link across the node.
These are retained profile declarations. An explicitly selected AICPU
`parallel_mesh_nhr_route0` schedule can run local Mesh reduce-scatter over SIO,
inter-package NHR, then reverse the gathers over these physical links.
Selecting that schedule does not establish that a captured A3 call used it.

[TP8/TP16 controls](../../data/verification/a3-physical-hierarchy-20261030.json)
use the ordinary body caller and isolate its two AllReduce calls. All twelve
0.5/1/2 bandwidth evaluations complete; both SIO and HCCS_SW bandwidth affect
duration monotonically. Physical wire bytes equal directed payload plus the
explicit four-byte signal declaration. Tx and Rx capacity demands each equal
wire bytes and are never added together as twice the payload. Delayed input
readiness and task/program release remain intact. Controller issue rates are
not calibrated by these synthetic controls.

HCCL's `TWO_DIE_REGULAR` is a separate contract. In pinned source
`c2d6790464654ccf58aca39448de4b65fdc7b184`,
`src/ops/op_common/topo_info/topo_host.cc:1178` classifies each source rank's
remote channels by their **source endpoint** Die ID. Regularity means two
nonempty channel groups whose sizes differ by one. It does not identify pairs
of NPU ranks sharing an SIO link. The CCU_MS AllGather template
`ccu_temp_all_gather_2dies_mesh_1D.cc` launches two channel-group kernels reading
the same local input; only the smaller group handles the local output copy.
CCU_SCHED uses a different mem2mem template. The physical A3 control above
neither claims either CCU implementation nor installs one as a hardware default.
