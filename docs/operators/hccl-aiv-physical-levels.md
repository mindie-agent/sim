# AIV over two physical topology levels

The explicit AIV `mesh_oneshot`, `mesh_twoshot` AllReduce and `mesh` AllGather
declarations accept `topology_levels: 2` with `topology: "MESH_1D"`. They retain
one full-peer algorithm. Each peer transfer uses the normal physical route,
including package, node and inter-node capacity. They do not insert an extra
local reduction or an NHR round merely because the hardware has two levels.

The pinned HCCL revision is `c2d6790464654ccf58aca39448de4b65fdc7b184`:

- `all_reduce_auto_selector.cc:606–698` and `all_gather_auto_selector.cc:412–474`
  reject three or more levels, and retain SoleMesh for the supported two-level
  domain.
- `ins_v2_all_reduce_sole_executor.cc:438–483` and
  `ins_v2_all_gather_sole_executor.cc:475–496` register a maximum of two physical
  levels with `TopoMatchOneLevel`.
- `topo_match_one_level.cc:23–104` selects a physical layer whose accessible
  ranks cover the entire group, and builds exactly one algorithm level.

Selecting this declaration asserts full-group peer accessibility. It does not
establish that an arbitrary RDMA cluster, UBoE network or captured native
selector supports this AIV branch. Automatic selection, MultiJetty/CLOS
channels, superkernels and executor repeat segmentation remain separate work.

Nine focused controls run TP8/TP16 AllReduce and AllGather, independently scale
package/node/inter-node links, and verify unchanged local work and exact
directed payload. Tx and Rx capacity charges describe the same wire bytes;
they are not two transmissions. The delayed-input controls finish with no live
tasks, programs or buffer reservations.

These controls use the declared A3 physical links with an explicit 256 KiB UB
capacity to isolate routing. They are not A3 deployment acceptance: this pinned
kernel reserves `190 KiB + 2560 B = 192.5 KiB`, exceeding the registered A3
192 KiB UB. The default capacity still rejects it. A supported A3 binary needs
its actual queue/allocation contract; the test override does not change the
hardware profile or any production case.
