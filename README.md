# `mindie-sim` (`mindie_sim`)

Standalone Ascend NPU 4-layer simulation and theoretical boundary solver for MindIE Agent (`v0.0.3`):

1. **Layer 1 (`hardware` / `sku_profile` / `comm`)**:
   - Multi-SKU `DeviceCapabilityProfile` (`cube_spec`, `vector_spec`, `memory_hierarchy_spec`, `clock_and_topology_spec`) across A2 (`Ascend910B1..B4`), A3 (`Ascend910_93*` dual-die), and A5 (`Ascend950DT`/`950PR`), with 3-tier resolution (`.ini` + SKU registry + external JSON -> live `npu-smi` & `lspci` probe override -> caller override) and safe degradation (`ambiguous_soc_sku_requires_exact_soc_or_ini_or_probe`).
   - Full-path 7-channel interconnect & host-device transfer solver (`d2d_intra_package`, `d2d_intra_node`, `d2d_inter_node`, `h2d`, `d2h`, `rh2d`, `rd2h`), including `pinned` vs `pageable` H2D/D2H, PCIe `LnkSta` downgrade detection, `direct_rdma` vs `host_staged` chunked pipeline for `RH2D`/`RD2H`, and 3-tier hierarchical collective bounds.
   - On-chip buffer (`L1`/`L0A`/`L0B`/`L0C`/`UB`) alignment/overlap validation and sustained thermal clock sensitivity matrices.
2. **Layer 2 (`pipeline`)**:
   - Multi-phase periodic microstage graph expansion (`MTE2`/`MTE1`/`M`/`FIX`/`V`/`MTE3`), Karp maximum cycle-mean solver (`minimum_period_ns`, `minimum_tile_ii_ns`), critical recurrence cycle extraction, and dimensionless topological `--self-check`.
3. **Layer 3 (`subgraph`)**:
   - Upfront Phase 0 producer-consumer subgraph fusion analyzer (`removable_gm_bytes`, hard feasibility gates for collective barriers, `L1`/`UB` residency limits, non-elidable `DCP` normalization, and concurrent critical-path overlap `max(P+C, Q+G)`).
4. **Layer 4 (`net`)**:
   - Multi-dimensional parallel step decomposition (`DP x TP x EP x PP x DCP`, `DSpark` acceptance length, `Prefix Cache` hit rate, `D2D`/`RH2D` transfer overlap) and read-only bridge to pinned `msmodeling` (`tensor_cast`).
