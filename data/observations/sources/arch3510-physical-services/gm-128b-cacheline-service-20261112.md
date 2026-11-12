# A5 GM 128-byte cacheline physical service

Status: one reusable **HBM line throughput** service is installed. No dependent
scalar-load response latency is claimed.

## Installed

`gm_hbm_cacheline_128b`, unit `cachelines`, owner `vector`, per-owner
`488,281,250 lines/s` (`4e12 B/s / 128 B / 64 AIV`). It is peak HBM line
throughput under an equal per-AIV share. A source must supply the count of
unique HBM 128-byte line transactions; an atomic RMW counts both its read and
write line transactions. It is not a scalar instruction service.

## Why this is the maximum defensible landing

- A5 architecture documents the 128-byte external transaction width and the
  SIMT 128 KiB DataCache. They do **not** document scalar `GetValue` response
  latency, outstanding-request depth, cache hit policy, or L1/L2 service.
- Six retained native captures give only composite AIV scalar/MTE busy slopes.
  The source review explicitly says no isolated GetValue/sort/atomic-RMW
  latency can be identified. There is no PMU target here.
- `pipe_s_issue`/GetSFFValue is a fixed PIPE_S issue service; BF16
  `scalar_store_issue` is a UB-to-GM store queue. Neither is a scalar GM load
  or a dependent pointer chase.
- Therefore `scalar_gm_access` and `scalar_alu_issue` remain unpriced. A
  dependent `index -> page table -> DMA` chain has no installed latency/MLP
  term. The new service only bounds an explicitly counted HBM-line workload;
  it must not be used to claim that a dependent scalar chain is resolved.
