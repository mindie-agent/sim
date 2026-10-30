# MXFP8 routing initialization: residual control work

Development-run reports cited below are historical context and are not shipped. Current acceptance is documented in [verification](../verification.md).

This read-only follow-up checks the explicit arch35 source branch already
implemented by `moe_init_mxfp8.py`. It changes no runtime or calibration.

The pinned host chooses one-core Sort when assignment count fits its bounded
sort arena (`moe_init_routing_v3_tiling_arch35.cpp:1318-1330`). The selected
kernel uses Concat → Sort → Extract (`moe_v3_sort_one_core_arch35.h:97-113`);
it is not an unmodeled bucket-sort path. The same source has larger multi-core
sorting branches, but those are outside the current explicit selector's domain.

The retained caller sets `expert_tokens_num_type=1` and declares group-list
count mode (`token_dispatcher.py:398-411`). In
`moe_v3_expert_tokens_count_arch35.h:309-317`, COUNT_MODE casts INT32 counts to
INT64 and sums all expert counts through owner-zero scalar `GetValue` accesses.
CUMSUM_MODE, which writes each cumulative prefix through `SetValue`, is a
separate branch at lines 319-326. Do not add that prefix as extra work to a
count-mode call.

Current plans already retain:

- One-core ID read and two sorted-array writes; known elementary vector work.
- Per-owner two SIMT passes over sorted IDs, atomic-add count payload, and the
  owner-zero count read/cast/publication.
- Inverse row IDs, total-count reads including idle launched owners, and actual
  gather-copy token reuse plus MXFP8 activation/scale publication.

The missing independently priced services are specialized Sort/Concat/Extract
issue, SIMT address/control issue, owner-zero scalar UB loads/add dependencies,
atomic cache RMW, and synchronization latency. For E288, owner zero executes
288 UB INT32 `GetValue` operations and the serial scalar total accumulation
under this mode. These are **not HBM reads**, and the independent BF16 global
scalar-store primitive does not calibrate them. A generic scalar or SIMT service
can be introduced once its units and independent throughput/latency are declared;
a measured entire-kernel residual cannot supply those parameters.

The arithmetic controls
pin the source hashes and derive q4/B16,32,64 work counts. At B64, 2,048
assignments occupy 64 sort32 blocks, 64 count owners with 32 IDs each, 8,064
logical INT32 SIMT loads, and 73,728 bytes of atomic count-store payload. These
are logical source accesses; treating every cache access as a fresh HBM
transaction would need a separate cache transaction model. Existing counts are
not converted into a fitted latency.

There is no newly identified large missing array copy in this branch. The
previous several-microsecond estimate versus tens-of-microseconds observation
remains unresolved source-control timing, with compiled identity still separate
from analytic simulation. No source label or activity total is added to body.
