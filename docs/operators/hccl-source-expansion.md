# HCCL source expansion contracts and programs

The source boundary is HCCL `c2d6790464654ccf58aca39448de4b65fdc7b184`.
The local architecture cards pin older commits (notably `88a2a65c...`), so their
API navigation was checked against the current source. This implementation
models an explicitly declared source branch. A kernel name, elapsed time or
hardware SKU alone does not establish the captured executor.

## Executable scope

`hccl_contracts.select` requires an expansion and template. `hccl_aicpu.contract`
produces a per-rank DAG with full directed exchange matrices; `build` executes
each local resource phase on the common Engine. The caller owns link routing,
participant readiness, one invocation boundary and exchange completion. Local
HBM endpoints are charged by these phases; transport must charge link bytes only.

| Explicit AICPU selection | Local phases and wire traffic |
|---|---|
| `allreduce / mesh_oneshot` | Pre-notify; local input→scratch and all-peer writes in parallel; post-notify; R−1 serial local reductions; final scratch→output. Each rank sends `(R−1)N` bytes. |
| `allreduce / mesh_twoshot` | ReduceScatter then AllGather, retaining separate pre/post synchronization and element-aligned tail shards, including empty final ranks. |
| `allgather / mesh` | Copy local contribution to output and scratch before peers read it; remote chunks land directly in output. R−1 threads for one channel per peer. |
| `reduce_scatter / mesh` | Concurrent remote writes; post-sync; copy own input shard to output; R−1 ordered reductions. Public local input contains R equal output shards. R threads. |

`buffer_mode="separate"` means distinct user input/output and HCCL scratch.
`remote_output` is supported for AllGather and removes only its scratch copy.
`symmetric` is supported for TwoShot/ReduceScatter: remote read+reduce modifies
the local input shard, then that result is copied to output. The wire matrix is
unchanged; remote reduction adds the destination read before overwrite and
removes the subsequent scratch fold. Input/output aliasing, multiple channels,
executor segmentation and dynamic scratch capacity are not reconstructed.

Local copy work is `2N` HBM bytes and `N` `hccl_copy` bytes; local reduction is
`3N` HBM bytes and `N` `hccl_reduce` bytes. These services do not borrow AIV
compute or MTE capacity. Hardware may explicitly declare their rates. Missing
rates preserve the HBM lower bound and report `unpriced_service_work`; they do
not prevent simulation. Notify/wait work can use `hccl_control` (calls); without
its rate the counter stays visible and untimed. No body residual enters a rate.

For example, a separate-buffer OneShot rank has HBM work
`[4 + 5(R−1)]N`: two local copies, network source/destination endpoints and
R−1 read-modify-write reductions. This differs from the ring link factor
`2(R−1)N/R`; reusing that factor would erase both bytes and protocol structure.

## Selection and topology findings

The current `op_common.cc:184–187` routes Mesh1D or device960 to
`SelectorEngine`; the old `AllReduceAutoSelector` thresholds are not the entire
dispatch rule. The helper `legacy_mesh_candidate` is only a diagnostic for that
old narrow branch. At ≤8 MiB it suggests AICPU OneShot, but beyond it the current
legacy code also considers `8/R²`, detours and MeshChunk. AIV's old on-board
cutover is 128 KiB. An explicit 491,520-byte/Block3 capture candidate must not be
overridden just because the current old threshold suggests TwoShot.

`SelectorEngine:70–88` preserves expansion fallback priority:
CCU_MS→CCU_SCHED→AICPU_TS→HOSTCPU; AIV→AICPU_TS→HOSTCPU; AIV_ONLY allows AIV only.
The simulator records the declared branch, not an invented successful fallback.

`op_common.cc:3667–3700` gets the A5+ expansion from `HcclConfigGetInfo` when
available. Older devices, or an unavailable configuration interface, use the
environment path. `alg_env_config.cc:876–898` defaults 910_93 to AICPU unfolding
and enables A5 task cache in its AICPU path. Runtime configuration and loaded
binary therefore belong in capture identity.

`topo_host.cc:45–59` treats 910_93 server/superpod topology separately from 950+
rank-graph L0/L1/L2. The latter network levels do not mean server/module levels.
`CalcLevel0MeshType:1178–1240` classifies CCU endpoint die links, and
`TWO_DIE_REGULAR` means the two endpoint-link counts differ by one. It does not
prove an A3 compute-die mapping. `ccu_kernel_utils.h:23–33` selects CCU_V1 for
950 and checks the HCOMM CCU_V2 capability for later paths. These facts do not
authorize automatically installing a CCU program on every A3/A5 profile.

## Additional executable domains and remaining gaps

| Branch | Executable source domain and remaining implementation |
|---|---|
| NHR | `hccl_nhr.py` executes AICPU AllReduce/AllGather/ReduceScatter with the pinned cyclic ±2^step peer recurrence, ordered slice readiness and separate buffers. Ordinary caller selection uses `template="nhr"`. The supported domain is power-of-two ranks, equal integer shards, one non-PCIe channel and one topology level. Uneven shards, symmetric buffers, multiple channels and multilevel composition remain outside this source branch. |
| CCU_MS / CCU_SCHED | `hccl_ccu.py` executes CCU_MS AllReduce `mesh_oneshot_v1/v2` (2–8 ranks, at most 4096 bytes, non-expanding SUM) and CCU_SCHED AllGather `mesh_mem2mem_v1/v2` (equal payloads, one executor repeat). [Two endpoint-die CCU_MS AllGather](hccl-ccu-two-die-source.md) adds explicit homogeneous channel maps, two CCU owners and physical notifications for one at-most-4096-byte fragment. Copy, SRAM, reduction and control retain separate units and explicit rates; no AIV or AICPU service substitution is made. Larger reductions and multiple repeats remain separate domains. |
| DPU | AllReduce selector `:701–737` chooses multi-level Mesh+NHR, including a pipeline branch for non-PCIe Mesh1D+CLOS. Source has preSync, channelFence, threadFence and host-DPU terms. The level hierarchy and channels must be declared before generating the program. |
| AlltoAllV | `hccl_alltoallv.py` executes the whole-cohort separate-buffer AICPU mesh read/write programs with at most 16 peer streams, independent slot reuse and exact zero/one-way peers. `start_participants` retains per-rank readiness. This source program is available through the whole-cohort API; ordinary representative transformer lowering is not yet connected. |
| AIV AllGather | `hccl_aiv_gather.py` executes the small-message control-core and large-message producer/consumer branches, including 95 KiB finite copy slots and successful flag transfers. Ordinary caller selection is `AIV / allgather / mesh` with an explicit block limit. Whole-phase publication currently approximates per-fragment peer readiness; additional polling and scalar synchronization are not assigned invented durations. AIV ReduceScatter remains separate work. |

The HCCL selector's constants are **source heuristics**, not measured hardware
parameters. `cost_model.cc:117–124` has copy1,400GB/s, reduce900GB/s,
CCU copy200GB/s, CCU reduce35GB/s and circular47.6GiB/s. `:587–628` has
2µs AICPU kernel terms and0.5µs task terms. OneShot records15 kernels and
`10(R−1)` tasks; TwoShot20 and`16(R−1)+10`. `cost_table.cc:402–410` ultimately
takes `max(segment costs, total task term)`, rather than blindly adding both.
NHR even labels its4µs/1µs constants empirical. None are injected into hardware
defaults. The GLM cases in `glm-hccl-source-complete-conditional/` explicitly
compare copy/reduce source-cost hypotheses with HBM-only controls; they do not
convert the selector's kernel/task constants into per-notify execution rates.

Independent actual-Engine and canonical-identity reviews are retained in
`docs/operators/structure-and-hccl-independent-review.md`. The GLM complete
body composition and its remaining errors are in
`docs/operators/glm-hccl-source-complete.md`.

## Verification

`tests/test_hccl_aicpu_source.py`:15 tests passed in0.04s. Independent oracles
cover directed-byte conservation, uneven and empty AR shards, buffer-mode
memory changes without wire changes, ordered local reductions, delayed input
readiness, resource monotonicity and zero live tasks. A complete local OneShot
DAG plus link service reproduces a22ms analytical control with simultaneous
input-copy/network stages and serial reductions. This is implementation
verification, not measured communication accuracy or the full-model scoreboard.

Source fingerprints are retained in
`data/verification/hccl-source-contracts-20261030.json`.

## Ordinary caller integration and AIV

`runtime.kernels.hccl` selects each family explicitly, for example:

```json
{"allreduce": {"expansion": "AIV", "template": "mesh_oneshot",
               "block_limit": 3, "axes": ["tp"]}}
```

The caller recovers the input payload from the semantic collective's declared
wire factor, constructs local phases and exact peer exchanges, and retains one
host/device invocation. A source mesh has one concurrent channel per peer; its
window is `R-1`, not `R-1` serial launch delays. Current caller integration covers
uniform intra-replica TP/PCP/ETP AllReduce, AllGather and ReduceScatter. Unsupported
source engines/templates remain explicit errors; the ordinary analytical path
remains available without this selection.

Integer TwoShot shards need not have equal per-rank wire totals. Source lowering
declares `peer_volume_basis="group_mean"` for its scalar link-work ledger, while
the full peer matrix still determines every physical send and receive. Ordinary
per-rank peer declarations retain their existing validation. AIV TwoShot local
work uses rank zero, which owns the largest early tail shard; it is a local
critical-rank representative, not a full-cohort HBM sum. Tiny inputs with zero
senders and nonzero incoming traffic remain executable without padding.

AIV OneShot uses R producers and one reducer. Each producer copies the complete
input (including the self contribution); the reducer copies rank0 then performs
R-1 ordered atomic GM reductions. TwoShot uses R producers and
`floor((blocks-R)/R)*R` consumers; only one consumer group reduces, followed by
parallel gather. The vector ALU is not charged for MTE3 atomic accumulation.
Payload copies keep source 95KiB iterations and a single read/write queue. The
active-buffer model uses the 190KiB copy queue plus 2560B flags, excluding unused
INT64 queues; this is not compiled UB allocation evidence. Ordinary grouped
execution reserves this storage for every launched AIV at invocation admission
and releases it only at whole-kernel retirement. Per-phase programs reuse that
reservation; isolated phase analysis still reserves its own active-owner queue.
This whole-call declaration conservatively keeps the allocation of an idle
TwoShot block that returns early. The selected TwoShot path requires at least2R launched
blocks. INT64 Reduce64 and small-core branches are separate, unimplemented
programs. A3's declared192KiB UB does not fit this source queue; its profile is
not enlarged to force admission.

The retained Qwen TP2 controls use measured BlockNum3/3/63 and **candidate**
payloads inferred from preceding tensors, not captured HCCL arguments:

| Candidate | Source prediction | Observed steady mean |
|---|---:|---:|
| c1 / OneShot | 4.129us | 10.511us |
| c12 / OneShot | 42.420us | 42.095us |
| c96 / TwoShot | 79.691us | 127.436us |

These are conditional program comparisons. First-layer wait tails remain in the
original population, and no duration sets a bandwidth or fixed delay. Current
source-selector thresholds alone would not establish the captured c12 template.
The c96 residual remains material. AICPU controller/peer issue and AIV flag retry
latencies still require independent service characterization.

## Shared buffer lifetime contract

A `device_kernel_group` entry may declare `buffers`, each with `id`, `pool`,
`bytes` per physical pool unit, and explicit `units`. The group acquires the
complete set atomically after all external inputs and earlier stream tails are
ready. Startup and the inclusive completion minimum begin after capacity
admission; all buffers remain live through every phase and that minimum.
Retirement releases them before publishing the group completion event.

`RecipeContext.group_buffers` lists only reservations supplied by an actual
group execution. A phase uses the matching reservation instead of allocating a
second copy. An isolated phase receives an empty scope, so source buffer limits
remain present in `operator_costs`. Buffer identities include invocation and
replica scope; repeated iterations never share live storage by accident.

Actual Engine controls cover a competing stream on the later reducer's UB, a
different free UB owner, delayed admission behind an existing allocation, late
external inputs, completion-floor lifetime, and repeated iteration cleanup.
No traffic, rate, launch parameter or source phase dependency is changed by
this lifetime correction.
