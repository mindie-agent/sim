# MoE initialization: Sort macros and SIMT logical accesses

The selected arch35 `MoeInitRoutingV3` program now exposes separate optional
services for source Sort macros and logical SIMT memory operations. It uses
the existing `compute_arch.instruction_services` interface and the common
Engine. No hardware rate is assigned here.

`source_concat32_blocks`, `source_sort32_blocks`, and
`source_extract32_blocks` each count `ceil(assignments / 32)` input blocks.
Each represents **one whole source macro normalized by its input block count**.
This is not a collection of independent 32-element sorts. In particular, the
caller source does not establish the SDK's merge expansion or a constant
per-block cost across different input sizes. A measured effective capacity
must retain its supported size/implementation domain. Do not substitute these
names for `vbitsort32` or `vmrgsort4v2` instruction counts.

If a macro service is declared, owner0 executes the source order: known
preparation lanes, Concat, Sort, Extract, then known output lanes. Each missing
macro service stays unpriced and visible. The previous elementary Vector
work is split across the boundary without adding or removing lanes. With none
of the macro services declared, the original Stage sequence is unchanged.

For the full-range, dropless GATHER branch, each active owner's `n` assignment
rows generate these source-level INT32 operations:

| Source phase | Logical GM loads | Logical GM stores |
|---|---:|---:|
| First-index and count scans together | `4*n - 2` | 0 |
| Inverse-index permutation | `n` | `n` |

The services `source_simt_gm_load32` and `source_simt_gm_store32` use units
`int32_thread_accesses`. They constrain the same owner as the existing direct
memory phase, alongside its unchanged HBM byte work. They are aggregate
source-access capacity constraints; they do not reconstruct per-thread
load/store dependencies, cache misses, warp issue or coalesced transactions.
The source launch declares2048 threads, but this does not identify compiled
warp instructions. Data-dependent expert-boundary UB accesses remain outside
the inventory because actual sorted IDs are not reconstructed from a histogram.

Count publication remains one `SetAtomicAdd<int32_t>` + `DataCopyPad(E*4)`
per active AIV. Every owner addresses the same count array. No `simt_atomic_rmw`
or speculative128-byte transaction count is charged: exact address alignment,
cache policy and shared-line serialization require their own contract.

Source revision is `ops-transformer`a3925fd49cd256ad478d2faf983bf2961b8ae2c8:

- `moe_v3_sort_one_core_arch35.h:97-113`: Concat/Sort/Extract and their block arguments;
  SHA-256 `920534d82fd33eb308a4b3493e978b598e5bfba1da51f6700794a12197e94709`.
- `moe_v3_expert_tokens_count_arch35.h:83-121,242-256`: the two SIMT scans and atomic DMA;
  SHA-256 `d0dfd8f977596357f37fddd1ea871789657b128e9c0d4bda03badbbd99ce7077`.
- `moe_v3_row_idx_gather_arch35.h:48-57`: the inverse permutation;
  SHA-256 `cfc375ee5bb0a5f9ddb845ba349ea09e3ed64ada80c3e947e6ff3fc25ce6d353`.
- `moe_v3_common.h:33`: launch threads;
  SHA-256 `697f6af1b8c9719adb264c6e03673175b72d752fef558b30bd77086572ae8e21`.

Eight new controls plus the existing scalar/MoE tests passed28 tests. They
cover owner0 macro ordering with0.5/1/2 synthetic capacity, literal independent
thread iteration counts, active-owner placement, unchanged payload, late7ms
input and released buffers. Eight additional comparisons against the literal
previous program (`d8c9d516`) preserve the complete trace, duration, serviced
work and Engine summary when these new services are absent, both with and
without the previously declared scalar services. These are mechanism checks;
they do not establish native operator or Body accuracy.
